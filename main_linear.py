from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

# from main_ce import set_loader
from rf_dataset import SPDataset
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import CustomCNN, CustomCNNmini, SupConResNet, LinearClassifier, sp_LinearClassifier, sp_MLPClassifier
from torchvision import transforms, datasets
# try:
#     import apex
#     from apex import amp, optimizers
# except ImportError:
#     pass
from torch.cuda import amp

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=256,
                    help='test_batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='CustomCNN')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100','rf','sp'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    # set the path according to the environment
    # opt.data_folder = './datasets/'
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--val_data_folder', type=str, default=None, help='path to custom val dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')

    parser.add_argument('--classifier',type=str, default='linear')
    opt = parser.parse_args()


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'rf':
        opt.n_cls = 4
    elif opt.dataset == 'sp':
        opt.n_cls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'rf':
        mean = eval(opt.mean)
        std = eval(opt.std)
    elif opt.dataset == 'sp':
        mean = 0
        std = 0
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.dataset == 'sp':
        train_transform = transforms.Compose([
        transforms.CenterCrop((500,500)),
        transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if opt.dataset == 'sp':
        val_transform = transforms.Compose([
        transforms.CenterCrop((500,500)),
        transforms.ToTensor()
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    # elif opt.dataset == 'rf':
    #     train_dataset=RFDataset(data_dir=opt.data_folder, train_transform=train_transform) 
    #     val_dataset=RFDataset(data_dir=opt.val_data_folder, train_transform=val_transform)
    elif opt.dataset == 'sp':
        train_dataset=SPDataset(data_dir=opt.data_folder,transform=train_transform,data_type='test')      
        val_dataset=SPDataset(data_dir=opt.val_data_folder,transform=val_transform,data_type='test')      
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.test_batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    #预训练的encode的模型
    if opt.dataset == 'sp':
        if opt.model=='CustomCNN':
            model = CustomCNN()
        elif opt.model=='CustomCNNmini':
            model = CustomCNNmini()
        else:
            print("没找到模型{}".format(opt.model))
    else:
        model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()
    #设置分类的模型
    if opt.dataset == 'sp':
        if opt.classifier == 'linear':
            classifier = sp_LinearClassifier(num_classes=opt.n_cls)
        elif opt.classifier == 'MLP':
            classifier = sp_MLPClassifier(num_classes=opt.n_cls)
        else:
            print("没找到分类器{}".format(opt.model))
    else:
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    # print(opt.n_cls)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())

        # with torch.no_grad():
        #     features = model(images)  # 使用整个模型提取特征，不计算梯度
        # output = classifier(features)
        
        loss = criterion(output, labels)
        # print(output.shape)

        # update metric
        losses.update(loss.item(), bsz)
        # acc1, acc5 = accuracy(output, labels, topk=(1, 2))
        acc1= accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            # print('Train: [{0}][{1}/{2}]\t'
            #       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #        epoch, idx + 1, len(train_loader), batch_time=batch_time,
            #        data_time=data_time, loss=losses, top1=top1))
            print('Train: [{0}][{1}/{2}]\t'
            'BT {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
            'DT {data_time_val:.3f} ({data_time_avg:.3f})\t'
            'loss {loss_val:.3f} ({loss_avg:.3f})\t'
            'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
            epoch, idx + 1, len(train_loader),
            batch_time_val=batch_time.val.item() if isinstance(batch_time.val, torch.Tensor) else batch_time.val,
            batch_time_avg=batch_time.avg.item() if isinstance(batch_time.avg, torch.Tensor) else batch_time.avg,
            data_time_val=data_time.val.item() if isinstance(data_time.val, torch.Tensor) else data_time.val,
            data_time_avg=data_time.avg.item() if isinstance(data_time.avg, torch.Tensor) else data_time.avg,
            loss_val=losses.val.item() if isinstance(losses.val, torch.Tensor) else losses.val,
            loss_avg=losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg,
            top1_val=top1.val.item() if isinstance(top1.val, torch.Tensor) else top1.val,
            top1_avg=top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg))

            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            # features = model(images)
            # output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                # print('Test: [{0}/{1}]\t'
                #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                #        idx, len(val_loader), batch_time=batch_time,
                #        loss=losses, top1=top1))
                print('Test: [{0}/{1}]\t'
                'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                idx, len(val_loader),
                batch_time_val=batch_time.val.item() if isinstance(batch_time.val, torch.Tensor) else batch_time.val,
                batch_time_avg=batch_time.avg.item() if isinstance(batch_time.avg, torch.Tensor) else batch_time.avg,
                loss_val=losses.val.item() if isinstance(losses.val, torch.Tensor) else losses.val,
                loss_avg=losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg,
                top1_val=top1.val.item() if isinstance(top1.val, torch.Tensor) else top1.val,
                top1_avg=top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg))


    print(' * Acc@1 {top1_avg:.3f}'.format(
        top1_avg=top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # 创建保存模型的目录（可选）
    save_dir = 'save/SecondStage/sp_models'
    os.makedirs(save_dir, exist_ok=True)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc.item()))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, 'best_classifier.pth')
            torch.save(classifier.state_dict(), save_path)
            print(f'epoch {epoch}: Best model saved to {save_path}')

    print('best accuracy: {:.2f}'.format(best_acc.item()))


if __name__ == '__main__':
    main()
