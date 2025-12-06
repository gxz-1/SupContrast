# -*- coding: utf-8 -*-
"""
@Time ： 2025/12/6 23:31
@Auth ： 高夕茁
@File ：new_predict.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torchvision import transforms
from rf_dataset import InferenceDataset, SPDataset
from networks.resnet_big import CustomCNN, CustomCNNmini, sp_LinearClassifier, sp_MLPClassifier
import pandas as pd  # 添加pandas用于保存CSV

class InferenceOptions:
    def __init__(self):
        # self.val_data_folder = '/disk/datasets/rf_data/newspectrum/UAV/secondUAVSet/test'
        # self.encode_ckpt = 'save/newSupCon/sp_models/tranSupCon_sp_CustomCNNmini_lr_0.05_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth'
        # self.classifier_ckpt = 'save/SecondStage/sp_models/new_best_classifier_93.73.pth'
        # self.batch_size = 32
        # self.num_workers = 8
        # self.model = 'CustomCNNmini'
        # self.classifier = 'MLP'
        # self.mode = 'data'

        # 迁移的secondUAVSet数据集的测试参数
        # self.val_data_folder = r'E:\RFcode\rf数据集\secondUAVSet\test'
        # self.encode_ckpt = 'save/ckpt_epoch_140.pth'
        # self.classifier_ckpt = 'save/SecondStage/sp_models/new_best_classifier_93.73.pth'
        # self.batch_size = 8
        # self.num_workers = 2
        # self.model = 'CustomCNNmini'
        # self.classifier = 'MLP'
        # self.mode = 'data'

        # UAVSet数据集的测试参数 CustomCNNmini_MLP
        self.val_data_folder = r'E:\RFcode\rf数据集\UAVSet\test'
        self.encode_ckpt = 'save/ckpt_epoch_140.pth'
        self.classifier_ckpt = "save/SecondStage/mini_mlp_supcon/best_classifier.pth"
        self.batch_size = 16
        self.num_workers = 2
        self.model = 'CustomCNNmini'
        self.classifier = 'MLP'
        self.mode = 'data'

def set_model_for_inference(opt):
    if opt.model == 'CustomCNN':
        model = CustomCNN()
    elif opt.model == 'CustomCNNmini':
        model = CustomCNNmini()
    else:
        raise ValueError(f"Unsupported model: {opt.model}")

    if opt.classifier == 'linear':
        classifier = sp_LinearClassifier(num_classes=5)
    elif opt.classifier == 'MLP':
        classifier = sp_MLPClassifier(num_classes=5)
    else:
        raise ValueError(f"Unsupported classifier: {opt.classifier}")

    encode_ckpt = torch.load(opt.encode_ckpt, map_location='cpu')
    state_dict = encode_ckpt['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    classifier_ckpt = torch.load(opt.classifier_ckpt, map_location='cpu')
    if 'classifier_state_dict' in classifier_ckpt:
        classifier.load_state_dict(classifier_ckpt['classifier_state_dict'])
    elif 'model_state_dict' in classifier_ckpt:
        classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    else:
        # 如果检查点直接就是分类器的状态字典
        classifier.load_state_dict(classifier_ckpt)

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier

def inference(val_loader, model, classifier):
    model.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            features = model.encoder(images)
            outputs = classifier(features)

            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = (all_preds == all_labels).mean()
    print(f'Inference Accuracy: {accuracy * 100:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    save_path = os.path.join('figures', 'confusion_matrix.png')

    plt.savefig(save_path)
    plt.close()

    print(f'Confusion matrix saved to {save_path}')


def predict(val_loader, model, classifier):
    model.eval()
    classifier.eval()

    class_names = ["UAV1", "UAV2", "UAV3", "UAV4", "UAV5"]
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):  # 获取标签
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            features = model.encoder(images)
            outputs = classifier(features)

            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 合并所有预测结果和标签
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算总体指标
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    # 计算每个类别的指标
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    class_accuracy = []
    for i in range(len(class_names)):
        class_mask = all_labels == i
        if class_mask.sum() == 0:
            class_accuracy.append(0.0)
        else:
            class_accuracy.append((all_preds[class_mask] == all_labels[class_mask]).mean())

    # 准备保存数据
    metrics_data = [
        {
            'Class': 'Overall',
            'Accuracy': overall_accuracy,
            'Precision': overall_precision,
            'Recall': overall_recall,
            'F1 Score': overall_f1
        }
    ]

    for i, class_name in enumerate(class_names):
        metrics_data.append({
            'Class': class_name,
            'Accuracy': class_accuracy[i],
            'Precision': class_precision[i],
            'Recall': class_recall[i],
            'F1 Score': class_f1[i]
        })

    # 保存到CSV
    df = pd.DataFrame(metrics_data)
    csv_path = 'evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)

    # 打印结果
    print("\n总体评估指标:")
    print(f"准确率: {overall_accuracy * 100:.2f}%")
    print(f"精确率: {overall_precision * 100:.2f}%")
    print(f"召回率: {overall_recall * 100:.2f}%")
    print(f"F1值: {overall_f1 * 100:.2f}%")

    print("\n每个类别的评估指标:")
    for i, class_name in enumerate(class_names):
        print(f"\n类别 {class_name}:")
        print(f"准确率: {class_accuracy[i] * 100:.2f}%")
        print(f"精确率: {class_precision[i] * 100:.2f}%")
        print(f"召回率: {class_recall[i] * 100:.2f}%")
        print(f"F1值: {class_f1[i] * 100:.2f}%")

    print(f"\n评估指标已保存到 {csv_path}")


if __name__ == '__main__':
    opt = InferenceOptions()
    val_transform = transforms.Compose([
        transforms.CenterCrop((500, 500)),
        transforms.ToTensor(),
    ])

    if opt.mode == 'data':
        val_dataset = SPDataset(data_dir=opt.val_data_folder, transform=val_transform, data_type='test')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True
        )
        model, classifier = set_model_for_inference(opt)
        inference(val_loader, model, classifier)
    elif opt.mode == 'predict':
        # 使用SPDataset代替InferenceDataset以获取标签
        val_dataset = SPDataset(data_dir=opt.val_data_folder, transform=val_transform, data_type='test')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True
        )
        model, classifier = set_model_for_inference(opt)
        predict(val_loader, model, classifier)
