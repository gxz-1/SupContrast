import argparse
import torch
from torchvision import transforms
from networks.resnet_big import CustomCNNmini, sp_LinearClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from rf_dataset import InferenceDataset


def parse_option():
    parser = argparse.ArgumentParser('argument for inference')
    
    parser.add_argument('--encoder_path', type=str, required=True, help='path to one-stage trained encoder pth')
    parser.add_argument('--classifier_path', type=str, required=True, help='path to two-stage trained classifier pth')
    parser.add_argument('--image_dir', type=str, required=True, help='path to image directory')
    parser.add_argument('--mode', type=str, choices=['data', 'result'], required=True, help='inference mode: data or result')
    
    opt = parser.parse_args()
    return opt

def load_models(opt):
    encoder = CustomCNNmini(feat_dim=64)
    classifier = sp_LinearClassifier(num_classes=5, feat_dim=64)  # 根据您的类别数调整

    # 加载编码器的检查点
    encoder_checkpoint = torch.load(opt.encoder_path, map_location='cpu')
    
    # 提取 'model' 部分
    if isinstance(encoder_checkpoint, dict):
        if 'model' in encoder_checkpoint:
            state_dict = encoder_checkpoint['model']
        else:
            state_dict = encoder_checkpoint
    else:
        state_dict = encoder_checkpoint

    # 替换 'encoder.module.' 为 'encoder.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.module.'):
            k = k.replace('encoder.module.', 'encoder.')
        new_state_dict[k] = v

    # 加载修改后的 state_dict
    encoder.load_state_dict(new_state_dict)

    # 加载分类器的 state_dict
    classifier.load_state_dict(torch.load(opt.classifier_path, map_location='cpu'))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    encoder.eval()
    classifier.eval()

    return encoder, classifier, device




def get_transforms():
    transform = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.0], std=[1.0])  # 根据训练时的归一化参数调整
    ])
    return transform


def inference(opt, encoder, classifier, device):
    transform = get_transforms()
    dataset = InferenceDataset(opt.image_dir, opt.mode, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    class_names = ["3pro","4pro","Dji","dao","ha"] # TODO:根据数据集的情况进行修改

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = encoder(images)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            if opt.mode == 'data':
                all_labels.extend(labels.numpy())

    if opt.mode == 'result':
        # 输出每张图片的预测类别
        for idx, (img_path, _) in enumerate(dataset.samples):
            pred_idx = all_preds[idx]
            pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
            print(f"Image: {img_path} --> Predicted Class: {pred_class}")
    elif opt.mode == 'data':
        # 计算混淆矩阵和准确率
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds) * 100
        print(f"Accuracy: {acc:.2f}%")

        # 保存混淆矩阵为图片
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('figures/confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")

def main():
    opt = parse_option()
    # 加载模型
    encoder, classifier, device = load_models(opt)
    # 执行推理
    inference(opt, encoder, classifier, device)

if __name__ == '__main__':
    main()

