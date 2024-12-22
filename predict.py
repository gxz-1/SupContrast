import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from rf_dataset import InferenceDataset, SPDataset
from networks.resnet_big import CustomCNN, CustomCNNmini, sp_LinearClassifier

# 设置推理相关参数
class InferenceOptions:
    def __init__(self):
        # self.val_data_folder = '/disk/datasets/rf_data/newspectrum/SelectAB/test'  # 请替换为SP验证数据的路径
        self.val_data_folder = '/disk/datasets/rf_data/newspectrum/dataset/test/哈博森/5.8G,10M,外场,D5900m,H260m/'  # 请替换为SP验证数据的路径
        # self.val_data_folder = '/disk/datasets/rf_data/newspectrum/SelectAB/test/4pro/5.8G,40M,外场,D1900m,H500m/'
        self.encode_ckpt = 'save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_30.pth'  # encode模型的路径
        self.classifier_ckpt = 'save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/best_classifier_96.920.pth'  # classifier模型的路径
        self.batch_size = 32
        self.num_workers = 8
        self.model = 'CustomCNNmini'  # 这里设置为'CustomCNN'，如果使用其他模型请修改
        # self.mode = 'predict'
        self.mode = 'data'

# 2. 设置模型
def set_model_for_inference(opt):
    # 加载特征提取部分（encode）模型
    if opt.model == 'CustomCNN':
        model = CustomCNN()
    elif opt.model == 'CustomCNNmini':
        model = CustomCNNmini()
    else:
        raise ValueError(f"Unsupported model: {opt.model}")
    
    # 加载分类器部分
    classifier = sp_LinearClassifier(num_classes=5)  # 对于sp数据集，类别数是5
    
    # 从硬盘加载特征提取模型权重（encode部分）
    encode_ckpt = torch.load(opt.encode_ckpt, map_location='cpu')
    state_dict = encode_ckpt['model']
    
    # 处理 "module." 前缀问题：去除所有键中的 "module."
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 加载模型权重
    model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 允许一些键缺失
    
    # 从硬盘加载分类器模型权重
    classifier_ckpt = torch.load(opt.classifier_ckpt, map_location='cpu')
    classifier.load_state_dict(classifier_ckpt)  # 假设'classifier'部分的权重在'classifier_ckpt'中

    # 使用GPU进行推理
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        model = torch.nn.DataParallel(model)  # 使用多GPU时
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier

# 3. 执行推理
def inference(val_loader, model, classifier):
    model.eval()  # 设置为评估模式
    classifier.eval()  # 设置为评估模式
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # 提取特征（encode）
            features = model(images)  # 使用整个模型提取特征
            # 分类器进行预测
            outputs = classifier(features)  # 分类器做最终预测
            
            _, preds = torch.max(outputs, 1)  # 获取最大概率的类别
            all_preds.append(preds.cpu().numpy())  # 将预测结果转换为cpu上的numpy数组
            all_labels.append(labels.cpu().numpy())  # 将标签转换为cpu上的numpy数组

    # 拼接所有的预测值和标签
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算准确率
    accuracy = (all_preds == all_labels).mean()
    print(f'Inference Accuracy: {accuracy * 100:.2f}%')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # 计算每个类别的准确率
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc * 100:.2f}%")

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


import torch

def predict(val_loader, model, classifier):
    model.eval()       # 设置为评估模式
    classifier.eval() # 设置为评估模式

    # 类别名称映射
    class_names = ["3pro", "4pro", "Dji", "dao", "ha"]

    # 初始化一个字典来统计每个类别的预测数量
    prediction_counts = {class_name: 0 for class_name in class_names}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)            

            # 提取特征（encode）
            features = model(images)  # 使用整个模型提取特征

            # 分类器进行预测
            outputs = classifier(features)  # 分类器做最终预测

            # 获取每个样本的预测类别索引
            _, preds = torch.max(outputs, 1)  

            # 遍历每个预测结果
            for pred in preds:
                class_index = pred.item()
                if 0 <= class_index < len(class_names):
                    predicted_class = class_names[class_index]
                    prediction_counts[predicted_class] += 1
                else:
                    print(f"Warning: Predicted class index {class_index} is out of range.")

            # 输出每张图片的预测类别（可选，如果数据量大可以省略）
            # for i, pred in enumerate(preds):
            #     print(f"Batch {batch_idx + 1}, Image {i + 1} predicted as: {class_names[pred.item()]}")

    # 打印每个类别的预测数量
    print("\n预测结果统计：")
    for class_name, count in prediction_counts.items():
        print(f"类别 '{class_name}' 的预测数量: {count}")



opt = InferenceOptions()
# 数据预处理
val_transform = transforms.Compose([
    transforms.RandomCrop((500, 500)),
    # transforms.RandomResizedCrop((500, 500), scale=(0.9, 1.1)),  # 随机缩放裁剪，裁剪比例可调
    # transforms.CenterCrop((500, 500)),
    # transforms.Resize((500,500)),
    transforms.ToTensor(),
])

if opt.mode =='data':
    val_dataset = SPDataset(data_dir=opt.val_data_folder, transform=val_transform, data_type='test')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    model, classifier = set_model_for_inference(opt)
    inference(val_loader, model, classifier)
elif opt.mode == 'predict':
    val_dataset = InferenceDataset(data_dir=opt.val_data_folder, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    model, classifier = set_model_for_inference(opt)
    predict(val_loader, model, classifier)