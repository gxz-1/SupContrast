import os
import torch
import numpy as np
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from rf_dataset import SPDataset
from networks.resnet_big import CustomCNN, CustomCNNmini, CustomCNNminidrop

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')
    
    parser.add_argument('--model_path', type=str, default='save/ckpt_epoch_140.pth', 
                        help='path to the trained model')
    parser.add_argument('--data_folder', type=str, default=r'E:\RFcode\rf数据集\UAVSet\test',
                        help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for data loader')
    parser.add_argument('--model', type=str, default='CustomCNNmini',
                        help='model type')
    parser.add_argument('--dataset', type=str, default='sp', choices=['rf','sp'], 
                        help='dataset type')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='number of sample pairs for distance calculation')
    
    opt = parser.parse_args()
    
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    
    return opt


def set_model(opt):
    # 构建模型
    if opt.dataset == 'sp':
        if opt.model == 'CustomCNN':
            model = CustomCNN()
        elif opt.model == 'CustomCNNmini':
            model = CustomCNNmini()
        elif opt.model == 'CustomCNNminidrop':
            model = CustomCNNminidrop()
        else:
            print(f"没找到模型{opt.model}")
            return None

    # 加载模型权重
    checkpoint = torch.load(opt.model_path)

    # 解决参数名称中包含'module.'前缀的问题
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除所有位置的'module.'前缀
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v

    # 加载调整后的状态字典
    model.load_state_dict(new_state_dict)

    # 设置为评估模式
    model.eval()

    # 移动到GPU
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model

def set_loader(opt):
    # 定义测试数据变换
    test_transform = transforms.Compose([
        transforms.Resize((500, 500)),  # 与训练时保持一致
        transforms.ToTensor(),
    ])
    
    # 创建测试数据集和数据加载器
    if opt.dataset == 'sp':
        test_dataset = SPDataset(data_dir=opt.data_folder, transform=test_transform, data_type='test')
    
    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    return test_loader, test_dataset.class_to_idx

def extract_features(model, test_loader):
    features = []
    labels = []
    
    with torch.no_grad():
        for images, lbls in test_loader:
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            # 提取特征
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    
    # 合并所有特征和标签
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels

def calculate_distances(features, labels, num_pairs=1000):
    # 按类别分组特征
    class_features = {}
    for feat, lbl in zip(features, labels):
        if lbl not in class_features:
            class_features[lbl] = []
        class_features[lbl].append(feat)
    
    # 转换为numpy数组
    for cls in class_features:
        class_features[cls] = np.array(class_features[cls])
    
    classes = list(class_features.keys())
    num_classes = len(classes)
    
    # 计算类内距离
    intra_distances = []
    for cls in classes:
        feats = class_features[cls]
        if len(feats) < 2:
            continue
            
        # 随机采样num_pairs个样本对
        indices = np.random.choice(len(feats), size=(num_pairs, 2), replace=True)
        pairs = feats[indices]
        
        # 计算欧氏距离
        dists = np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=1)
        intra_distances.extend(dists)
    
    # 计算类间距离
    inter_distances = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            feats_i = class_features[classes[i]]
            feats_j = class_features[classes[j]]
            
            if len(feats_i) == 0 or len(feats_j) == 0:
                continue
                
            # 随机采样num_pairs个样本对
            indices_i = np.random.choice(len(feats_i), size=num_pairs, replace=True)
            indices_j = np.random.choice(len(feats_j), size=num_pairs, replace=True)
            
            pairs_i = feats_i[indices_i]
            pairs_j = feats_j[indices_j]
            
            # 计算欧氏距离
            dists = np.linalg.norm(pairs_i - pairs_j, axis=1)
            inter_distances.extend(dists)
    
    return {
        'intra_class': np.array(intra_distances),
        'inter_class': np.array(inter_distances)
    }

def calculate_knn_accuracy(features, labels, k=1):
    # 使用k近邻分类器
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    
    # 预测并计算准确率
    predictions = knn.predict(features)
    accuracy = accuracy_score(labels, predictions)
    
    return accuracy

def main():
    opt = parse_option()
    
    # 加载模型
    model = set_model(opt)
    if model is None:
        return
    
    # 准备数据加载器
    test_loader, class_to_idx = set_loader(opt)
    print(f"Number of classes: {len(class_to_idx)}")
    
    # 提取特征
    print("Extracting features...")
    features, labels = extract_features(model, test_loader)
    print(f"Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    # 计算距离
    print("Calculating distances...")
    distances = calculate_distances(features, labels, opt.num_pairs)

    # 计算最近邻准确率
    print("Calculating KNN accuracy...")
    # 计算k=1和k=5两种情况下的准确率
    knn_accuracy_k1 = calculate_knn_accuracy(features, labels, k=1)
    knn_accuracy_k3 = calculate_knn_accuracy(features, labels, k=3)
    knn_accuracy_k5 = calculate_knn_accuracy(features, labels, k=5)

    # 打印结果
    print("\nEvaluation Results:")
    print(
        f"Intra-class distance - Mean: {np.mean(distances['intra_class']):.4f}, Std: {np.std(distances['intra_class']):.4f}")
    print(
        f"Inter-class distance - Mean: {np.mean(distances['inter_class']):.4f}, Std: {np.std(distances['inter_class']):.4f}")
    print(f"KNN (k=1) Accuracy: {knn_accuracy_k1:.4f}")
    # 添加k=5的KNN准确率输出
    print(f"KNN (k=3) Accuracy: {knn_accuracy_k3:.4f}")
    print(f"KNN (k=5) Accuracy: {knn_accuracy_k5:.4f}")


if __name__ == '__main__':
    main()
