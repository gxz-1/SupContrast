import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import Subset
from rf_dataset import SPDataset  # 确保该模块在您的工作目录中
from networks.resnet_big import CustomCNN, CustomCNNmini, CustomCNNminidrop  # 确保该模块在您的工作目录中

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import math
import random

def parse_option():
    parser = argparse.ArgumentParser('Argument for t-SNE visualization')

    # 模型和数据路径
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the pre-trained model checkpoint')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to the custom dataset')
    parser.add_argument('--val_data_folder', type=str, required=True,
                        help='Path to the custom validation dataset')

    # 超参数
    parser.add_argument('--batch_size', type=int, default=64,  # 减小批量大小
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=8,  # 减少工作线程
                        help='Number of workers for data loading')

    # 可视化参数
    parser.add_argument('--perplexity', type=int, default=5,  # 调低perplexity以适应小样本量
                        help='Perplexity parameter for t-SNE')
    parser.add_argument('--n_iter', type=int, default=1000,
                        help='Number of iterations for t-SNE')
    parser.add_argument('--save_dir', type=str, default='save/tSNE',
                        help='Directory to save the t-SNE plots')
    parser.add_argument('--feature_type', type=str, default='all',help='获取encode部分或者整个模型all的特征')
    parser.add_argument('--model', type=str, default='CustomCNNmini',help='选择CNN模型')
    args = parser.parse_args()
    return args

def set_loader(opt, subset_indices=None):
    # 数据预处理
    transform = transforms.Compose([
        transforms.CenterCrop((500, 500)),
        transforms.ToTensor()
    ])

    # 加载sp数据集
    dataset = SPDataset(data_dir=opt.val_data_folder, transform=transform, data_type='test')

    # 如果提供了子集索引，则创建子集
    if subset_indices is not None:
        dataset = Subset(dataset, subset_indices)

    # 创建DataLoader，禁用打乱
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # 禁用打乱
        num_workers=opt.num_workers,
        pin_memory=True
    )

    return val_loader

def set_model(opt, device):
    # 初始化模型
    if opt.model == 'CustomCNNmini':
        model = CustomCNNmini()    
    elif opt.model == 'CustomCNN':
        model = CustomCNN()
    elif opt.model == 'CustomCNNminidrop':
        model = CustomCNNminidrop()
    else:
        print("没找到模型{}".format(opt.model))

    # 加载预训练模型权重
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    # 打印state_dict键名以调试
    print("Checkpoint state_dict keys (示例):")
    for k in list(state_dict.keys())[:5]:
        print(k)

    # 调整state_dict的键名，去除'encoder.module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.module."):
            # 替换 'encoder.module.' 为 'encoder.'
            new_k = k.replace("encoder.module.", "encoder.")
            new_state_dict[new_k] = v
        elif k.startswith("encoder."):
            # 不需要修改
            new_state_dict[k] = v
        else:
            # 如果有其他模块的键名，可以根据需要处理
            new_state_dict[k] = v

    # 打印调整后的键名以验证
    print("Adjusted state_dict keys (示例):")
    for k in list(new_state_dict.keys())[:5]:
        print(k)

    # 加载调整后的state_dict
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print("加载state_dict时出错：", e)
        print("检查state_dict的键名是否正确。")
        raise e

    # 将模型移动到指定设备
    if torch.cuda.is_available():
        model = model.to(device)
        cudnn.benchmark = True
    else:
        print("警告: 使用CPU进行计算，可能会非常缓慢。")

    model.eval()
    return model

def extract_features_subset(val_loader, model, device, feature_type):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            if feature_type == 'all':
                features = model.forward(images) 
            elif feature_type == 'encode':
                # 获取经过编码器处理后的特征，并展平
                features = model.encoder(images) 
                features = features.view(features.size(0), -1)
            else:
                print("参数feature_type没找到：{}".format(feature_type))

            features = features.cpu().numpy()
            labels = labels.cpu().numpy()
            # 可选：打印特征某一列值，检查结果
            print(features[:,23])
            print(labels)

            all_features.append(features)
            all_labels.append(labels)

            # 释放显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 打印特征统计信息
    print(f"所有特征的形状: {all_features.shape}")
    print(f"特征的最小值: {all_features.min()}, 最大值: {all_features.max()}")
    print(f"特征的平均值: {all_features.mean()}, 标准差: {all_features.std()}")

    return all_features, all_labels

def visualize_tsne(features, labels, num_classes, save_dir, batch_num):
    """
    使用t-SNE将高维特征降维到2维，并生成散点图
    :param features: N x D 的numpy数组，D为特征维度（例如64）
    :param labels: N的numpy数组，表示类别标签
    :param num_classes: 类别数量
    :param save_dir: 保存图像的目录
    :param batch_num: 当前批次编号，用于命名文件
    """
    print(f"开始t-SNE降维，批次 {batch_num}...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        n_iter=1000,
        init='pca',          # 显式设置初始化方式
        learning_rate='auto' # 显式设置学习率
    )
    try:
        features_2d = tsne.fit_transform(features)
    except ValueError as e:
        print(f"t-SNE降维时出错：{e}")
        print("检查特征数组的形状和数值是否有效。")
        return
    print(f"t-SNE降维完成，批次 {batch_num}。")

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 定义颜色映射
    cmap = plt.get_cmap('tab10')  # 适用于最多10个类别
    colors = [cmap(i) for i in range(num_classes)]

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    class_name=["3pro","4pro","Dji","dao","ha"]
    for class_idx in range(num_classes):
        idxs = labels == class_idx
        plt.scatter(
            features_2d[idxs, 0],
            features_2d[idxs, 1],
            c=[colors[class_idx]],
            label=class_name[class_idx],
            alpha=0.6,
            s=10
        )

    plt.title(f"t-SNE Visualization of SP Dataset Features (Batch {batch_num})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Classes")
    plt.grid(True)

    # 保存图像
    save_path = os.path.join(save_dir, f'tsne_batch_{batch_num}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE散点图已保存到{save_path}")

def main():
    # 解析命令行参数
    opt = parse_option()

    # 设置设备
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            device = torch.device('cuda:1')
            print("使用GPU 1")
        else:
            device = torch.device('cuda:0')
            print("使用GPU 0")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    # 加载模型
    model = set_model(opt, device)
    print("模型加载完成。")

    # 加载完整的数据集以获取总样本数
    transform = transforms.Compose([
        transforms.CenterCrop((500, 500)),
        transforms.ToTensor()
    ])
    full_dataset = SPDataset(data_dir=opt.val_data_folder, transform=transform, data_type='test')
    total_dataset_size = len(full_dataset)
    print(f"验证数据集总样本数: {total_dataset_size}")

    # 设置生成图像的次数和每次的样本数
    num_plots = 4  # 您希望生成的图像数量
    total_samples = 1000  # 每次生成图像时抽取的样本数

    for plot_num in range(1, num_plots + 1):
        print(f"\n正在生成第{plot_num}张t-SNE散点图...")

        # 随机选择total_samples个索引
        if total_samples > total_dataset_size:
            selected_indices = list(range(total_dataset_size))
            print(f"警告: total_samples ({total_samples}) 大于数据集大小 ({total_dataset_size})，将选择所有样本。")
        else:
            selected_indices = random.sample(range(total_dataset_size), total_samples)

        # 创建一个子集数据集
        subset_loader = set_loader(opt, subset_indices=selected_indices)

        # 提取特征和标签
        features, labels = extract_features_subset(subset_loader, model, device, opt.feature_type)
        print(f"已选择 {features.shape[0]} 个样本进行t-SNE可视化。")

        # 确定类别数量
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        print(f"当前抽取样本的类别数量: {num_classes}")

        variances = np.var(features, axis=0)
        if np.any(variances == 0):
            zero_var_dims = np.where(variances == 0)[0]
            print(f"警告: 特征数组的以下维度方差为零，可能导致PCA初始化问题：{zero_var_dims}")
            # 选择方差不为零的特征维度
            features = features[:, variances > 0]
            print(f"已移除零方差的特征维度，新的特征形状为{features.shape}")

        # 检查是否有足够的特征维度
        if features.shape[1] < 2:
            print(f"错误: 特征维度过低（{features.shape[1]}），无法进行t-SNE。")
            continue

        # 可视化t-SNE
        visualize_tsne(features, labels, num_classes, opt.save_dir, plot_num)

        # 释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nt-SNE散点图生成完成。")

if __name__ == '__main__':
    main()
