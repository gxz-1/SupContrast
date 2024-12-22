import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from rf_dataset import SPDataset

# 假设 SPDataset 已正确实现，并返回灰度图像
train_dataset = SPDataset(data_dir='/disk/datasets/rf_data/newspectrum/SelectAB/train', transform=transforms.ToTensor(), data_type='test')

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

# 初始化均值和标准差
mean = 0.0
std = 0.0
num_samples = 0

# 遍历数据集
for data, _ in train_loader:
    # data shape: (batch_size, channels, height, width)
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # (batch_size, channels, height*width)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    num_samples += batch_samples

# 计算最终均值和标准差
mean /= num_samples
std /= num_samples

print(f"Mean: {mean}")
print(f"Std: {std}")
