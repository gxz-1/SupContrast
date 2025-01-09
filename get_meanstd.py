import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from rf_dataset import SPDataset

# ========== 1. 创建数据集和 DataLoader ==========
train_dataset = SPDataset(
    data_dir='/disk/datasets/rf_data/newspectrum/SelectAB/train',
    transform=transforms.ToTensor(),
    data_type='test'
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

# ========== 2. 计算全局(按图像均值再平均)的 mean 和 std ==========
mean = 0.0
std = 0.0
num_samples = 0

for data, _ in train_loader:
    # data shape: (batch_size, channels, height, width)
    batch_samples = data.size(0)
    # 将 (B, C, H, W) reshape 为 (B, C, H*W), 方便计算 mean / std
    data = data.view(batch_samples, data.size(1), -1)

    mean += data.mean(dim=2).sum(dim=0)
    std  += data.std(dim=2).sum(dim=0)
    num_samples += batch_samples

mean /= num_samples
std /= num_samples

print(f"Mean: {mean}")
print(f"Std:  {std}")

# ========== 3. 从整个 Dataset 随机抽若干张图进行可视化并保存 ==========
num_to_show = 5   # 你想要随机查看的图像数量
random_indices = random.sample(range(len(train_dataset)), num_to_show)

os.makedirs('visual_random', exist_ok=True)

for i, idx in enumerate(random_indices):
    # 从 Dataset 中按索引取出 (img_tensor, label)
    img_tensor, label = train_dataset[idx]  
    
    # 这里假设是灰度图，形状 (1, H, W)；如果是 RGB 则形状 (3, H, W)
    # 使用 to_pil_image 将张量转换为 PIL Image
    pil_img = to_pil_image(img_tensor)

    # 保存
    save_path = os.path.join('visual_random', f"random_idx_{idx}_label_{label}.png")
    pil_img.save(save_path)

    # 如果想在屏幕上直接显示，也可以使用:
    # pil_img.show() 
    # 不过频繁调用会打开多个窗口，一般调试阶段酌情使用或使用plt.imshow().
    
print(f"随机抽取的 {num_to_show} 张图片已保存到 'visual_random' 文件夹。")
