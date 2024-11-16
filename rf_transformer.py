import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class EnergyToTensor:
    def __init__(self, normalization='minmax', augment=False, precision='float32'):
        """
        normalization: 'minmax', 'log', 'zscore', 'mean', 'max' 或 'log-symmetric'
        augment: 是否进行数据增强
        precision: 'float32' 或 'float16'，指定数据精度
        """
        self.normalization = normalization
        self.augment = augment
        self.precision = precision

        # 定义数据增强流程，包含随机裁剪和水平翻转
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop((224, 224)),
            ])

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            Pxx = Image.fromarray(data)  # 将 NumPy array 转为 PIL Image

            # 使用定义好的 transforms 进行裁剪和数据增强
            Pxx = self.transform(Pxx)

            # Pxx = np.array(Pxx)  # 再将 PIL Image 转为 NumPy array
            # Pxx = np.expand_dims(Pxx, axis=0)  # 增加通道维度 (1, H, W)

            # 归一化处理
            if self.normalization == 'minmax':
                Pxx_min = np.min(Pxx)
                Pxx_max = np.max(Pxx)
                Pxx_normalized = (Pxx - Pxx_min) / (Pxx_max - Pxx_min + 1e-8)
            elif self.normalization == 'log':
                Pxx_normalized = np.log(Pxx + 1e-8)
            elif self.normalization == 'zscore':
                Pxx_mean = np.mean(Pxx)
                Pxx_std = np.std(Pxx)
                Pxx_normalized = (Pxx - Pxx_mean) / (Pxx_std + 1e-8)
            elif self.normalization == 'mean':
                Pxx_mean = np.mean(Pxx)
                Pxx_min = np.min(Pxx)
                Pxx_max = np.max(Pxx)
                Pxx_normalized = (Pxx - Pxx_mean) / (Pxx_max - Pxx_min + 1e-8)
            elif self.normalization == 'max':
                Pxx_max = np.max(Pxx)
                Pxx_normalized = Pxx / (Pxx_max + 1e-8)
            elif self.normalization == 'log-symmetric':
                Pxx_normalized = np.sign(Pxx) * np.log1p(np.abs(Pxx))
            else:
                raise ValueError("Unknown normalization method")

            # 根据设置选择数据精度
            if self.precision == 'float32':
                dtype = torch.float32
            elif self.precision == 'float64':
                dtype = torch.float64
            else:
                raise ValueError("Unsupported precision type")

            # Pxx_normalized = torch.tensor(Pxx_normalized, dtype=dtype)
            # Pxx_normalized = Pxx_normalized.repeat(3, 1, 1)  # 模拟 RGB 通道

            return Pxx_normalized

        elif isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32)  # 默认返回 float32 类型
        else:
            raise TypeError("Input should be a NumPy array or PyTorch Tensor")
