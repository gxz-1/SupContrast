import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F



import os
from PIL import Image
from torch.utils.data import Dataset

class SPDataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type='train'):
        """
        Args:
            data_dir (str): 数据目录路径。'train' 模式下包含类别文件夹，每个类别文件夹有两个子文件夹 'a' 和 'b'；
                            'test' 模式下包含类别文件夹，每个类别文件夹直接包含图片。
            transform (callable, optional): 对每个样本应用的变换函数或变换方法。
            data_type (str): 数据类型，'train' 或 'test'。默认值为 'train'。
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.data_type = data_type

        # 创建类别名称到索引的映射
        for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx

        if data_type == 'train':
            # 遍历所有类别文件夹并收集样本
            for class_name in self.class_to_idx.keys():
                class_path = os.path.join(data_dir, class_name)
                path_a = os.path.join(class_path, 'a')
                path_b = os.path.join(class_path, 'b')

                images_a = self._collect_images(path_a) if os.path.exists(path_a) else []
                images_b = self._collect_images(path_b) if os.path.exists(path_b) else []
                print(f"Class: {class_name}, Images in 'a': {len(images_a)}, Images in 'b': {len(images_b)}")
                # 假设 'a' 和 'b' 文件夹中的图像数量相等
                for img_a, img_b in zip(sorted(images_a), sorted(images_b)):
                    # 将图像路径和类别索引一起存储
                    self.samples.append((img_a, img_b, self.class_to_idx[class_name]))

        elif data_type == 'test':
            # 遍历类别文件夹并读取其中的图片，包括所有子文件夹
            for class_name in self.class_to_idx.keys():
                class_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_path):
                    # 使用 os.walk 递归遍历子文件夹中的所有文件
                    for root, _, files in os.walk(class_path):
                        for file in sorted(files):
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path):
                                # 将图像路径和类别索引一起存储
                                self.samples.append((file_path, self.class_to_idx[class_name]))

    def _collect_images(self, folder_path):
        """
        递归收集文件夹及其子文件夹中的所有图片路径。

        Args:
            folder_path (str): 文件夹路径。

        Returns:
            list: 图片路径列表。
        """
        images = []
        for root, _, files in os.walk(folder_path):
            images.extend(
                [os.path.join(root, f) for f in sorted(files) if os.path.isfile(os.path.join(root, f))]
            )
        return images

    def __len__(self):
        # 返回数据集中样本的总数量
        return len(self.samples)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            # 获取样本路径和类别索引
            img_path_a, img_path_b, label = self.samples[idx]
            image_a = Image.open(img_path_a).convert('L')
            image_b = Image.open(img_path_b).convert('L')

            # 如果有传入变换，则分别对图像应用变换
            if self.transform:
                image_a = self.transform(image_a)
                image_b = self.transform(image_b)

            # 返回两个图像的列表和类别索引
            return [image_a, image_b], label

        elif self.data_type == 'test':
            # 获取测试图像和标签
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('L')

            # 如果有传入变换，则对图像应用变换
            if self.transform:
                image = self.transform(image)

            # 返回图像和标签
            return image, label