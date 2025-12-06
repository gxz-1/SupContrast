# 数据
## 数据集
1. 在演示的pdf中有数据的数量情况
2. tranform情况
```commandline
        train_transform = transforms.Compose([
            transforms.RandomCrop((500, 500)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])
```
3.构建数据集的过程可以用算法描述

## 训练  new_main_supcon_Generalization.py
源码提供了SimCLR和supcon对比损失

常见的对比损失类型

| 损失类型    | 输入参数特征                       | 核心计算逻辑                                     |
| ----------- | ---------------------------------- | ------------------------------------------------ |
| 对比损失    | 需`output1/output2 + label`（0/1） | 计算欧式距离 + 边际值（margin）                  |
| 三元组损失  | 需`anchor/positive/negative`       | 最大化锚点 - 负样本距离，最小化锚点 - 正样本距离 |
| SimCLR 损失 | 仅需`output1/output2`（无需标签）  | 余弦相似度 + 温度系数 + InfoNCE                  |
| 交叉熵损失  | 需`特征 + 类别标签`                | 特征映射为类别概率 + 交叉熵计算                  |

1. 使用的模型 CustomCNNmini 相关的参数如下
```commandline
nohup python new_main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNmini --method SupCon\
  --learning_rate 0.05 \
  --temp 0.2  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/UAV/UAVSet/train \
  --dataset sp --epochs 200 \
  --save_freq 5 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```
2. 训练的loss变化在csv中有保存
3. 表征模型可以用tSNE可视化、用指标表征 val_train.py
**损失函数为 SupConLoss**
```commandline

Evaluation Results:
Intra-class distance - Mean: 0.5221, Std: 0.5046
Inter-class distance - Mean: 1.2344, Std: 0.5405
类内/类间距离比: 0.4324
正负样本距离差 - Mean: 0.7123
KNN (k=1) Accuracy: 1.0000
KNN (k=5) Accuracy: 0.9738
```
## 微调 main_linear.py
classifier有MLP和linear两种
```commandline
python main_linear.py ^
  --batch_size 32 ^
  --model CustomCNNmini ^
  --classifier linear ^
  --test_batch_size 32 ^
  --learning_rate 0.1 ^
  --ckpt save/ckpt_epoch_140.pth ^
  --data_folder "E:\RFcode\rf数据集\UAVSet\train" ^
  --val_data_folder "E:\RFcode\rf数据集\UAVSet\test" ^
  --dataset sp --split_ratio 1 --n_cls 5 ^
  --epochs 30 ^
  --num_workers 2
```
## 微调模型的指标 new_predict.py
predict opt=data输出混淆矩阵 opt=predict输出指标'准确率', '精确率', '召回率', 'F1值'