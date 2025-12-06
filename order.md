### train main_supcon.py
conda:swin

```bash
python main_supcon.py --batch_size 2 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --data_folder /disk/datasets/rf_data/Spectrogram/dataset/train \
  --dataset sp \
  --epochs 100
  ```

1. 尝试降低learning_rate和增加温度系数temp
```bash
nohup python main_supcon.py --batch_size 4 \
  --learning_rate 0.05 \
  --temp 0.2 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp \
  --epochs 20 > runlog.txt \
  --save_freq 2 2>&1 &
```

2. 压缩模型CustomCNNmini，从而增大batchsize,再次降学习率
```bash
nohup python main_supcon.py --batch_size 16 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.2 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 20 \
  --save_freq 2 > runlog.txt 2>&1 &
```

3.观察loss发现 相比压缩前（调整学习率前），loss有下降的趋势，因此增加训练的轮次数
再次增加temp从0.2到0.3
[SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine](save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine)
```bash
nohup python main_supcon.py --batch_size 16 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.3 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 40 \
  --save_freq 2 > runlog.txt 2>&1 &
```
4.增加temp=0.3后loss增加，因此回调temp=0.2，测试epochs 40的结果
[SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine](save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine)
```bash
nohup python main_supcon.py --batch_size 16 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.2 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 40 \
  --save_freq 2 > runlog.txt 2>&1 &
```
结果来看：loss更低，且拟合速度更快，因此temp=0.2更佳


5.模型泛化性不足
- 新增不同的数据增强策略
- 模型dropout（0.3-0.5之间）
- 优化器L2正则化参数的大小weight_decay（在 1e-4 到 1e-2 之间）
- 数据标准化
```bash
nohup python main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNminidrop \
  --learning_rate 0.01 \
  --temp 0.2 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 50 \
  --save_freq 2 --weight_decay 1e-3 > runlog.txt 2>&1 &

nohup python main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNminidrop \
  --learning_rate 0.1 \ 
  --temp 0.2  --cosine \                                                         
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \     
  --dataset sp --epochs 50 \                                                                                                    
  --save_freq 2 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```
t-SNE成一个环形，非常奇怪，同时loss在ep50时仍然有下降的趋势

6.取消数据标准化、调整亮度和对比度的数据增强策略、取消dropout的策略，增大轮次
与4相比相当于只修改了：中心裁剪->随机缩放裁剪
```bash
nohup python main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.2  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 150 \
  --save_freq 2 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```
拟合效果不错,ep150时loss未完全停止下降，后续考虑继续增加轮次
泛化效果仍然不佳

7.不使用随机缩放裁剪，使用随机裁剪，训练200轮（覆盖步骤6的模型）
[text](save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine)
```bash
nohup python main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.2  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 200 \
  --save_freq 2 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```

8.增加数据标准化、修改饱和度亮度、水平翻转的策略,训练300轮
```bash
nohup python main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.2  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 300 \
  --save_freq 2 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```
效果不好,全部预测为class1

9.取消修改饱和度亮度、水平翻转，只增加数据标准化，覆盖步骤8
[text](save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine)

10.只加数据标准化无法拟合，在7的基础上加修改饱和度亮度+水平翻转,不标准化，覆盖步骤8
效果很好，说明问题主要由标准化导致
总结目前采取的数据增强策略：随机裁剪、随机调整亮度和对比度、随机水平翻转


11.增强CNN model的drop=0.3 学习率=0.05
```bash
nohup python main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNminidrop \
  --learning_rate 0.05 \
  --temp 0.2  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 400 \
  --save_freq 5 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```
### train main_linear.py

```bash
python main_linear.py --batch_size 2 \
  --test_batch_size 2 \
  --learning_rate 5 \
  --ckpt save/SupCon/sp_models/SupCon_sp_resnet50_lr_0.5_decay_0.0001_bsz_2_temp_0.1_trial_0_cosine/last.pth \
  --data_folder /disk/datasets/rf_data/Spectrogram/dataset/train \
  --val_data_folder /disk/datasets/rf_data/Spectrogram/dataset/test \
  --dataset sp \
  --epochs 10
```

1.以一阶段训练的这个模型进行二阶段训练
[ckpt_epoch_40.pth](save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_40.pth)
```bash
nohup python main_linear.py --batch_size 32 \
  --model CustomCNNmini \
  --test_batch_size 64 \
  --learning_rate 0.1 \
  --ckpt save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_30.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
  --dataset sp \
  --epochs 10 > runlog2.txt 2>&1 &
```
结果：ep40:96.894 ep30:96.920 ep22:95.83  ep24:95.96

2.测试temp=0.3的结果
[ckpt_epoch_30.pth](save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/ckpt_epoch_30.pth)
结果：ep30：96.7065  ep36:97.16

3.训练的模型在推理时，泛化性不佳

4.由一阶段步骤6的模型进行二阶段训练
[ckpt_epoch_140.pth](save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth)
由散点图看出分离界面非线性，新增MLP分类器
```bash
nohup python main_linear.py --batch_size 32 \
  --model CustomCNNmini \
  --test_batch_size 64 \
  --learning_rate 0.1 \
  --ckpt save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
  --dataset sp \
  --epochs 10 > runlog2.txt 2>&1 &

nohup python main_linear.py --batch_size 32 \
  --model CustomCNNmini \
  --classifier MLP \
  --test_batch_size 64 \
  --learning_rate 0.1 \
  --ckpt save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_190.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
  --dataset sp \
  --epochs 10 > runlog2.txt 2>&1 &
```
ep190:linear95.78 MLP95.78(保留)

5.测试步骤10产生的模型
[generalizetion_tranSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine](save/SupCon/sp_models/generalizetion_tranSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine)
```bash
nohup python main_linear.py --batch_size 32 \
  --model CustomCNNmini \
  --classifier MLP \
  --test_batch_size 64 \
  --learning_rate 0.5 \
  --ckpt save/SupCon/sp_models/generalizetion_tranSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_300.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
  --dataset sp \
  --epochs 20 > runlog2.txt 2>&1 &
```
|轮次|linear|MLP|
|---|---|---|
|ep300| linear 96.11 | MLP 96.07|
|ep170| linear 93.56 | MLP 93.57|
### main_tSNE.py
```bash
python main_tSNE.py \
    --model CustomCNNmini \
    --feature_type all \
    --ckpt save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/ckpt_epoch_30.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
    --batch_size 4 \
    --num_workers 8

python main_tSNE.py \
    --model CustomCNNmini \
    --feature_type all \
    --ckpt save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_26.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
    --batch_size 32 \
    --num_workers 8

python main_tSNE.py \
    --model CustomCNNminidrop \
    --feature_type all \
    --ckpt save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNminidrop_lr_0.1_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_18.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
    --batch_size 32 \
    --num_workers 8

python main_tSNE.py \
    --model CustomCNNmini \
    --feature_type all \
    --ckpt save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
    --batch_size 32 \
    --num_workers 8
```
save/SupCon/sp_models/generalizetionSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_170.pth

### 推理
python inference.py \
--encoder_path save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/ckpt_epoch_36.pth \
--classifier_path save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/best_classifier_97.16.pth \
--image_dir /disk/datasets/rf_data/newspectrum/SelectAB/test \
--mode data



nohup python main_linear.py --batch_size 2 \
  --model CustomCNNmini \
  --classifier MLP \
  --test_batch_size 64 \
  --learning_rate 0.1 \
  --ckpt save/SupCon/sp_models/generalizetion_tranSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_170.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/few/train5shot \
  --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/few/val \
  --dataset sp \
  --epochs 30 > runlog2.txt 2>&1 &

  
nohup python main_linear.py --batch_size 32 \
  --model CustomCNNmini \
  --classifier MLP \
  --test_batch_size 64 \
  --learning_rate 0.1 \
  --ckpt save/SupCon/sp_models/generalizetion_tranSupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_170.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/val \
  --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/val \
  --dataset sp \
  --epochs 30 > runlog2.txt 2>&1 &