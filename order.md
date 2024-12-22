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
结果：ep40:96.894 ep30:96.920

2.测试temp=0.3的结果
[ckpt_epoch_30.pth](save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/ckpt_epoch_30.pth)
结果：ep30：96.7065  ep36:97.16
### train_tSNE.py
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
```


python inference.py \
--encoder_path save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/ckpt_epoch_36.pth \
--classifier_path save/SupCon/sp_models/SupCon_sp_CustomCNNmini_lr_0.01_decay_0.0001_bsz_16_temp_0.3_trial_0_cosine/best_classifier_97.16.pth \
--image_dir /disk/datasets/rf_data/newspectrum/SelectAB/test \
--mode data