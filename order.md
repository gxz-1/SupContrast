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
nohup python main_supcon.py --batch_size 4 \
  --learning_rate 0.05 \
  --temp 0.2 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp \
  --epochs 20 > runlog.txt \
  --save_freq 2 2>&1 &

2. 压缩模型CustomCNNmini，从而增大batchsize
nohup python main_supcon.py --batch_size 32 \
  --model CustomCNNmini \
  --learning_rate 0.01 \
  --temp 0.2 \
  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
  --dataset sp --epochs 20 \
  --save_freq 2 > runlog.txt2>&1 &

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




python main_tSNE.py \
    --ckpt save/SupCon/sp_models/SupCon_sp_resnet50_lr_0.5_decay_0.0001_bsz_4_temp_0.1_trial_0_cosine/ckpt_epoch_20.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
    --batch_size 8 \
    --num_workers 8


python main_tSNE.py \
    --ckpt save/SupCon/sp_models/SupCon_sp_resnet50_lr_0.1_decay_0.0001_bsz_4_temp_0.2_trial_0_cosine/ckpt_epoch_2.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/SelectAB/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/SelectAB/test \
    --batch_size 8 \
    --num_workers 8