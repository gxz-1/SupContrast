### train main_supcon.py

```bash
python main_supcon.py --batch_size 2 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --data_folder /disk/datasets/rf_data/Spectrogram/dataset/train \
  --dataset sp \
  --epochs 100
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