## Fine-tuning Pre-trained MAE for Classification

### ✨ Image Classification - GastroVision - 22 categories - Testing set

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1 | Accuracy | ACC   |
|:--------------:|:----------:|:-----------:   |:------------:|:--------:|:--------:|:-----:|
| Gastrovison [1]| 224x224    | 73,38          | 62,31        | 65,04    | 82,03    | 79,89 |
| Ours (SSL)     | 224x224    | 74,47          | 75,69        | 87,37    | 85,37    | 83,69 |

```
PRETRAINED_DIR = "..\mae_vit_large\checkpoint-399.pth"
DATA_PATH = "..\datasets"
OUTPUT_DIR = "finetuning/GastroVision"
LOG_DIR = "finetuning/GastroVision" 

python main_finetune_gastrovision.py
  --batch_size 50 ^
  --model vit_large_patch16 ^
  --finetune ${PRETRAINED_DIR} ^
  --epochs 50 ^
  --blr 1e-3 ^
  --smoothing 0 ^
  --num_workers 8 ^
  --data_path ${DATA_PATH} ^
  --output_dir ${OUTPUT_DIR}  ^
  --log_dir ${LOG_DIR} ^
  --nb_classes 22
```
Get our pre-trained checkpoints from here:

| Model                  | Checkpoint                        | Description               |
|:--------------:        |:----------:                       |:-----------:              |
| ViT-Large (pretrained) | [Download](https://drive.google.com/uc?id=1A5gXzQBJc9XCRhZ8-E2GzghyS2IMHndG)| Model trained at epoch 400| 
| ViT-Large (finetuning) | [Download](https://drive.google.com/uc?id=1tLlzCu1OQRC_bDEwBraz5S4GqBZvwVww)| Model trained at epoch 50 | 

### ✨ Image Classification - Hyperkvasir - 16 categories - Validation set fold 1 and fold 2 **official split**

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1     | Accuracy    | MCC   |
|:--------------:|:----------:|:--------------:|:--------:    |:------------:|:-----------:|:-----:|
| Guo [2]        | 224x224    | 73,68 ± 0,20   | 75,00 ± 0,20 | 73,39 ± 0,30 | 88,92 ± 0,40| -     |
| Ours (SSL)     | 224x224    | 88.56 ± 0,62   | 89.35 ±0,69  | 88.75 ± 0,65 | 93.72 ± 0,29| 93.18 ± 0,32|

```
PRETRAINED_DIR = "..\mae_vit_large\checkpoint-399.pth"
DATA_PATH = "..\datasets"
OUTPUT_DIR = "finetuning\hyper-kvasir\23 categories"
LOG_DIR = "finetuning\hyper-kvasir\23 categories" 

python main_finetune_hyperkvasir.py ^
  --batch_size 50 ^
  --model vit_large_patch16 ^
  --finetune ${PRETRAINED_DIR} ^
  --epochs 50 ^
  --blr 1e-3 ^
  --smoothing 0 ^
  --num_workers 8 ^
  --data_path ${DATA_PATH} ^
  --output_dir ${OUTPUT_DIR} ^
  --log_dir ${LOG_DIR} ^
  --nb_classes 23
```
Get our pre-trained checkpoints from here:

| Model                  | Checkpoint                        | Description               |
|:--------------:        |:----------:                       |:-----------:              |
| ViT-Large (pretrained) | [Download](https://drive.google.com/uc?id=1A5gXzQBJc9XCRhZ8-E2GzghyS2IMHndG)| Model trained at epoch 400| 
| ViT-Large (finetuning fold 1) | [Download](https://drive.google.com/uc?id=17terVuGDmh6BqcGV7XE9o1xHgR5ivBUK)| Model trained at epoch 50 | 
| ViT-Large (finetuning fold 2) | [Download](https://drive.google.com/uc?id=1bLj3OiRBrndPHCPRGpJcoCKI1Ds5I61p)| Model trained at epoch 50 | 

### ✨ Image Classification - Hyperkvasir - 16 categories - Validation set fold 1 and fold 2 **official split**

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1     | Accuracy    | MCC   |
|:--------------:|:----------:|:--------------:|:--------:    |:------------:|:-----------:|:-----:|
| Guo [2]        | 224x224    | 73,68 ± 0,20   | 75,00 ± 0,20 | 73,39 ± 0,30 | 88,92 ± 0,40| -     |
| Ours (SSL)     | 224x224    | 88.56 ± 0,62   | 89.35 ±0,69  | 88.75 ± 0,65 | 93.72 ± 0,29| 93.18 ± 0,32|

```
PRETRAINED_DIR = "..\mae_vit_large\checkpoint-399.pth"
DATA_PATH = "..\datasets"
OUTPUT_DIR = "finetuning\hyper-kvasir\16 categories"
LOG_DIR = "finetuning\hyper-kvasir\16 categories" 

python main_finetune_hyperkvasir.py ^
  --batch_size 50 ^
  --model vit_large_patch16 ^
  --finetune ${PRETRAINED_DIR} ^
  --epochs 50 ^
  --blr 1e-3 ^
  --smoothing 0 ^
  --num_workers 8 ^
  --data_path ${DATA_PATH} ^
  --output_dir ${OUTPUT_DIR} ^
  --log_dir ${LOG_DIR} ^
  --nb_classes 16
```
Get our pre-trained checkpoints from here:

| Model                  | Checkpoint                        | Description               |
|:--------------:        |:----------:                       |:-----------:              |
| ViT-Large (pretrained) | [Download](https://drive.google.com/uc?id=1A5gXzQBJc9XCRhZ8-E2GzghyS2IMHndG)| Model trained at epoch 400| 
| ViT-Large (finetuning fold 1) | [Download](https://drive.google.com/uc?id=1A5gXzQBJc9XCRhZ8-E2GzghyS2IMHndG)| Model trained at epoch 50 | 
| ViT-Large (finetuning fold 2) | [Download](https://drive.google.com/uc?id=1SEe4DRNmUvbBZuiGEmBmEsY8rKBVGX-A)| Model trained at epoch 50 | 

## Notes 
To fine-tune with **multi-node distributed training**, run the following on 4 nodes with 8 GPUs each:
```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
- Install submitit (`pip install submitit`) first.
- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`nodes`) * 8 (gpus per node) = 1024.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- We have run 4 trials with different random seeds. The resutls are 83.63, 83.66, 83.52, 83.46 (mean 83.57 and std 0.08).
- Training time is ~7h11m in 32 V100 GPUs.

Script for ViT-Large:
```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 --use_volta32 \
    --batch_size 32 \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
- We have run 4 trials with different random seeds. The resutls are 85.95, 85.87, 85.76, 85.88 (mean 85.87 and std 0.07).
- Training time is ~8h52m in 32 V100 GPUs.
- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`accum_iter`) * 8 (gpus) = 1024. `--accum_iter 4` simulates 4 nodes.

#### Notes

- The [pre-trained models we provide](https://github.com/fairinternal/mae/#pre-trained-checkpoints) are trained with *normalized* pixels `--norm_pix_loss` (1600 epochs, Table 3 in paper). The fine-tuning hyper-parameters are slightly different from the default baseline using *unnormalized* pixels.

- The original MAE implementation was in TensorFlow+TPU with no explicit mixed precision. This re-implementation is in PyTorch+GPU with automatic mixed precision (`torch.cuda.amp`). We have observed different numerical behavior between the two platforms. In this repo, we use `--global_pool` for fine-tuning; using `--cls_token` performs similarly, but there is a chance of producing NaN when fine-tuning ViT-Huge in GPUs. We did not observe this issue in TPUs. Turning off amp could solve this issue, but is slower.

- Here we use RandErase following DeiT: `--reprob 0.25`. Its effect is smaller than random variance.
