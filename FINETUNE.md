## Fine-tuning MAE for Classification

Download our pre-trained SSL checkpoints to use in PRETRAINED_DIR for transfer learning and fine-tuning training:
| Model                  | Checkpoint                        | Description               |
|:--------------:        |:----------:                       |:-----------:              |
| ViT-Large (pretrained) | [Download](https://drive.google.com/uc?id=1rdBKa4vV9vtrASuMtnH3-26V0DvjF7kd)| SSL(ImageNet)|
| ViT-Large (pretrained) | [Download](https://drive.google.com/uc?id=1-9gRPPb_3LnXXd8nV4kG9WxrIFlMKCm3)| SSL(RandomW)| 
| ViT-Large (Endocopy) | [Download](https://drive.google.com/uc?id=1zfOa9G7ZYxtqNM-z7gm5kxVNIg2b0nsJ)| **SSL(Endoscopy dataset)** masking ratio 50%|

### ✨ Image Classification - GastroVision - 22 categories - Testing set

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1 | Accuracy | ACC   | Checkpoint   |
|:--------------:|:----------:|:-----------:   |:------------:|:--------:|:--------:|:-----:|:-----:|
| Gastrovison [1]| 224x224    | 73.38          | 62.31        | 65.04    | 82.03    | 79.89 | **see reference**   |
| SSL(RandomW)     | 224x224    | 38.41         | 43.53        | 38.65     | 50.06    | 45.98 |[Download](https://drive.google.com/uc?id=15CQuQZgpaRgH3yJ35MtHx6_ve4eSBaen)| SSL(ImageNet)   |
| SSL(ImageNet)     | 224x224    | 73.96         | 70.28        | 70.32   | 82.72    | 80.74 |[Download](https://drive.google.com/uc?id=1y37dLMDrTjMudkGM8XnnwpDIk5jtx3UL)   |
| SSL(EN)     | 224x224    | 74.01          | 79.20        | 75.72    | 84.44    | 82.78 |[Download](https://drive.google.com/uc?id=1pumFg8Q-78RrSJof05GGUCGiuVheRGiT)   |
| SSL(EN)+prepro.     | 224x224    | **75.13**         | **81.78**        | **77.40**    | **84.63**    | **83.02** |[Download](https://drive.google.com/uc?id=1jLPsxvXSFZBiYeoQqptQv3rT9V0pS1h6)    |

```
PRETRAINED_DIR = "..\mae_vit_large\mask_ratio0.5\checkpoint-399.pth"
DATA_PATH = "..\datasets"
SPLIT_PATH = "..\datasets\supervised_gastrovision.csv"
OUTPUT_DIR = "finetuning\GastroVision"
LOG_DIR = "finetuning\GastroVision" 

python main_finetune_gastrovision.py
  --batch_size 50 ^
  --model vit_large_patch16 ^
  --finetune ${PRETRAINED_DIR} ^
  --epochs 50 ^
  --blr 1e-3 ^
  --smoothing 0 ^
  --num_workers 8 ^
  --data_path ${DATA_PATH} ^
  --official_split ${SPLIT_PATH} ^
  --output_dir ${OUTPUT_DIR}  ^
  --log_dir ${LOG_DIR} ^
  --nb_classes 22
```

### ✨ Image Classification - Hyperkvasir - 16 categories - Validation set fold 1 and fold 2 **official split**

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1     | Accuracy    | MCC   |Checkpoint   |
|:--------------:|:----------:|:--------------:|:--------:    |:------------:|:-----------:|:-----:|:-----:|
| Guo [2]        | 224x224    | 73.68 ±0.20   | 75.00 ±0.20 | 73.39 ±0.30 | 88.92 ±0.40| -     |view the paper     |
| SSL(RandomW)     | 224x224    | 70.92 ±0.44    | 73.81 ±0.57  | 71.79 ± ±0.32 | 78.57 ±0.05| 76.80 ±0.11|[fold1](https://drive.google.com/uc?id=15nkD62j7ySGil-6LiYEdfJ_vxIcOkgrh) [fold2](https://drive.google.com/uc?id=1L-AfJFUJLV1-0PEKnNx_wfjG5lmYVCAe)    |
| SSL(ImageNet)    | 224x224    | 86.20 ±0.20   | 87.56 ±0.11  | 86.77 ± 0.13 | 92.49 ±0.33| 91.83 ±0.36|[fold1](https://drive.google.com/uc?id=1LeuSLtjY_tJgBfmJal57ckSOkJu5qaXB) [fold2](https://drive.google.com/uc?id=1eJ9pB2Uwbz421ze3TkkfYEgc-NUoD7s5)     |
| SS(EN)+quality     | 224x224    | **89.09 ±0.58**   | **89.86 ±0.79**  | **89.23 ±0.03** | **93.74 ±0.11**| **93.19 ±0.12**|a b    |
| SSL(EN)     | 224x224    | 87.63 ±0.36   | 89.32 ±0.15  | 88.23 ±0.20 | 93.50 ±0.30| 92.94 ± 0.33|[fold1](https://drive.google.com/uc?id=14k8j0ejzLSHdpY3VrL7AQ6bSZGoynNJc) [fold2](https://drive.google.com/uc?id=1ldKddgjg6HNfSdQ0N_scPVhAnVxdQndp)     |


```
PRETRAINED_DIR = "..\mae_vit_large\mask_ratio0.5\checkpoint-399.pth"
DATA_PATH = "..\datasets"
SPLIT_PATH = "..\datasets\supervised_hyper-kvasir_16Categories_fold1.csv" 
OUTPUT_DIR = "finetuning\hyper-kvasir"
LOG_DIR = "finetuning\hyper-kvasir" 

python main_finetune_hyperkvasir.py
  --batch_size 50 ^
  --model vit_large_patch16 ^
  --finetune ${PRETRAINED_DIR} ^
  --epochs 50 ^
  --blr 1e-3 ^
  --smoothing 0 ^
  --num_workers 8 ^
  --data_path ${DATA_PATH} ^
  --official_split ${SPLIT_PATH} ^
  --output_dir ${OUTPUT_DIR}  ^
  --log_dir ${LOG_DIR} ^
  --nb_classes 16
```
SPLIT_PATH = supervised_hyper-kvasir_16Categories_fold1.csv or supervised_hyper-kvasir_16Categories_fold2.csv or supervised_hyper-kvasir_16Categories_fold1_filt.csv or supervised_hyper-kvasir_16Categories_fold2_filt.csv. 


### ✨ Image Classification - Hyperkvasir - 23 categories - Validation set fold 1 and fold 2 **official split**

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1    | Accuracy    | MCC   | MCC   |
|:--------------:|:----------:|:--------------:|:--------:    |:-----------:|:-----------:|:-----:|:-----:|
| Borgoli [3]    | 224x224    | 63.30          | 61.50        | 61.70       | 91.00       | 90.20        |view paper        |
| SSL(RandomW)     | 224x224    | 47.55 ±1.28   | 50.48 ±0.57 | 47.57 ±0.94| 72.16 ±0.45| 70.08 ±0.55 |[fold1](https://drive.google.com/uc?id=1wI8tQmozR8K3Ye8veUfATjciYd8mctu3) [fold2](https://drive.google.com/uc?id=1io8Zg7Exa7ep--Un5ZW8P7p0L9mdyUsI)        |
| SSL(ImageNet)    | 224x224    | **73.01 ±3.01**    | 63.15 ±0.76 | 62.01 ±0.40| 89.64 ±0.31| 88.77 ±0.33 |[fold1](https://drive.google.com/uc?id=1wI8tQmozR8K3Ye8veUfATjciYd8mctu3) [fold2](https://drive.google.com/uc?id=1L4ptZCYz-NS1rmXoymzg4G1SucC2gQJe)        |
| SSL(EN)+quality     | 224x224    | 71.31 ±3.09   | 65.99 ±0.37 | 64.68 ±0.39 | 90.85 ±0.67| 90.11 ±0.71 |[fold1](https://drive.google.com/uc?id=1cyhJjvJqDPkw99ZF1fg1d5M26hjlZ8Bb) [fold2](https://drive.google.com/uc?id=1ohK0MgfzfiYzwp97KPU2h0gJhwJJuLKJ)|
| SSL(EN)     | 224x224    | 67.96 ±2.93   | **66.41 ±0.41** | **65.73 ±0.39**| **91.00 ±0.04** | **90.25 ±0.04** |[fold1](https://drive.google.com/uc?id=1sVb09fUy-Aj55xgP11WwV8o6bujhia12) [fold2](https://drive.google.com/uc?id=166CrTd7tOhhvNA1X7PyPdoQdFRdIa-aR) | 


```
PRETRAINED_DIR = "..\mae_vit_large\mask_ratio0.5\checkpoint-399.pth"
DATA_PATH = "..\datasets"
SPLIT_PATH = "..\datasets\supervised_hyper-kvasir_23Categories_fold1.csv" 
OUTPUT_DIR = "finetuning\hyper-kvasir"
LOG_DIR = "finetuning\hyper-kvasir" 

python main_finetune_hyperkvasir.py
  --batch_size 50 ^
  --model vit_large_patch16 ^
  --finetune ${PRETRAINED_DIR} ^
  --epochs 50 ^
  --blr 1e-3 ^
  --smoothing 0 ^
  --num_workers 8 ^
  --data_path ${DATA_PATH} ^
  --official_split ${SPLIT_PATH} ^
  --output_dir ${OUTPUT_DIR}  ^
  --log_dir ${LOG_DIR} ^
  --nb_classes 23
```

SPLIT_PATH = supervised_hyper-kvasir_23Categories_fold1.csv or supervised_hyper-kvasir_23Categories_fold2.csv or supervised_hyper-kvasir_23Categories_fold1_filt.csv or supervised_hyper-kvasir_23Categories_fold2_filt.csv. 
