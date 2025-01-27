## Pre-training MAE

If you have unique GPU, it is recommended to use:
```
OUTPUT_DIR ='..\mae_vit_large_scratch'
LOG_DIR ='..\mae_vit_large_scratch' 
DATA_PATH='..\data\Labeled Images'
LOAD_PRETRAINED_DIR='..\mae_visualize_vit_large.pth'

python main_pretrain_presonalized.py \
  --batch_size 89 \
  --model mae_vit_large_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.5 \
  --epochs 400 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --num_workers 4 \
  --data_path ${DATA_DIR} \
  --input_size 224 \
  --output_dir ${OUTPUT_DIR}  \
  --log_dir ${LOG_DIR}  \
  --resume ${LOAD_PRETRAINED_DIR} 

```

If you have multiple GPUs, it is recommended to use:
```
torchrun --nproc_per_node=2 main_pretrain_presonalized.py \
  --batch_size 152 \
  --model mae_vit_large_patch16 \
  --mask_ratio 0.8 \
  --epochs 400 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --num_workers 22 \
  --data_path ${DATA_DIR} \
  --input_size 224 \
  --output_dir ${OUTPUT_DIR}  \
  --log_dir ${LOAD_PRETRAINED_DIR} 

```


We chose to use the pre-trained weights available in `LOAD_PRETRAINED_DIR` for:

### ✨ SSL 

| Model                  | Checkpoint                        | Description                                                 |
|:--------------:        |:----------:                       |:-----------:                                                |
| ViT-Large (Pre-trained)| [Download](https://drive.google.com/uc?id=1rdBKa4vV9vtrASuMtnH3-26V0DvjF7kd)| Initial model checkpoint used for training in `LOAD_PRETRAINED_DIR` |
| SSL-MAE (masrk ratio 10%)            | [Download](https://drive.google.com/uc?id=1PVdLU89zAX_9xnRYKw-opV1rElQOeGLr)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 20%)            | [Download](https://drive.google.com/uc?id=1SSryOv1ysxYM3hB8RJBTAcfD3ErySfJr)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 30%)            | [Download](https://drive.google.com/uc?id=1I_O-qFrJT1JwsUXYXAALBUI8cNHAxyPf)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 40%)            | [Download](https://drive.google.com/uc?id=1cTIpQh3DIxP25BdFu_vVhE-MMkJzqZEy)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 50%)            | [Download](https://drive.google.com/uc?id=1zfOa9G7ZYxtqNM-z7gm5kxVNIg2b0nsJ)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 60%)            | [Download](https://drive.google.com/uc?id=1qP9cNM7YxX2OmfkuI2FFxTIJw4Cb60q_)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 70%)            | [Download](https://drive.google.com/uc?id=1XaQoLRKaQ-8xa0c3gZjzd_LBHRND3OB6)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 80%)            | [Download](https://drive.google.com/uc?id=1o4ti8tQGX-JSaozro3lzd2WaomtJaN3R)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|
| SSL-MAE (masrk ratio 90%)            | [Download](https://drive.google.com/uc?id=1x41PRFb9-ezYRhStn8WFrowkxm9aLjxu)| SSL-trained model after 400 epochs, saved in `OUTPUT_DIR`|


#### Notes

- The pre-trained models we provide are trained with *normalized* pixels `--norm_pix_loss`, which enhances representation quality, as shown in the [original paper MAE](https://arxiv.org/pdf/2111.06377). Deactivating `--norm_pix_loss` (using  *unnormalized*) improves visual reconstruction but reduces performance on downstream classification tasks [Download unnormalized model](https://osf.io/84e7f/).


Taking from the original repository [linear scaling rule](https://github.com/facebookresearch/mae.git):
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.

To train ViT-Base or ViT-Huge, set `--model mae_vit_base_patch16` or `--model mae_vit_huge_patch14`.
