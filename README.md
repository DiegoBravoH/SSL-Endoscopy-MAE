# Self-Supervised Learning for Multi-Category Endoscopy classification and Data Quality evaluation Using Masked Autoencoders
## Pipeline

<p align="center">
  <img src="figs/Pipeline_GI.jpg" alt="GastroUNAL" width="700">
</p>

Overview of the proposed self-supervised methodology: (I) **Pre-training:** A large portion of image patches (i.e $50\%$) is randomly masked. The encoder processes the visible patches, and mask tokens are introduced after the encoder. The combined encoded patches and mask tokens are fed into a decoder to reconstruct the original image. (II) **Supervised Training:** The decoder is removed, and the pre-trained encoder is fine-tuned to classify images into specific categories within EN images, spanning normal, anatomical, pathological, and therapeutic regions, with category definitions varying by dataset.

## Dataset Overview

We used two endoscopic datasets: HyperKvasir (2020) and GastroVision (2023). These are public available on their official sites. You can download them under the agreements.

- **HyperKvasir**: 110,079 images, with 10,662 labeled images across 23 classes (anatomical landmarks, mucosal view quality, pathological findings, and therapeutic interventions) and 99,417 unlabeled. [Download](https://osf.io/mh9sj/)
- **GastroVision**: 8,000 labeled images across 27 classes. [Download](https://osf.io/84e7f/.)


Here is a BibTeX entry that you can use to cite the dataset:
```
@article{borgli2020hyperkvasir,
  title={HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy},
  author={Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, Sigrun L and Randel, Kristin Ranheim and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc Tien Dang and others},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={283},
  year={2020},
  publisher={Nature Publishing Group UK London}
}

@inproceedings{jha2023gastrovision,
  title={Gastrovision: A multi-class endoscopy image dataset for computer aided gastrointestinal disease detection},
  author={Jha, Debesh and Sharma, Vanshali and Dasu, Neethi and Tomar, Nikhil Kumar and Hicks, Steven and Bhuyan, Manas Kamal and Das, Pradip K and Riegler, Michael A and Halvorsen, P{\aa}l and Bagci, Ulas and others},
  booktitle={Workshop on Machine Learning for Multimodal Healthcare Data},
  pages={125--140},
  year={2023},
  organization={Springer}
}
```


## Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).


## Finetuning
The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).


### Fine-tuning with pre-trained checkpoints
The following table provides the pre-trained checkpoints used in Table 1:


## 🚀 Main Results

### ✨ Image Classification - GastroVision - 22 categories - Testing set

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1 | Accuracy | ACC   |
|:--------------:|:----------:|:-----------:   |:------------:|:--------:|:--------:|:-----:|
| Gastrovison [1]| 224x224    | 73.38          | 62.31        | 65.04    | 82.03    | 79.89 |
| SSL(RandomW)     | 224x224    | 38.41         | 43.53        | 38.65     | 50.06    | 45.98 |
| SSL(ImageNet)     | 224x224    | 73.96         | 70.28        | 70.32   | 82.72    | 80.74 |
| SSL(EN)     | 224x224    | 74.01          | 79.20        | 75.72    | 84.44    | 82.78 |
| SSL(EN)+prepro.     | 224x224    | **75.13**         | **81.78**        | **77.40**    | **84.63**    | **83.02** |


*Matthews correlation coefficient (MCC)
```
[1]: 
@inproceedings{jha2023gastrovision,
  title={Gastrovision: A multi-class endoscopy image dataset for computer aided gastrointestinal disease detection},
  author={Jha, Debesh and Sharma, Vanshali and Dasu, Neethi and Tomar, Nikhil Kumar and Hicks, Steven and Bhuyan, Manas Kamal and Das, Pradip K and Riegler, Michael A and Halvorsen, P{\aa}l and Bagci, Ulas and others},
  booktitle={Workshop on Machine Learning for Multimodal Healthcare Data},
  pages={125--140},
  year={2023},
  organization={Springer}
}
```

### ✨ Image Classification - Hyperkvasir - 16 categories - Validation set fold 1 and fold 2 **official split

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1     | Accuracy    | MCC   |
|:--------------:|:----------:|:--------------:|:--------:    |:------------:|:-----------:|:-----:|
| Guo [2]        | 224x224    | 73.68 ±0.20   | 75.00 ±0.20 | 73.39 ±0.30 | 88.92 ±0.40| -     |
| SSL(RandomW)     | 224x224    | 70.92 ±0.44    | 73.81 ±0.57  | 71.79 ± ±0.32 | 78.57 ±0.05| 76.80 ±0.11|
| SSL(ImageNet)    | 224x224    | 86.20 ±0.20   | 87.56 ±0.11  | 86.77 ± 0.13 | 92.49 ±0.33| 91.83 ±0.36|
| SS(EN)+quality     | 224x224    | **89.09 ±0.58**   | **89.86 ±0.79**  | **89.23 ±0.03** | **93.74 ±0.11**| **93.19 ±0.12**|
| SSL(EN)     | 224x224    | 87.63 ±0.36   | 89.32 ±0.15  | 88.23 ±0.20 | 93.50 ±0.30| 92.94 ± 0.33|


*Matthews correlation coefficient (MCC) 
```
[2]:

@article{guo2024improving,
  title={Improving image classification of gastrointestinal endoscopy using curriculum self-supervised learning},
  author={Guo, Han and Somayajula, Sai Ashish and Hosseini, Ramtin and Xie, Pengtao},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={6100},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

### ✨ Image Classification - Hyperkvasir - 23 categories - Validation set fold 1 **official split

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1    | Accuracy    | MCC   |
|:--------------:|:----------:|:--------------:|:--------:    |:-----------:|:-----------:|:-----:|
| Borgoli [3]    | 224x224    | 63.30          | 61.50        | 61.70       | 91.00       | 90.20        |
| SSL(RandomW)     | 224x224    | 47.55 ±1.28   | 50.48 ±0.57 | 47.57 ±0.94| 72.16 ±0.45| 70.08 ±0.55 |
| SSL(ImageNet)    | 224x224    | **73.01 ±3.01**    | 63.15 ±0.76 | 62.01 ±0.40| 89.64 ±0.31| 88.77 ±0.33 |
| SSL(EN)+quality     | 224x224    | 71.31 ±3.09   | 65.99 ±0.37 | 64.68 ±0.39 | 90.85 ±0.67| 90.11 ±0.71 |
| SSL(EN)     | 224x224    | 67.96 ±2.93   | **66.41 ±0.41** | **65.73 ±0.39**| **91.00 ±0.04** | **90.25 ±0.04** |


*Matthews correlation coefficient (MCC)
```
[3]:

@article{borgli2020hyperkvasir,
  title={HyperKvasir, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy},
  author={Borgli, Hanna and Thambawita, Vajira and Smedsrud, Pia H and Hicks, Steven and Jha, Debesh and Eskeland, Sigrun L and Randel, Kristin Ranheim and Pogorelov, Konstantin and Lux, Mathias and Nguyen, Duc Tien Dang and others},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={283},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
```


### ✨ TSNE - Quality indicator
<p align="center">
  <img src="figs/TSNE.jpg" width="700">
</p>

## 📓 Notebook for Quick Testing
`run_image_classification_prediction.ipynb`:  Use this notebook to run image prediction classification task.

Note 🗈:  To run this code in Google Colab, click the logo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ov02p9Q9W3_1dFasir96NHA3we0wmOdp)


## 🔒 License
This research study was conducted retrospectively using human subject data made available in open access. Ethical approval was not required as confirmed by the license attached with the open access data.


## ☎️ Contact 

Diego Bravo: dbravoh@unal.edu.co

## 👍 Acknowledgements
Hypekvasir and GastroVision authors :D