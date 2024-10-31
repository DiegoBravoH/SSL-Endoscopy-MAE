# Self-Supervised Learning for Gastrointestinal Endoscopy: Improving Classification and Data Quality with Masked Autoencoders
## Pipeline

<p align="center">
  <img src="figs/Pipeline_GI.jpg" alt="GastroUNAL" width="700">
</p>

Overview of the proposed self-supervised methodology: (I) **Pre-training:** A large portion of image patches (50%) is randomly masked. The encoder processes the visible patches, and mask tokens are introduced after the encoder. The combined encoded patches and mask tokens are fed into a decoder to reconstruct the original image. (II) **Supervised Training:** The decoder is removed, and the pre-trained encoder is fine-tuned to classify images into specific categories across four groups: normal, anatomical, pathological, and therapeutic.

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
| Gastrovison [1]| 224x224    | 73,38          | 62,31        | 65,04    | 82,03    | 79,89 |
| Ours (SSL)     | 224x224    | 74,47          | 75,69        | 87,37    | 85,37    | 83,69 |

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
| Guo [2]        | 224x224    | 73,68 ± 0,20   | 75,00 ± 0,20 | 73,39 ± 0,30 | 88,92 ± 0,40| -     |
| Ours (SSL)     | 224x224    | 88.56 ± 0,62   | 89.35 ±0,69  | 88.75 ± 0,65 | 93.72 ± 0,29| 93.18 ± 0,32|

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
| Ours (SSL)     | 224x224    | 74.09 ± 2,61   | 63.97 ± 0,41 | 64.06 ± 0,12| 91.10 ± 0,23| 90.35 ± 0,25 |

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