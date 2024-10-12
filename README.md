# Masked Autoencoders MAE:
## Pipeline

<p align="center">
  <img src="figs/Pipeline_GI.jpg" alt="GastroUNAL" width="700">
</p>


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
|:--------------:|:----------:|:-----------:   |:--------:    |:--------:|:--------:|:-----:|
| Gastrovison [1]    | 224x224    | 73.38          | 62.31        | 65.04    | 82,03    | 79,89 |
| Ours (SSL)     | 224x224    | 74.47          | 75.64        | 87.57    | 85,37    | 83,69 |

```
@inproceedings{jha2023gastrovision,
  title={Gastrovision: A multi-class endoscopy image dataset for computer aided gastrointestinal disease detection},
  author={Jha, Debesh and Sharma, Vanshali and Dasu, Neethi and Tomar, Nikhil Kumar and Hicks, Steven and Bhuyan, Manas Kamal and Das, Pradip K and Riegler, Michael A and Halvorsen, P{\aa}l and Bagci, Ulas and others},
  booktitle={Workshop on Machine Learning for Multimodal Healthcare Data},
  pages={125--140},
  year={2023},
  organization={Springer}
}
```

### ✨ Image Classification - Hyperkvasir - 16 categories - Validation set fold 1

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1 | Accuracy | ACC   |
|:--------------:|:----------:|:-----------:   |:--------:    |:--------:|:--------:|:-----:|
| Han [3]            | 224x224    | 73.68          | 75.00        | 73.39    | 88.92    | -     |
| Ours (SSL)     | 224x224    | 88.56          | 89.35        | 88.75    | 93.72    | 93.18 |

```
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

### ✨ Image Classification - Hyperkvasir - 23 categories - Validation set fold 1

| Backbone       | Resolution |Macro Precision | Macro Recall | Macro F1 | Accuracy | ACC   |
|:--------------:|:----------:|:-----------:   |:--------:    |:--------:|:--------:|:-----:|
| Hyperkvasir [2]    | 224x224    | 63.33          | 61.50        | 61.70    | 91.00    | 90.20 |
| Ours (SSL)     | 224x224    | 74.09          | 63.97        | 64.06    | 91.10    | 90.35 |

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
```


### ✨ TSNE - Quality indicator
<p align="center">
  <img src="figs/TSNE.jpg" width="700">
</p>




## 🔒 License

The data is released fully open for research and educational purposes. The use of the dataset for purposes such as competitions purposes needs prior written permission. In all documents and papers that use or refer to the dataset or report experimental results based on the GastroHUN, a reference to the related article needs to be added and the data.


## ☎️ Contact 

Diego Bravo: dbravoh@unal.edu.co

## 👍 Acknowledgements

Universidad Nacional de Colombia <br>
Hypekvasir and GastroVision authors :D