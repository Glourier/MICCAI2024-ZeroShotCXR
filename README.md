# Vision-language models for zero-shot multi-label fine-grained classification of chest X-ray images
+ This is the repository of MICCAI 2024 Challenge: CXR-LT.
+ Winner solution for Task3: zero-shot classification.  

## 1 References
+ Website of the challenge: https://bionlplab.github.io/2024_MICCAI_CXRLT/
+ Task3 on CodaLab: https://codalab.lisn.upsaclay.fr/competitions/18604

## 2 Model architecture
+ See [slides](https://drive.google.com/file/d/1czlTv7kdaZ3Z43NkCiBywU3fzbP9Pu_z/view?usp=sharing). (Updated: 2024-10-31) 

## 3 Usage
### 3.1 Train vision-language model:
+ Pretrain on unified dataset:
  + `python main_task3_pretrain.py --lora`
+ Train on classification task: 
  + `python main_task3.py --lora --ckpt_path saved/20240718_0002-Class-LT-16279-task3-baseline-pretrain-nghsg8mz/ckpt/best34.ckpt --desc_file description.csv`

### 3.2 Train vision model: 
+ Single-view model to fientune the image encoder: 
  + `python main_task1.py --lora`
+ Multi-view model with frozen image encoder:
  + `python main_task1_mv.py --lora --ckpt_path saved/20240722_1142-Class-LT-16411-task1-baseline-5hjh936c/ckpt/best92.ckpt --freeze_encoder image`


## Citation
If you find this repository useful, please cite the following:
```
@misc{MICCAI2024-ZeroShotCXR,
  author = {Yuyan Ge},
  title = {MICCAI2024-ZeroShotCXR},
  year = {2024},
  howpublished = {\url{https://github.com/Glourier/MICCAI2024-ZeroShotCXR}},
}
```

