# Dataset for task3, vision-language model
# 2024-07-04 by xtc

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split, KFold
from transformers import AutoTokenizer, DataCollatorWithPadding
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import WeightedRandomSampler


def get_transforms(img_size=224):  # TODO: increase img_size to 1024
    transform_train = A.Compose([
        A.Equalize(p=0.5),
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    transform_test = A.Compose([
        A.Equalize(p=1.0),
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return transform_train, transform_test


def read_txt(data_dir, file_name):
    with open(os.path.join(data_dir, file_name), 'r') as f:
        return [name.strip() for name in f]


## Part1: vanilla vision-language model training
class VLDataset(Dataset):
    def __init__(self, dataframe, img_dir, columns=None, transform=None, desc_df=None):
        # Vision-language dataset
        # dataframe['label'] = dataframe[columns].values.tolist()
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_names = columns
        self.n_classes = len(columns)
        self.desc_df = desc_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        class_id = random.choice(self.class_names)
        label = torch.tensor([study[class_id]]).squeeze().long()
        if self.desc_df is not None:
            desc_row = self.desc_df[self.desc_df['class_names'] == class_id].iloc[0]
            desc_text = desc_row['description'] + " " + desc_row['key_features']
            text = f"Is {class_id} seen in the image? {class_id}'s description: {desc_text} "
        else:
            text = random.choice(
                [f"{class_id}.", f"{class_id}? ", f"Has {class_id}? ", f"Has {class_id}. ", f"{class_id} seen. ",
                 f"{class_id} is seen. "])  # TODO: Find better text presentation

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class VLDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 class_names=None, batch_size=32, img_size=224, seed=0):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)
        self.transform_train, self.transform_test = get_transforms(img_size)

        self.class_names = class_names
        self.seed = seed

    def setup(self, stage=None, fold_idx=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        if fold_idx is None:
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=self.seed)
        else:
            assert fold_idx < 5, "Fold index should be in [0, 1, 2, 3, 4]."
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            folds = list(kf.split(df))
            train_indices, val_indices = folds[fold_idx]
            train_df = df.iloc[train_indices]
            val_df = df.iloc[val_indices]

        if self.desc_file is not None:
            desc_df = pd.read_csv(os.path.join(self.data_dir, self.desc_file))
        else:
            desc_df = None

        self.train_dataset = VLDataset(train_df, self.img_dir, columns=self.class_names,
                                       transform=self.transform_train, desc_df=desc_df)
        self.val_dataset = VLDataset(val_df, self.img_dir, columns=self.class_names, transform=self.transform_test,
                                     desc_df=desc_df)
        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")
        print(f"Training with fold {fold_idx}.")

        # # Save train_df and val_df for temporal testing
        train_df.to_csv(os.path.join(self.data_dir, f'train_df_fold{fold_idx}.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, f'val_df_fold{fold_idx}.csv'), index=False)
        print(f"Saved datasets to {self.data_dir}.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, texts, labels, dicom_ids = zip(*batch)
        images = torch.stack(images)
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.stack(labels)
        item = {'images': images, 'texts': texts, 'input_ids': input_ids, 'attention_masks': attention_mask,
                'labels': labels, 'dicom_ids': dicom_ids}
        return item


## Part2: Inference
class InferenceDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, class_name=None, desc_df=None):
        # For inference purpose only, need name of the subject, no label
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_name = class_name
        self.desc_df = desc_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        name = study['dicom_id']
        # text = random.choice(
        #     [f"{self.class_name}. ", f"{self.class_name}? ", f"Has {self.class_name}? ", f"Has {self.class_name}. ",
        #      f"{self.class_name} seen. ",
        #      f"{self.class_name} is seen. "])
        text = f"Is {self.class_name} seen in the image? {self.class_name}'s description: "
        if self.desc_df is not None:
            desc_row = self.desc_df[self.desc_df['class_names'] == self.class_name].iloc[0]
            desc_text = desc_row['description'] + " " + desc_row['key_features']
            text += desc_text

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, name


class InferenceDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 batch_size=32, img_size=224, class_name=None):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)

        self.transform_test = get_transforms(img_size)[1]
        self.class_name = class_name

    def setup(self, stage=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        if self.desc_file is not None:
            desc_df = pd.read_csv(os.path.join(self.data_dir, self.desc_file))
        else:
            desc_df = None
        self.dataset = InferenceDataset(df, self.img_dir, transform=self.transform_test, class_name=self.class_name,
                                        desc_df=desc_df)
        print(f"Loaded {len(self.dataset)} inference samples.")

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, texts, names = zip(*batch)
        images = torch.stack(images)
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        item = {'images': images, 'texts': texts, 'input_ids': input_ids, 'attention_masks': attention_mask,
                'names': names}
        return item


## Part3: 4 datasets pretraining
class UnifiedDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 class_names=None, batch_size=32, img_size=224, seed=0):
        super().__init__()
        # Data loader for task3, aggregates 4 datasets.
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)
        self.transform_train, self.transform_test = get_transforms(img_size)

        self.class_names = class_names
        self.seed = seed

    def setup(self, stage=None):
        data_lt = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        data_lt_train, data_lt_val = train_test_split(data_lt, test_size=0.2, random_state=self.seed)
        train_lt = VLDataset(data_lt_train, self.img_dir, columns=self.class_names,
                             transform=self.transform_train)
        val_lt = VLDataset(data_lt_val, self.img_dir, columns=self.class_names, transform=self.transform_test)

        # Other datasets come to help!
        train_hash2 = Hash2Dataset(data_dir=self.data_dir, img_dir=self.img_dir, transform=self.transform_train)
        train_facts = FactsDataset(data_dir=self.data_dir, img_dir=self.img_dir, transform=self.transform_train)
        train_label93 = Label93Dataset(data_dir=self.data_dir, img_dir=self.img_dir, transform=self.transform_train)

        self.train_dataset = ConcatDataset([train_hash2, train_facts, train_label93, train_lt])
        self.val_dataset = val_lt

        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")
        print(f"Training samples consists of {len(train_hash2)} hash2, {len(train_facts)} facts, {len(train_label93)} label93, {len(train_lt)} LT.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, texts, labels, dicom_ids = zip(*batch)
        images = torch.stack(images)
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.stack(labels)
        item = {'images': images, 'texts': texts, 'input_ids': input_ids, 'attention_masks': attention_mask,
                'labels': labels, 'dicom_ids': dicom_ids}
        return item


class Hash2Dataset(Dataset):
    def __init__(self, data_dir, img_dir, transform=None):
        # Hash2 dataset, pos/neg answers
        dataframe = pd.read_pickle(os.path.join(data_dir, 'hash2_whole.pkl'))

        dataframe = dataframe.dropna(subset=['pos_repr_facts'])
        dataframe = dataframe.dropna(subset=['neg_repr_facts'])
        dataframe = dataframe[dataframe['pos_repr_facts'].apply(len) > 0]
        dataframe = dataframe[dataframe['neg_repr_facts'].apply(len) > 0]

        self.data = dataframe.to_dict(orient='records')
        self.facts = read_txt(data_dir, "hash2_text.txt")
        self.transform = transform
        self.img_dir = img_dir
        self.n_classes = len(self.facts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')

        label = random.choice([0, 1])
        if label == 1:
            fact = self.facts[random.choice(study['pos_repr_facts'])]
        else:
            fact = self.facts[random.choice(study['neg_repr_facts'])]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])
        label = torch.tensor(label).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class FactsDataset(Dataset):
    def __init__(self, data_dir, img_dir, transform=None):
        # Datasets from original facts, only positive answers
        dataframe = pd.read_pickle(os.path.join(data_dir, 'facts_whole.pkl'))
        dataframe = dataframe.dropna(subset=['fact_idxs'])
        dataframe = dataframe[dataframe['fact_idxs'].apply(len) > 0]

        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        # self.facts = read_txt(data_dir, "facts_text.txt")  # Low memory error
        self.facts = pd.read_pickle(os.path.join(data_dir, "facts_text.pkl"))
        self.n_classes = len(self.facts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')

        fact_id = random.choice(study['fact_idxs'])
        fact = self.facts[fact_id]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])
        label = torch.tensor(1).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class Label93Dataset(Dataset):
    def __init__(self, data_dir, img_dir, transform=None):
        # Label93 dataset, facts are from label-based classes,
        dataframe = pd.read_pickle(os.path.join(data_dir, 'label93_whole.pkl'))
        dataframe = dataframe.dropna(subset=['pos_ids'])
        dataframe = dataframe.dropna(subset=['neg_ids'])
        dataframe = dataframe[dataframe['pos_ids'].apply(len) > 0]
        dataframe = dataframe[dataframe['neg_ids'].apply(len) > 0]

        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.facts = read_txt(data_dir, "label93_text.txt")
        self.n_classes = len(self.facts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')

        label = random.choice([0, 1])
        if label == 1:
            fact = self.facts[random.choice(study['pos_ids'])]
        else:
            fact = self.facts[random.choice(study['neg_ids'])]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])  # TODO: Find better text presentation
        label = torch.tensor(label).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']

