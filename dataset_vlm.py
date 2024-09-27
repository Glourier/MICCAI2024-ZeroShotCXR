# Dataset for task3, vision-language model
# 2024-07-04 by xtc

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split, KFold
from transformers import AutoTokenizer, DataCollatorWithPadding
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import WeightedRandomSampler


def get_transforms(img_size=224):  # TODO: increase img_size to 1024
    # # Augmentation 1
    # self.transform_train = transforms.Compose([
    #     # transforms.Resize((1024, 1024)),
    #     transforms.Resize((224, 224)),  # TODO: 1024
    #     transforms.ColorJitter(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(degrees=10),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # self.transform_test = transforms.Compose([
    #     # transforms.Resize((1024, 1024)),
    #     transforms.Resize((224, 224)),  # TODO: 1024
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # # Augmentation 2
    transform_train = A.Compose([
        A.Equalize(p=0.5),
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        # A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.2),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
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


# def read_txt(data_dir, file_name):
#     names = pd.read_csv(os.path.join(data_dir, file_name), header=None).values
#     names = [name[0] for name in names]
#     return names  # List
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
        #      f"{self.class_name} is seen. "])  # TODO: Find better text presentation
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
        # Data loader for task3, aggeragates 4 datasets.

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
        # Added 3 levels of class45
        train_class45l1 = class45l1Dataset(data_dir=self.data_dir, img_dir=self.img_dir, transform=self.transform_train)
        train_class45l2 = class45l2Dataset(data_dir=self.data_dir, img_dir=self.img_dir, transform=self.transform_train)
        train_class45l3 = class45l3Dataset(data_dir=self.data_dir, img_dir=self.img_dir, transform=self.transform_train)

        # self.train_dataset = ConcatDataset([train_hash2, train_facts, train_label93, train_lt])
        self.train_dataset = ConcatDataset([train_lt, train_class45l2, train_class45l3])
        self.val_dataset = val_lt

        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")
        # print(
        #     f"Training samples consists of {len(train_hash2)} hash2, {len(train_facts)} facts, {len(train_label93)} label93, {len(train_lt)} LT.")

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


## Part 4: Few-shot finetuning (FS)
class FSDataset(Dataset):
    def __init__(self, dataframe, img_dir, class_name=None, transform=None, desc_df=None, class_names=None):
        # For few-shot learning finetuning, 50% choose from class_names, 50% use class_name
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_name = class_name
        self.desc_df = desc_df
        self.class_names = class_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        if self.class_names is not None:
            # 50% choose from class_names, 50% use class_name
            if random.random() < 0.5:
                class_id = random.choice(self.class_names)
            else:
                class_id = self.class_name
        else:
            class_id = self.class_name
        label = torch.tensor([study[class_id]]).squeeze().long()
        if self.desc_df is not None:
            desc_row = self.desc_df[self.desc_df['class_names'] == class_id].iloc[0]
            desc_text = desc_row['description'] + " " + desc_row['key_features']
            text = f"Is {class_id} seen in the image? {class_id}'s description: {desc_text} "
        else:
            text = random.choice(
                [f"{class_id}.", f"{class_id}? ", f"Has {class_id}? ", f"Has {class_id}. ",
                 f"{class_id} seen. ",
                 f"{class_id} is seen. "])

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class FSDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 class_name=None, class_names=None, batch_size=32, img_size=224, seed=0):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)
        self.transform_train, self.transform_test = get_transforms(img_size)

        self.class_name = class_name  # target class
        self.class_names = class_names  # other classes
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

        self.create_sampler(train_df, self.class_name)
        self.train_dataset = FSDataset(train_df, self.img_dir, class_name=self.class_name,
                                       transform=self.transform_train, desc_df=desc_df,
                                       class_names=self.class_names)
        self.val_dataset = FSDataset(val_df, self.img_dir, class_name=self.class_name, transform=self.transform_test,
                                     desc_df=desc_df)
        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")

        # # Save train_df and val_df for temporal testing
        # train_df.to_csv(os.path.join(self.data_dir, 'train_df.csv'), index=False)
        # val_df.to_csv(os.path.join(self.data_dir, 'val_df.csv'), index=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8,
                          collate_fn=self.collate_fn, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def create_sampler(self, df, class_name):
        print(f"Using balanced sampling for training, class '{class_name}'...")
        labels = df[class_name].values
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

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


## Part 5: Multi-view model
class VLDataset_MV(Dataset):
    def __init__(self, dataframe, img_dir, columns=None, transform=None, desc_df=None, max_views=3):
        # Vision-language dataset
        # dataframe['label'] = dataframe[columns].values.tolist()
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_names = columns
        self.n_classes = len(columns)
        self.desc_df = desc_df
        self.max_views = max_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]

        # For image
        img_names = study['fpath']
        if len(img_names) > self.max_views:
            img_names = random.sample(img_names, self.max_views)
        images = []
        for img_name in img_names:
            image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            if self.transform:
                image = np.array(image)
                augmented = self.transform(image=image)
                image = augmented['image']
            images.append(image)
        images = torch.stack(images)  # [V, C, H, W]

        # For text
        class_id = random.choice(self.class_names)
        label = torch.tensor([study[class_id]]).squeeze().long()
        if self.desc_df is not None:
            desc_row = self.desc_df[self.desc_df['class_names'] == class_id].iloc[0]
            desc_text = desc_row['description'] + " " + desc_row['key_features']
            text = f"Is {class_id} seen in the image? {class_id}'s description: {desc_text} "
        else:
            text = random.choice(
                [f"{class_id}.", f"{class_id}? ", f"Has {class_id}? ", f"Has {class_id}. ", f"{class_id} seen. ",
                 f"{class_id} is seen. "])

        return images, text, label, study['study_id']


class VLDataModule_MV(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 class_names=None, batch_size=32, img_size=224, seed=0, max_views=3):
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
        self.max_views = max_views

    def setup(self, stage=None, fold_idx=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        df = df.groupby('study_id').agg(
            {'subject_id': 'first', 'study_id': 'first', 'fpath': list, 'dicom_id': list,
             **{col: 'first' for col in self.class_names}})
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

        self.train_dataset = VLDataset_MV(train_df, self.img_dir, columns=self.class_names,
                                          transform=self.transform_train, desc_df=desc_df, max_views=self.max_views)
        self.val_dataset = VLDataset_MV(val_df, self.img_dir, columns=self.class_names, transform=self.transform_test,
                                        desc_df=desc_df, max_views=self.max_views)
        print(
            f"Loaded {len(self.train_dataset)} study_id training samples, {len(self.val_dataset)} study_id validation samples.")

        # # Save train_df and val_df for temporal testing
        # train_df.to_csv(os.path.join(self.data_dir, 'train_df.csv'), index=False)
        # val_df.to_csv(os.path.join(self.data_dir, 'val_df.csv'), index=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, texts, labels, dicom_ids = zip(*batch)

        # For image
        masks = []
        for seq in images:
            masks.append(torch.tensor([0] * len(seq), device=seq.device))
        padded_images = pad_sequence(images, batch_first=True, padding_value=0)  # [B, V, C, H, W]
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=1)  # [B, V]
        padded_masks = padded_masks.bool()

        # For text
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        labels = torch.stack(labels)
        item = {'images': padded_images, 'image_masks': padded_masks, 'texts': texts, 'input_ids': input_ids,
                'attention_masks': attention_mask,
                'labels': labels, 'dicom_ids': dicom_ids}
        return item


## Part 6: Inference of multi-view model
class InferenceDataset_MV(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, class_name=None, desc_df=None, max_views=3):
        # For inference purpose only, need name of the subject, no label
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_name = class_name
        self.desc_df = desc_df
        self.max_views = max_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]

        # For image
        img_names = study['fpath']
        if len(img_names) > self.max_views:
            img_names = random.sample(img_names, self.max_views)
        images = []
        for img_name in img_names:
            image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            if self.transform:
                image = np.array(image)
                augmented = self.transform(image=image)
                image = augmented['image']
            images.append(image)
        images = torch.stack(images)  # [V, C, H, W]

        # For text
        if self.desc_df is not None:
            desc_row = self.desc_df[self.desc_df['class_names'] == self.class_name].iloc[0]
            desc_text = desc_row['description'] + " " + desc_row['key_features']
            text = f"Is {self.class_name} seen in the image? {self.class_name}'s description: {desc_text} "
        else:
            text = random.choice(
                [f"{self.class_name}.", f"{self.class_name}? ", f"Has {self.class_name}? ", f"Has {self.class_name}. ",
                 f"{self.class_name} seen. ",
                 f"{self.class_name} is seen. "])

        return images, text, study['study_id']


class InferenceDataModule_MV(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 batch_size=32, img_size=224, class_name=None, max_views=3):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)

        self.transform_test = get_transforms(img_size)[1]
        self.class_name = class_name
        self.max_views = max_views

    def setup(self, stage=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        df_grouped = df.groupby('study_id').agg(
            {'subject_id': 'first', 'study_id': 'first', 'fpath': list, 'dicom_id': list})
        if self.desc_file is not None:
            desc_df = pd.read_csv(os.path.join(self.data_dir, self.desc_file))
        else:
            desc_df = None
        self.dataset = InferenceDataset_MV(df_grouped, self.img_dir, transform=self.transform_test,
                                           class_name=self.class_name,
                                           desc_df=desc_df, max_views=self.max_views)
        print(f"Loaded {len(self.dataset)} study_id inference samples.")

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, texts, study_ids = zip(*batch)

        # For image
        masks = []
        for seq in images:
            masks.append(torch.tensor([0] * len(seq), device=seq.device))
        padded_images = pad_sequence(images, batch_first=True, padding_value=0)  # [B, V, C, H, W]
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=1)  # [B, V]
        padded_masks = padded_masks.bool()

        # For text
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        item = {'images': padded_images, 'image_masks': padded_masks, 'texts': texts, 'input_ids': input_ids,
                'attention_masks': attention_mask, 'study_ids': study_ids}
        return item


## Part 7: Few-shot finetuning (FS) of Multi-view (MV) model
class MV_FSDataset(Dataset):
    def __init__(self, dataframe, img_dir, class_name=None, transform=None, desc_df=None, class_names=None,
                 max_views=3):
        # For few-shot learning finetuning, 50% choose from class_names, 50% use class_name
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_name = class_name
        self.desc_df = desc_df
        self.class_names = class_names
        self.max_views = max_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]

        # For image
        img_names = study['fpath']
        if len(img_names) > self.max_views:
            img_names = random.sample(img_names, self.max_views)
        images = []
        for img_name in img_names:
            image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            if self.transform:
                image = np.array(image)
                augmented = self.transform(image=image)
                image = augmented['image']
            images.append(image)
        images = torch.stack(images)  # [V, C, H, W]

        # For text
        if self.class_names is not None:
            # 50% choose from class_names, 50% use class_name
            if random.random() < 0.5:
                class_id = random.choice(self.class_names)
            else:
                class_id = self.class_name
        else:
            class_id = self.class_name
        label = torch.tensor([study[class_id]]).squeeze().long()
        if self.desc_df is not None:
            desc_row = self.desc_df[self.desc_df['class_names'] == class_id].iloc[0]
            desc_text = desc_row['description'] + " " + desc_row['key_features']
            text = f"Is {class_id} seen in the image? {class_id}'s description: {desc_text} "
        else:
            text = random.choice(
                [f"{class_id}.", f"{class_id}? ", f"Has {class_id}? ", f"Has {class_id}. ",
                 f"{class_id} seen. ",
                 f"{class_id} is seen. "])

        return images, text, label, study['study_id']


class MV_FSDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 class_name=None, class_names=None, batch_size=32, img_size=224, seed=0, max_views=3):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)
        self.transform_train, self.transform_test = get_transforms(img_size)

        self.class_name = class_name  # target class
        self.class_names = class_names  # other classes
        self.class_names_all = class_names + [class_name]
        self.seed = seed
        self.max_views = max_views

    def setup(self, stage=None, fold_idx=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        df = df.groupby('study_id').agg(
            {'subject_id': 'first', 'study_id': 'first', 'fpath': list, 'dicom_id': list,
             **{col: 'first' for col in self.class_names_all}})
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

        self.create_sampler(train_df, self.class_name)
        self.train_dataset = MV_FSDataset(train_df, self.img_dir, class_name=self.class_name,
                                          transform=self.transform_train, desc_df=desc_df,
                                          class_names=self.class_names, max_views=self.max_views)
        self.val_dataset = MV_FSDataset(val_df, self.img_dir, class_name=self.class_name, transform=self.transform_test,
                                        desc_df=desc_df, max_views=self.max_views)
        print(
            f"Loaded {len(self.train_dataset)} study_id training samples, {len(self.val_dataset)} study_id validation samples.")

        # # Save train_df and val_df for temporal testing
        # train_df.to_csv(os.path.join(self.data_dir, 'train_df.csv'), index=False)
        # val_df.to_csv(os.path.join(self.data_dir, 'val_df.csv'), index=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8,
                          collate_fn=self.collate_fn, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def create_sampler(self, df, class_name):
        print(f"Using balanced sampling for training, class '{class_name}'...")
        labels = df[class_name].values
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def collate_fn(self, batch):
        images, texts, labels, dicom_ids = zip(*batch)

        # For image
        masks = []
        for seq in images:
            masks.append(torch.tensor([0] * len(seq), device=seq.device))
        padded_images = pad_sequence(images, batch_first=True, padding_value=0)  # [B, V, C, H, W]
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=1)  # [B, V]
        padded_masks = padded_masks.bool()

        # For text
        encoding = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        labels = torch.stack(labels)
        item = {'images': padded_images, 'image_masks': padded_masks, 'texts': texts, 'input_ids': input_ids,
                'attention_masks': attention_mask,
                'labels': labels, 'dicom_ids': dicom_ids}
        return item


## Part 8: Hierarchical concepts
class HierarchyDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, tokenizer_url="microsoft/BiomedVLP-CXR-BERT-specialized",
                 class_names=None, batch_size=32, img_size=224, seed=0):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url, trust_remote_code=True)
        self.transform_train, self.transform_test = get_transforms(img_size)

        self.class_names = class_names
        self.seed = seed

    def setup(self, stage=None, fold_idx=None):
        facts_l1 = read_txt(self.data_dir, "class45l1_text.txt")
        facts_l2 = read_txt(self.data_dir, "class45l2_text.txt")
        facts_l3 = read_txt(self.data_dir, "class45l3_text.txt")
        self.facts = facts_l1 + facts_l2 + facts_l3

        # Make a new column with all facts
        dataframe = pd.read_pickle(os.path.join(self.data_dir, self.csv_file))
        dataframe['pos_l2'] = dataframe['pos_l2'].apply(lambda x: add_n_to_list(x, len(facts_l1)))
        dataframe['neg_l2'] = dataframe['neg_l2'].apply(lambda x: add_n_to_list(x, len(facts_l1)))
        dataframe['pos_l3'] = dataframe['pos_l3'].apply(lambda x: add_n_to_list(x, len(facts_l1) + len(facts_l2)))
        dataframe['neg_l3'] = dataframe['neg_l3'].apply(lambda x: add_n_to_list(x, len(facts_l1) + len(facts_l2)))
        dataframe['pos_l123'] = dataframe['pos_l1'] + dataframe['pos_l2'] + dataframe['pos_l3']
        dataframe['neg_l123'] = dataframe['neg_l1'] + dataframe['neg_l2'] + dataframe['neg_l3']

        if fold_idx is None:
            train_df, val_df = train_test_split(dataframe, test_size=0.2, random_state=self.seed)
        else:
            assert fold_idx < 5, "Fold index should be in [0, 1, 2, 3, 4]."
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            folds = list(kf.split(dataframe))
            train_indices, val_indices = folds[fold_idx]
            train_df = dataframe.iloc[train_indices]
            val_df = dataframe.iloc[val_indices]

        self.train_dataset = HierarchyDataset(train_df, self.img_dir, facts=self.facts, transform=self.transform_train)
        self.val_dataset = HierarchyDataset(val_df, self.img_dir, facts=self.facts, transform=self.transform_test)

        print(f"Using hierarchical concepts for training and validation.")
        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")
        print(f"Number of facts: {len(self.facts)}. ")

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


def add_n_to_list(lst, n):
    return [x + n for x in lst]


class HierarchyDataset(Dataset):
    def __init__(self, dataframe, img_dir, facts, transform=None):
        # Hierarchical concept dataset, facts are from label-based classes, 3 levels: class_names, definition, and key_features

        dataframe = dataframe[dataframe['pos_l123'].apply(len) > 0]
        dataframe = dataframe[dataframe['neg_l123'].apply(len) > 0]
        self.data = dataframe.to_dict(orient='records')
        self.facts = facts
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
            fact = self.facts[random.choice(study['pos_l123'])]
        else:
            fact = self.facts[random.choice(study['neg_l123'])]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])
        label = torch.tensor(label).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class class45l1Dataset(Dataset):
    def __init__(self, data_dir, img_dir, transform=None):
        # Hierarchical concept dataset, facts are from label-based classes, 3 levels: class_names, definition, and key_features
        dataframe = pd.read_pickle(os.path.join(data_dir, 'class45_whole.pkl'))
        dataframe = dataframe.dropna(subset=['pos_l1'])
        dataframe = dataframe.dropna(subset=['neg_l1'])
        dataframe = dataframe[dataframe['pos_l1'].apply(len) > 0]
        dataframe = dataframe[dataframe['neg_l1'].apply(len) > 0]

        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.facts = read_txt(data_dir, "class45l1_text.txt")
        self.n_classes = len(self.facts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')

        label = random.choice([0, 1])
        if label == 1:
            fact = self.facts[random.choice(study['pos_l1'])]
        else:
            fact = self.facts[random.choice(study['neg_l1'])]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])
        label = torch.tensor(label).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class class45l2Dataset(Dataset):
    def __init__(self, data_dir, img_dir, transform=None):
        # Hierarchical concept dataset, facts are from label-based classes, 3 levels: class_names, definition, and key_features
        dataframe = pd.read_pickle(os.path.join(data_dir, 'class45_whole.pkl'))
        dataframe = dataframe.dropna(subset=['pos_l2'])
        dataframe = dataframe.dropna(subset=['neg_l2'])
        dataframe = dataframe[dataframe['pos_l2'].apply(len) > 0]
        dataframe = dataframe[dataframe['neg_l2'].apply(len) > 0]

        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.facts = read_txt(data_dir, "class45l2_text.txt")

        self.n_classes = len(self.facts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')

        label = random.choice([0, 1])
        if label == 1:
            fact = self.facts[random.choice(study['pos_l2'])]
        else:
            fact = self.facts[random.choice(study['neg_l2'])]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])
        label = torch.tensor(label).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


class class45l3Dataset(Dataset):
    def __init__(self, data_dir, img_dir, transform=None):
        # Hierarchical concept dataset, facts are from label-based classes, 3 levels: class_names, definition, and key_features
        dataframe = pd.read_pickle(os.path.join(data_dir, 'class45_whole.pkl'))
        dataframe = dataframe.dropna(subset=['pos_l3'])
        dataframe = dataframe.dropna(subset=['neg_l3'])
        dataframe = dataframe[dataframe['pos_l3'].apply(len) > 0]
        dataframe = dataframe[dataframe['neg_l3'].apply(len) > 0]

        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.facts = read_txt(data_dir, "class45l3_text.txt")
        self.n_classes = len(self.facts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')

        label = random.choice([0, 1])
        if label == 1:
            fact = self.facts[random.choice(study['pos_l3'])]
        else:
            fact = self.facts[random.choice(study['neg_l3'])]
        text = random.choice([f"{fact}.", f"{fact}? ", f"Has {fact}? ", f"Has {fact}. ", f"{fact} seen. ",
                              f"{fact} is seen. "])
        label = torch.tensor(label).long()

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, text, label, study['dicom_id']


## PartX: use simple dataset, just for debug
class VLDataset_debug(Dataset):
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
        text = f"Is {class_id} seen in the image?"

        return image, text, label, study['dicom_id']


class VLDataModule_debug(LightningDataModule):
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
        # train_df.to_csv(os.path.join(self.data_dir, 'train_df.csv'), index=False)
        # val_df.to_csv(os.path.join(self.data_dir, 'val_df.csv'), index=False)

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


## Part11: dataset for CLIP model
class CLIPDataset(Dataset):
    def __init__(self, dataframe, img_dir, columns=None, transform=None, processor=None):
        # Vision-language dataset
        # level: 1 means class names only; 2 means class names + definitions; 3 means class names + definitions + key features
        # dataframe['label'] = dataframe[columns].values.tolist()
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_names = columns
        self.n_classes = len(columns)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor([study[class_name] for class_name in self.class_names]).squeeze().long()  # [C]
        inputs = self.processor(images=image,
                                return_tensors="pt")
        image = inputs['pixel_values'].squeeze(0)

        return image, label, study['dicom_id']


class CLIPDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 class_names=None, batch_size=32, img_size=224, seed=0, processor=None, level=1):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size

        self.class_names = class_names
        self.seed = seed
        self.processor = processor
        self.level = level

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

        # Encode text
        self.texts = get_texts(self.class_names, desc_df, level=self.level)
        encoding = self.processor(text=self.texts,
                                  return_tensors='pt',
                                  padding=True)
        self.input_ids = encoding['input_ids']
        self.attention_mask = encoding['attention_mask']

        # Dataset
        self.train_dataset = CLIPDataset(train_df, self.img_dir, columns=self.class_names, processor=self.processor)
        self.val_dataset = CLIPDataset(val_df, self.img_dir, columns=self.class_names, processor=self.processor)
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
        images, labels, dicom_ids = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        item = {'images': images, 'input_ids': self.input_ids, 'attention_masks': self.attention_mask,
                'labels': labels, 'dicom_ids': dicom_ids, 'texts': self.texts}
        return item


def get_texts(class_names, desc_df, level=1):
    if level == 1:
        texts = [f"{i} is seen in the image. " for i in class_names]
    elif level == 2:
        texts = [f"{i} is seen in the image. {i}'s description: {desc_df[desc_df['class_names'] == i]['description'].values[0]}" for i in class_names]
    elif level == 3:
        texts = [
            f"{i} is seen in the image. {i}'s description: {desc_df[desc_df['class_names'] == i]['description'].values[0]} {desc_df[desc_df['class_names'] == i]['key_features'].values[0]}"
            for i in class_names]
    else:
        raise ValueError("Level should be in [1, 2, 3].")

    return texts



## Part12: dataset for CLIP model, Work with VLM_CLIP model
class VLMCLIPDataset(Dataset):
    def __init__(self, dataframe, img_dir, columns=None, transform=None):
        # Vision-language dataset
        # level: 1 means class names only; 2 means class names + definitions; 3 means class names + definitions + key features
        # dataframe['label'] = dataframe[columns].values.tolist()
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.class_names = columns
        self.n_classes = len(columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor([study[class_name] for class_name in self.class_names]).squeeze().long()  # [C]
        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, study['dicom_id']


class VLMCLIPDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, desc_file=None,
                 class_names=None, batch_size=32, img_size=224, seed=0, model_url_language=None, level=1):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.desc_file = desc_file
        self.batch_size = batch_size

        self.class_names = class_names
        self.seed = seed
        self.level = level
        self.tokenizer = AutoTokenizer.from_pretrained(model_url_language, trust_remote_code=True)
        self.train_transform, self.val_transform = get_transforms(img_size)

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

        # Encode text
        self.texts = get_texts(self.class_names, desc_df, level=self.level)
        encoding = self.tokenizer(self.texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
        self.input_ids = encoding['input_ids']
        self.attention_mask = encoding['attention_mask']
        print(f"texts for VLM-CLIP: length {len(self.texts)}\n  {self.texts}")

        # Dataset
        self.train_dataset = VLMCLIPDataset(train_df, self.img_dir, columns=self.class_names, transform=self.train_transform)
        self.val_dataset = VLMCLIPDataset(val_df, self.img_dir, columns=self.class_names, transform=self.val_transform)
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
        images, labels, dicom_ids = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        item = {'images': images, 'input_ids': self.input_ids, 'attention_masks': self.attention_mask,
                'labels': labels, 'dicom_ids': dicom_ids, 'texts': self.texts}
        return item
