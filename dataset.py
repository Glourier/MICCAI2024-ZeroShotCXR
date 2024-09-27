# Load dataset
# 2024-07-02 by xtc

import os
import pdb
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence


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
        # A.Equalize(p=0.5),
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        # A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.2),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Equalize(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    transform_test = A.Compose([
        # A.Equalize(p=1.0),
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return transform_train, transform_test


## Version1: Multi-label classification
# class ImageDataset(Dataset):
#     def __init__(self, dataframe, img_dir, columns=None, transform=None):
#         dataframe['label'] = dataframe[columns].values.tolist()
#         self.data = dataframe.to_dict(orient='records')
#         self.transform = transform
#         self.img_dir = img_dir
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         study = self.data[idx]
#         img_name = os.path.join(self.img_dir, study['fpath'])
#         image = Image.open(img_name).convert('RGB')
#         label = torch.tensor(study['label']).long()
#
#         if self.transform:
#             # # Augmentation 1
#             # image = self.transform(image)
#
#             ##  Augmentation 2
#             image = np.array(image)
#             augmented = self.transform(image=image)
#             image = augmented['image']
#
#         return image, label, study['dicom_id']
#
#
# class ImageDataModule(LightningDataModule):
#     def __init__(self, data_dir, csv_file, img_dir, class_names=None, batch_size=32, seed=0, img_size=224):
#         super().__init__()
#
#         self.data_dir = data_dir
#         self.csv_file = csv_file
#         self.img_dir = img_dir
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.transform_train, self.transform_test = get_transforms(img_size=img_size)
#         self.class_names = class_names
#         self.seed = seed
#
#     def setup(self, stage=None):
#         df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
#         train_df, val_df = train_test_split(df, test_size=0.2, random_state=self.seed)
#
#         self.train_dataset = ImageDataset(train_df, self.img_dir, columns=self.class_names,
#                                           transform=self.transform_train)
#         self.val_dataset = ImageDataset(val_df, self.img_dir, columns=self.class_names, transform=self.transform_test)
#         print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")
#
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
#
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)


## Version2: Extended Version 1 by adding support to just single class
class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, columns=None, transform=None):
        dataframe['label'] = dataframe[columns].values.tolist()
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(study['label']).long()
        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, study['dicom_id']


class ImageDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, class_names=None, batch_size=32, seed=0, img_size=224):
        super().__init__()
        # Binary classification of just 1 class, to do balanced sampling in training

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform_train, self.transform_test = get_transforms(img_size=img_size)
        self.class_names = class_names
        self.seed = seed
        self.sampler = None

    def setup(self, stage=None, fold_idx=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        if fold_idx is not None:
            assert fold_idx < 5, "Fold index should be in [0, 1, 2, 3, 4]."
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            folds = list(kf.split(df))
            train_indices, val_indices = folds[fold_idx]
            train_df = df.iloc[train_indices]
            val_df = df.iloc[val_indices]
        else:
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=self.seed)

        if len(self.class_names) == 1:  # If single class, do balanced sampling; otherwise, do average sampling
            self.create_sampler(train_df)

        self.train_dataset = ImageDataset(train_df, self.img_dir, columns=self.class_names,
                                          transform=self.transform_train)
        self.val_dataset = ImageDataset(val_df, self.img_dir, columns=self.class_names,
                                        transform=self.transform_test)
        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")

    def train_dataloader(self):
        if self.sampler is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8,
                              sampler=self.sampler)
        else:  # No sampler
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def create_sampler(self, df):
        # Create a sampler for balanced sampling
        labels = df[self.class_names[0]].values
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        print("Using balanced sampling for training...")


class InferenceDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        # For inference purpose only, need name of the subject, no label
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        img_name = os.path.join(self.img_dir, study['fpath'])
        image = Image.open(img_name).convert('RGB')
        name = study['dicom_id']

        if self.transform:
            ## Augmentation 1
            # image = self.transform(image)

            ##  Augmentation 2
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, name


class InferenceDataModule(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, batch_size=32, img_size=224):
        super().__init__()

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform_test = get_transforms(img_size)[1]

    def setup(self, stage=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        self.dataset = InferenceDataset(df, self.img_dir, transform=self.transform_test)
        print(f"Loaded {len(self.dataset)} inference samples.")

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)


## Version3: Finetune on balanced dataset, each epoch balance one class (because I cannot balance all classes at the same time)
class ImageDataModule_Balanced(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, class_names=None, batch_size=32, seed=0, img_size=224):
        super().__init__()
        # Balanced sampling of one class one epoch, iterate over all classes

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform_train, self.transform_test = get_transforms(img_size=img_size)
        self.class_names = class_names
        self.seed = seed
        self.sampler = None
        self.current_epoch = 0

    def setup(self, stage=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        self.train_df, self.val_df = train_test_split(df, test_size=0.2, random_state=self.seed)
        self.train_dataset = ImageDataset(self.train_df, self.img_dir, columns=self.class_names,
                                          transform=self.transform_train)
        self.val_dataset = ImageDataset(self.train_df, self.img_dir, columns=self.class_names,
                                        transform=self.transform_test)
        print(f"Loaded {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")

    def train_dataloader(self):
        class_id = self.current_epoch % len(self.class_names)
        self.create_sampler(self.train_df, class_id=class_id)
        self.current_epoch += 1
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8,
                          sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def create_sampler(self, df, class_id):
        # Create a sampler for balanced sampling
        labels = df[self.class_names[class_id]].values
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"Using balanced sampling for training, class {class_id}: '{self.class_names[class_id]}'...")


## Version4: Multi-view classification
class ImageDataset_MV(Dataset):
    def __init__(self, dataframe, img_dir, columns=None, transform=None, max_views=3):
        dataframe['label'] = dataframe[columns].values.tolist()
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.max_views = max_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
        label = torch.tensor(study['label']).long()
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
        # print(f"Index: {idx}")

        return {'images': images, 'label': label, 'study_id': study['study_id'], 'study_index': idx}


class ImageDataModule_MV(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, class_names=None, class_name=None, batch_size=32, seed=0,
                 img_size=224, max_views=3):
        super().__init__()
        # Balanced sampling of one class one epoch, iterate over all classes

        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform_train, self.transform_test = get_transforms(img_size=img_size)
        self.class_names = class_names
        self.seed = seed
        self.max_views = max_views
        self.class_name = class_name

    def setup(self, stage=None, fold_idx=None):
        df_original = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        df = df_original.groupby('study_id').agg(
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

        if self.class_name is not None:  # Balanced sampling of the specified class_name
            self.create_sampler(train_df, self.class_name)

        self.train_dataset = ImageDataset_MV(train_df, self.img_dir, columns=self.class_names,
                                             transform=self.transform_train, max_views=self.max_views)
        self.val_dataset = ImageDataset_MV(val_df, self.img_dir, columns=self.class_names,
                                           transform=self.transform_test, max_views=self.max_views)
        print(
            f"Loaded {len(self.train_dataset)} study_id training samples, {len(self.val_dataset)} study_id validation samples.")

        # Save train_df and val_df
        # Filter out the original df to get the train_df and val_df
        train_df_save = df_original[df_original['study_id'].isin(train_df['study_id'])]
        val_df_save = df_original[df_original['study_id'].isin(val_df['study_id'])]
        if fold_idx is not None:
            train_df_save.to_csv(os.path.join(self.data_dir, f'train_df_fold{fold_idx}.csv'), index=False)
            val_df_save.to_csv(os.path.join(self.data_dir, f'val_df_fold{fold_idx}.csv'), index=False)
        else:
            train_df_save.to_csv(os.path.join(self.data_dir, 'train_df_foldNone.csv'), index=False)
            val_df_save.to_csv(os.path.join(self.data_dir, 'val_df_foldNone.csv'), index=False)
        print(f"Saved datasets to {self.data_dir}.")

    def train_dataloader(self):
        if self.class_name is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8,
                              collate_fn=self.collate_fn, sampler=self.sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                              collate_fn=self.collate_fn)

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
        # "batch" is a list of dict
        images = [item['images'] for item in batch]
        labels = [item['label'] for item in batch]
        study_ids = [item['study_id'] for item in batch]
        study_index = [item['study_index'] for item in batch]
        masks = []
        for seq in images:
            masks.append(torch.tensor([0] * len(seq), device=seq.device))
        padded_images = pad_sequence(images, batch_first=True, padding_value=0)  # [B, V, C, H, W]
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=1)  # [B, V]
        padded_masks = padded_masks.bool()
        labels = torch.stack(labels)
        return {'images': padded_images, 'masks': padded_masks, 'labels': labels, 'study_ids': study_ids,
                'study_index': study_index}


class InferenceDataset_MV(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, max_views=3):
        # For inference purpose only, need name of the subject, no label
        self.data = dataframe.to_dict(orient='records')
        self.transform = transform
        self.img_dir = img_dir
        self.max_views = max_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study = self.data[idx]
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

        return images, study['study_id']


class InferenceDataModule_MV(LightningDataModule):
    def __init__(self, data_dir, csv_file, img_dir, batch_size=32, img_size=224, max_views=3, seed=0):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform_test = get_transforms(img_size)[1]
        self.max_views = max_views
        self.seed = seed

    def setup(self, stage=None):
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        df_grouped = df.groupby('study_id').agg({'study_id': 'first', 'fpath': list})
        self.dataset = InferenceDataset_MV(df_grouped, self.img_dir, transform=self.transform_test,
                                           max_views=self.max_views)
        print(f"Loaded {len(self.dataset)} study_id inference samples.")

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        images, study_ids = zip(*batch)
        masks = []
        for seq in images:
            masks.append(torch.tensor([0] * len(seq), device=seq.device))
        padded_images = pad_sequence(images, batch_first=True, padding_value=0)  # [B, V, C, H, W]
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=1)  # [B, V]
        padded_masks = padded_masks.bool()
        return {'images': padded_images, 'image_masks': padded_masks, 'study_ids': study_ids}
