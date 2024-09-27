# Pretrained ViT model.
# 2024-07-02 by xtc

import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel
from pytorch_lightning import LightningModule
from torchmetrics.classification import F1Score, AveragePrecision
from peft import get_peft_model, LoraConfig
from utils import utils as utils


def apply_lora(model, rank=4, target_modules=None):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)
    return model


def apply_lora_to_model(model, model_name, rank=4):
    if model_name == 'vit':
        target_modules = ["attention.query", "attention.key", "attention.value"]
    elif model_name == 'dino':
        target_modules = ["attn.qkv"]
    elif model_name == 'rad-dino':
        target_modules = ["attention.query", "attention.key", "attention.value"]
    elif model_name == 'bert':
        target_modules = ["attention.self.query", "attention.self.key", "attention.self.value"]
    elif model_name == 'transformer_encoder':
        target_modules = ['self_attn']
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return apply_lora(model, rank=rank, target_modules=target_modules)


class ViT(LightningModule):
    def __init__(self, n_classes, model_url=None, epochs=100, steps_per_epoch=None, lr=1e-4, weight_decay=1e-5,
                 criterion=None, with_lora=False, lora_rank=4, save_dir=None, backbone='dino', dropout=0):
        super().__init__()
        self.backbone = backbone
        if backbone == 'vit':
            # Version 1: ViT
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features=self.vit.config.hidden_size, out_features=512, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=n_classes, bias=True)
            )
        elif backbone == 'dino':
            # Version 2: Dino-v2
            self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features=self.vit.embed_dim, out_features=512, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=n_classes, bias=True)
            )
        elif backbone == 'rad-dino':
            # # Vsersion 3: Rad-DINO
            # Pretrained on CXR datasets, model DINOv2, but without register tokens
            self.vit = AutoModel.from_pretrained("microsoft/rad-dino")
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features=self.vit.config.hidden_size, out_features=512, bias=True),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=512, out_features=n_classes, bias=True)
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}, please choose from ['vit', 'dino', 'rad-dino'].")

        if with_lora:
            self.vit = apply_lora_to_model(self.vit, model_name=backbone, rank=lora_rank)

        self.set_forward_function()

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.criterion = criterion  # TODO: make criterion
        self.n_classes = n_classes
        self.save_dir = save_dir
        self.dropout = dropout

        # Functions
        self.sigmoid = nn.Sigmoid()
        if n_classes > 1:
            self.f1 = F1Score(num_labels=n_classes, zero_division=0, task='multilabel')
            self.map = AveragePrecision(num_labels=n_classes, task='multilabel')
        elif n_classes == 1:
            self.f1 = F1Score(zero_division=0, task='binary')
            self.map = AveragePrecision(task='binary')
        else:
            raise ValueError(f"Invalid number of classes: {n_classes}")

        self.val_outputs = []

    def set_forward_function(self):
        if self.backbone == 'vit':
            self.forward_fn = self.forward_vit
        elif self.backbone == 'dino':
            self.forward_fn = self.forward_dino
        elif self.backbone == 'rad-dino':
            self.forward_fn = self.forward_rad_dino
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}, please choose from ['vit', 'dino', 'rad-dino'].")

    def forward_vit(self, x):
        outputs = self.vit(x)
        pooled_output = outputs[1]  # == LayerNorm(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier_head(pooled_output)
        return logits

    def forward_dino(self, x):
        pooled_output = self.vit(x)
        logits = self.classifier_head(pooled_output)
        return logits

    def forward_rad_dino(self, x):
        pooled_output = self.vit(x).pooler_output
        logits = self.classifier_head(pooled_output)
        return logits

    def forward(self, x):
        return self.forward_fn(x)

    def training_step(self, batch, batch_idx):
        images, labels, _ = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())
        probs = self.sigmoid(logits)
        preds = torch.round(probs)

        # Log metrics
        f1 = self.f1(preds, labels)
        ap = self.map(probs, labels)
        gradnorm_vit = utils.get_grad_norm(self.vit)
        gradnorm_head = utils.get_grad_norm(self.classifier_head)
        self.log('train/gradnorm_vit', gradnorm_vit, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/gradnorm_head', gradnorm_head, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/map', ap, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, names = batch
        logits = self(images)
        loss = self.criterion(logits, labels.float())
        outputs = {'loss': loss, 'labels': labels, 'logits': logits, 'names': names}
        self.val_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        loss = torch.stack([out['loss'] for out in self.val_outputs]).mean()
        logits = torch.cat([out['logits'] for out in self.val_outputs], dim=0)
        labels = torch.cat([out['labels'] for out in self.val_outputs], dim=0)
        names = [name for out in self.val_outputs for name in out['names']]
        probs = self.sigmoid(logits)
        preds = torch.round(probs)
        f1 = self.f1(preds, labels)
        ap = self.map(probs, labels)
        save_probs(names, probs, labels, os.path.join(self.save_dir, 'val_last.csv'))
        self.log('val/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/map_epoch', ap, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = 0.1 * total_steps

        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        combined_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, cosine_scheduler])
        scheduler = {
            'scheduler': combined_scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler]


def save_probs(dicom_ids, probs, labels, file_name):
    probs = probs.cpu().numpy().astype('float16')
    labels = labels.cpu().numpy().astype('int8')
    n_classes = probs.shape[1]
    data = {'dicom_ids': dicom_ids}
    columns = ['dicom_ids']
    for i in range(n_classes):
        columns += [f'probs_{i}', f'labels_{i}']
        data[f'probs_{i}'] = probs[:, i]
        data[f'labels_{i}'] = labels[:, i]
    pd.DataFrame(data, columns=columns).to_csv(file_name, header=True, index=False)
