# Multi-view version of ViT model.
# Step1: Extract image features from frozen ViT model separately.
# Step2: Use transformer encoder to aggregate features from different views.
# 2024-07-16 by xtc

import os
import pandas as pd
import math
import torch
import torch.nn as nn
from transformers import ViTModel
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
    elif model_name == 'bert':
        target_modules = ["attention.self.query", "attention.self.key", "attention.self.value"]
    elif model_name == 'transformer_encoder':
        target_modules = ['self_attn']
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return apply_lora(model, rank=rank, target_modules=target_modules)


class ViT_MV(LightningModule):
    def __init__(self, model_url, n_classes, epochs=100, steps_per_epoch=None, lr=1e-4, weight_decay=1e-5,
                 criterion=None, with_lora=False, lora_rank=4, save_dir=None, max_views=4, n_heads=8, n_layers=1,
                 dropout=0, class_id=None):
        super().__init__()

        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.vit.embed_dim).uniform_(-0.0001, 0.0001))
        self.positional_embedding = nn.Parameter(self.sinusoidal_embedding(max_views + 1, self.vit.embed_dim))

        if dropout > 0:
            print(f"Training with dropout {dropout}...")
            encoder_layers = nn.TransformerEncoderLayer(d_model=self.vit.embed_dim, nhead=n_heads, batch_first=True,
                                                        dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features=self.vit.embed_dim, out_features=512, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(in_features=512, out_features=n_classes, bias=True)
            )
        else:
            print("Training with no dropout...")
            encoder_layers = nn.TransformerEncoderLayer(d_model=self.vit.embed_dim, nhead=n_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
            self.classifier_head = nn.Sequential(
                nn.Linear(in_features=self.vit.embed_dim, out_features=512, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=n_classes, bias=True)
            )
        if with_lora:
            self.vit = apply_lora_to_model(self.vit, model_name='dino', rank=lora_rank)
            # self.transformer_encoder = apply_lora_to_model(self.transformer_encoder, model_name='transformer_encoder',
            #                                                rank=lora_rank)

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.criterion = criterion
        self.n_classes = n_classes
        self.save_dir = save_dir
        self.max_views = max_views
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
        if class_id is not None:
            self.class_id = class_id
            self.f1_single = F1Score(zero_division=0, task='binary')
            self.map_single = AveragePrecision(task='binary')
        self.val_outputs = []

    def forward(self, x, mask):
        bs, n_views, _, _, _ = x.size()
        features = []
        for i in range(n_views):
            features.append(self.vit(x[:, i]))
        features = torch.stack(features, dim=1)  # [B, V, dim]
        features = self.fuse_views(features, mask)  # [B, dim]
        logits = self.classifier_head(features)
        return logits

    def fuse_views(self, features, mask):
        bs, n_views, _ = features.size()
        cls_tokens = self.cls_token.expand(bs, -1, -1)  # [B, 1, dim]
        features = torch.cat([cls_tokens, features], dim=1)  # [B, V+1, dim]
        features += self.positional_embedding[:, :n_views + 1].expand(bs, -1, -1)  # [B, V+1, dim]
        cls_mask = torch.zeros(bs, 1).bool().to(features.device)  # [B, 1], 0 for seen
        mask = torch.cat([cls_mask, mask], dim=1)  # [B, V+1]
        pooled_output = self.transformer_encoder(features, src_key_padding_mask=mask)
        return pooled_output[:, 0, :]  # [B, dim]

    def training_step(self, batch, batch_idx):
        # images: float of [B, V, C, H, W], masks: bool of [B, V], labels: int of [B, C]
        images, masks, labels = batch['images'], batch['masks'], batch['labels']
        logits = self(images, masks)
        loss = self.criterion(logits, labels.float())
        probs = self.sigmoid(logits)
        preds = torch.round(probs)
        f1 = self.f1(preds, labels)
        ap = self.map(probs, labels)
        gradnorm_vit = utils.get_grad_norm(self.vit)
        gradnorm_fuse = utils.get_grad_norm(self.transformer_encoder)
        gradnorm_head = utils.get_grad_norm(self.classifier_head)
        self.log('train/gradnorm_vit', gradnorm_vit, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/gradnorm_fuse', gradnorm_fuse, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/gradnorm_head', gradnorm_head, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/map', ap, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, labels, names = batch['images'], batch['masks'], batch['labels'], batch['study_ids']
        logits = self(images, masks)
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

        if self.class_id is not None:
            f1_single = self.f1_single(preds[:, self.class_id], labels[:, self.class_id])
            ap_single = self.map_single(probs[:, self.class_id], labels[:, self.class_id])
            self.log('val/f1_single', f1_single, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/map_single', ap_single, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

    def sinusoidal_embedding(self, num_positions, dim):
        """Create sinusoidal positional embeddings."""
        position = torch.arange(0, num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        embeddings = torch.zeros(num_positions, dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings.unsqueeze(0)


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
