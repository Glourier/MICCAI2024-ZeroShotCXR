# Vision language model, to process images and texts.
# Has support with LoRA, to improve efficiency while preserving performance.
# 2024-07-04 by xtc

import os
import torch
import torch.nn as nn
from transformers import ViTModel
from pytorch_lightning import LightningModule
from torchmetrics.classification import F1Score, AveragePrecision
from transformers import BertModel
import peft
from peft import get_peft_model, LoraConfig
import pandas as pd
from itertools import chain
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


class VLM(LightningModule):
    def __init__(self, model_url_vision, model_url_language, n_classes=1, hidden_dim=512, epochs=100,
                 steps_per_epoch=None, lr=1e-5, weight_decay=1e-5,
                 criterion=None, with_lora=False, lora_rank=4, save_dir=None):
        super().__init__()
        # # Vision choice 1: ViT
        # self.vit = ViTModel.from_pretrained(model_url_vision)
        # if with_lora:
        #     self.vit = apply_lora_to_model(self.vit, model_name='vit', rank=lora_rank)
        # self.vit_mapping = nn.Linear(in_features=self.vit.config.hidden_size, out_features=hidden_dim, bias=True)

        # Vision choice 2: Dino-v2
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.vit_mapping = nn.Linear(in_features=self.vit.embed_dim, out_features=hidden_dim, bias=True)
        self.bert = BertModel.from_pretrained(model_url_language)
        self.bert_mapping = nn.Linear(in_features=self.bert.config.hidden_size, out_features=hidden_dim, bias=True)
        self.connector = nn.Sequential(
            nn.Linear(in_features=hidden_dim + hidden_dim, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes, bias=True)
        )  # Output yes or no
        if with_lora:
            self.vit = apply_lora_to_model(self.vit, model_name='dino', rank=lora_rank)
            self.bert = apply_lora_to_model(self.bert, model_name='bert', rank=lora_rank)


        # Hyperparameters
        # self.vit.config.image_size = 1024
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.criterion = criterion  # TODO: make criterion
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim  # Same for vision and language features
        self.save_dir = save_dir

        # Functions
        self.sigmoid = nn.Sigmoid()
        self.map = AveragePrecision(task='binary')
        self.f1 = F1Score(zero_division=0, task='binary')
        self.val_outputs = []

    def forward(self, images, input_ids, attention_mask):
        # # Version 1: ViT
        # img_features = self.vit(images).last_hidden_state[:, 0, :]  # Or use the pooled_output: outputs[1]
        # img_features = self.vit_mapping(img_features)

        # Version 2: Dino-v2
        img_features = self.vit(images)
        img_features = self.vit_mapping(img_features)

        lang_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0,
                        :]  # Or use the pooled_output: outputs[1]
        lang_features = self.bert_mapping(lang_features)

        features = torch.cat([img_features, lang_features], dim=1)
        logits = self.connector(features)


        return logits

    def training_step(self, batch, batch_idx):
        images, input_ids, attention_masks, labels = batch['images'], batch['input_ids'], batch[
            'attention_masks'], batch['labels']
        logits = self(images, input_ids, attention_masks).squeeze()  # [B]
        loss = self.criterion(logits, labels.float())
        probs = self.sigmoid(logits)
        preds = torch.round(probs)
        f1 = self.f1(preds, labels)
        ap = self.map(probs, labels)
        gradnorm_vit = utils.get_grad_norm(self.vit)
        gradnorm_bert = utils.get_grad_norm(self.bert)
        gradnorm_head = utils.get_grad_norm(self.connector)
        self.log('train/gradnorm_vit', gradnorm_vit, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/gradnorm_bert', gradnorm_bert, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/gradnorm_head', gradnorm_head, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/map', ap, on_step=True, on_epoch=True, prog_bar=True)

        # print(f"labels: {labels}")
        # print(f"preds: {preds}")
        # print(f"loss: {loss}, f1: {f1}, map: {ap}")

        return loss

    def validation_step(self, batch, batch_idx):
        images, input_ids, attention_masks, labels = batch['images'], batch['input_ids'], batch[
            'attention_masks'], batch['labels']
        texts, dicom_ids = batch['texts'], batch['dicom_ids']
        logits = self(images, input_ids, attention_masks).squeeze()
        loss = self.criterion(logits, labels.float())
        outputs = {'loss': loss, 'labels': labels, 'logits': logits, 'texts': texts, 'dicom_ids': dicom_ids}
        self.val_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        loss = torch.stack([out['loss'] for out in self.val_outputs]).mean()
        logits = torch.cat([out['logits'] for out in self.val_outputs], dim=0)
        labels = torch.cat([out['labels'] for out in self.val_outputs], dim=0)
        probs = self.sigmoid(logits)
        preds = torch.round(probs)
        f1 = self.f1(preds, labels)
        ap = self.map(probs, labels)

        dicom_ids = [dicom_id for out in self.val_outputs for dicom_id in out['dicom_ids']]
        texts = [text for out in self.val_outputs for text in out['texts']]
        save_probs(dicom_ids, texts, probs, labels, os.path.join(self.save_dir, 'val_last.csv'))
        self.log('val/loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/f1_epoch', f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/map_epoch', ap, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_outputs = []

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     total_steps = self.epochs * self.steps_per_epoch
    #     warmup_steps = 0.1 * total_steps
    #
    #     def warmup_lambda(step):
    #         if step < warmup_steps:
    #             return float(step) / float(max(1, warmup_steps))
    #         return 1.0
    #
    #
    #     print(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}\n epoch: {self.epochs}, steps_per_epoch: {self.steps_per_epoch}")
    #     warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    #     cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    #     combined_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, cosine_scheduler])
    #     scheduler = {
    #         'scheduler': combined_scheduler,
    #         'interval': 'step',
    #         'frequency': 1,
    #         'name': 'learning_rate'
    #     }
    #     return [optimizer], [scheduler]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.epochs * self.steps_per_epoch
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps),
            'interval': 'step',
            'frequency': 1,
            'name': 'learning_rate'
        }
        # TODO: add get_cosine_schedule_with_warmup?

        return [optimizer], [scheduler]

    # def check_trainable_params(self):
    #     for name, param in self.named_parameters():
    #         print(f"{name} is {'frozen' if not param.requires_grad else 'unfrozen'}.")


def save_probs(dicom_ids, texts, probs, labels, file_name):
    probs = probs.cpu().numpy().astype('float16')
    labels = labels.cpu().numpy().astype('int8')
    data = {'dicom_ids': dicom_ids, 'texts': texts, 'probs': probs, 'labels': labels}
    pd.DataFrame(data, columns=['dicom_ids', 'texts', 'probs', 'labels']).to_csv(file_name, header=True, index=False)
