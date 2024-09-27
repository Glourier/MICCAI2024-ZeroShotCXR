# Lightnign model, image classification
# 2024-07-02 by xtc

import os
import pdb
import argparse
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import dataset
import utils.ops as ops
import utils.utils as utils
from arch.vit import ViT


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MIMIC/task1_development_starting_kit')
    parser.add_argument('--csv_file', type=str, default='train_labeled.csv')
    parser.add_argument('--img_dir', type=str, default='data/mimic-cxr-jpg/2.0.0')
    parser.add_argument('--names_file', type=str, default='CLASSES.txt')  # Or 'CLASSES_45.txt'
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model_url', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--loss', type=str, default='wbce')
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRA.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--accum_grad_batches', '-acc', type=int, default=4)
    parser.add_argument('--class_name', type=str, default=None, help='If specified, do single-class classification.')
    parser.add_argument('--n_val', type=float, default=1.0,
                        help='If 0 < n_val <= 1, val every v_val epochs; if > 1 and is int, val every n_val batches.')
    parser.add_argument('--lora_rank', '-rank', type=int, default=4)
    parser.add_argument('--lr_T', type=float, default=1.0, help='Number of circles of lr_scheduler.')
    parser.add_argument('--backbone', type=str, default='dino',
                        help='Which backbone to use. Choose from [vit, dino, rad-dino]')
    parser.add_argument('--freeze_encoder', type=str, default=None, help='Which parts to freeze.')
    parser.add_argument('--fold', type=int, default=None, help='If specified, use the fold for training.')
    parser.add_argument('--dropout', type=float, default=0)

    # Usually default hyperparams
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--name', type=str, default='task1-baseline')
    parser.add_argument('--mode', type=str, default='online')
    return parser.parse_args()


def set_up(args):
    PROJECT_NAME = "Class-LT"
    SAVE_CODE_LIST = ["arch/vit.py", "dataset.py", "utils/ops.py", "utils/utils.py", "main_task1.py",
                      'job/train_task1.job']
    TIME = utils.get_time()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID', None)

    if args.slurm_job_id is not None:
        args.name = f'{args.slurm_job_id}-' + args.name
    if args.class_name is not None:
        args.name = f'{args.name}-{args.class_name}'
    if args.fold is not None:
        args.name = f'{args.name}-fold{args.fold}'
    if args.debug:
        args.name = args.name + '-try'

    logger = WandbLogger(project=PROJECT_NAME, name=TIME + '-' + args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{TIME}-{PROJECT_NAME}-{args.name}-{logger.experiment.id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    utils.save_code(model_dir, SAVE_CODE_LIST)
    print(f"\nSave to {model_dir}")

    return model_dir, logger


def freeze_params(model, freeze_encoder):
    if freeze_encoder == 'image':
        for name, param in model.named_parameters():
            if 'vit' in name:
                param.requires_grad = False
        print("Froze image encoder.")
    else:
        print("No encoder is frozen.")
    return model


def main(args):
    ## Setup
    model_dir, logger = set_up(args)
    if args.class_name is not None:
        class_names = [args.class_name]
    else:
        class_names = ops.read_txt(args.data_dir, args.names_file)
    n_classes = len(class_names)
    pl.seed_everything(args.seed, workers=True)

    # Load dataset
    data_loader = dataset.ImageDataModule(data_dir=args.data_dir, csv_file=args.csv_file, img_dir=args.img_dir,
                                          class_names=class_names, batch_size=args.batch_size, seed=args.seed,
                                          img_size=args.img_size)
    data_loader.setup(fold_idx=args.fold)
    train_loader, val_loader = data_loader.train_dataloader(), data_loader.val_dataloader()

    steps_per_epoch = len(train_loader) // (args.accum_grad_batches * torch.cuda.device_count() * args.lr_T)
    model = ViT(model_url=args.model_url, n_classes=n_classes, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, criterion=ops.get_criterion(args.loss, n_classes),
                steps_per_epoch=steps_per_epoch, dropout=args.dropout,
                with_lora=args.lora, lora_rank=args.lora_rank, save_dir=model_dir, backbone=args.backbone)

    model = freeze_params(model, args.freeze_encoder)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_callback = ModelCheckpoint(monitor='val/map_epoch', mode='max', save_top_k=3, save_last=True, verbose=True,
                                    dirpath=os.path.join(model_dir, 'ckpt'),
                                    filename='best-{epoch}')

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(max_epochs=args.epochs,
                         enable_progress_bar=True,
                         enable_model_summary=True,
                         logger=logger,
                         callbacks=[lr_monitor, ckpt_callback],
                         accelerator="auto",  # gpu
                         # strategy="ddp",  # DDP is default  # TODO: set seeds for DDP
                         # distributed_backend='ddp',
                         strategy=DDPStrategy(find_unused_parameters=True),
                         precision='16-mixed',
                         # check_val_every_n_epoch=1,  # Currently not support float
                         val_check_interval=args.n_val,  # val every 0.5 epoch
                         accumulate_grad_batches=args.accum_grad_batches)

    if trainer.global_rank == 0:
        logger.experiment.config.update(vars(args))

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = parseargs()
    main(args)
