# Train use multiple datasets: LT, hash2, facts, label93
# Lightnign model, image classification
# 2024-07-08 by xtc

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import dataset_vlm as dataset
import utils.ops as ops
import utils.utils as utils
from arch.vlm import VLM


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MIMIC/task3_development_starting_kit')
    parser.add_argument('--csv_file', type=str, default='train_labeled.csv')  # or 'train_labeled_45.csv'
    parser.add_argument('--img_dir', type=str, default='data/mimic-cxr-jpg/2.0.0')
    parser.add_argument('--names_file', type=str, default='CLASSES.txt')  # or 'CLASSES_45.txt'
    parser.add_argument('--batch_size', '-bs', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model_url_vision', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--model_url_lang', type=str, default='microsoft/BiomedVLP-CXR-BERT-specialized')
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--loss', type=str, default='wbce')
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRA.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=504)
    parser.add_argument('--accum_grad_batches', '-acc', type=int, default=4)
    parser.add_argument('--lora_rank', '-rank', type=int, default=4)

    # Usually default hyperparams
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--name', type=str, default='task3-baseline-pretrain')
    parser.add_argument('--mode', type=str, default='online')
    return parser.parse_args()


def set_up(args):
    PROJECT_NAME = "Class-LT"
    SAVE_CODE_LIST = ["arch/vlm.py", "dataset_vlm.py", "utils/ops.py", "utils/utils.py", "main_task3_pretrain.py",
                      'job/train_task3.job']
    TIME = utils.get_time()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID', None)

    if args.slurm_job_id is not None:
        args.name = f'{args.slurm_job_id}-' + args.name
    if args.debug:
        args.name = args.name + '-try'

    logger = WandbLogger(project=PROJECT_NAME, name=TIME + '-' + args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{TIME}-{PROJECT_NAME}-{args.name}-{logger.experiment.id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    utils.save_code(model_dir, SAVE_CODE_LIST)
    print(f"Save to {model_dir}")

    return model_dir, logger


def main(args):
    # Setup
    model_dir, logger = set_up(args)
    class_names = ops.read_txt(args.data_dir, args.names_file)
    pl.seed_everything(args.seed, workers=True)

    # Load dataset
    data_loader = dataset.UnifiedDataModule(data_dir=args.data_dir, csv_file=args.csv_file, img_dir=args.img_dir,
                                            tokenizer_url=args.model_url_lang,
                                            class_names=class_names, batch_size=args.batch_size, seed=args.seed,
                                            img_size=args.img_size)
    data_loader.setup()
    train_loader, val_loader = data_loader.train_dataloader(), data_loader.val_dataloader()

    steps_per_epoch = len(train_loader) // (args.accum_grad_batches * torch.cuda.device_count())
    model = VLM(model_url_vision=args.model_url_vision, model_url_language=args.model_url_lang, n_classes=1,
                epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, criterion=ops.get_criterion(args.loss),
                steps_per_epoch=steps_per_epoch, with_lora=args.lora, lora_rank=args.lora_rank, save_dir=model_dir)

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
                         accelerator="auto",
                         strategy=DDPStrategy(find_unused_parameters=True),
                         precision='16-mixed',
                         check_val_every_n_epoch=1,
                         accumulate_grad_batches=args.accum_grad_batches)

    if trainer.global_rank == 0:
        logger.experiment.config.update(vars(args))

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = parseargs()
    main(args)
