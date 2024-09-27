# Inference for task1
# Produce a csv file for predictions
# 2024-07-04 by xtc

import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import dataset
import utils.ops as ops
import utils.utils as utils
from arch.vit import ViT


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MIMIC/task1_development_starting_kit')
    parser.add_argument('--csv_file', type=str, default='development.csv')
    parser.add_argument('--img_dir', type=str, default='data/mimic-cxr-jpg/2.0.0')
    parser.add_argument('--names_file', type=str, default='CLASSES_45.txt')
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--model_url', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--ckpt_path', type=str,
                        default='saved/20240703_1854-Class-LT-task1-baseline-1xl15o0x/ckpt/best-epoch=35.ckpt')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRA.')
    parser.add_argument('--loss', type=str, default='wbce')
    parser.add_argument('--class_name', type=str, default=None, help='If specified, do single-class classification.')
    parser.add_argument('--lora_rank', '-rank', type=int, default=4)

    # Usually default hyperparams
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--name', type=str, default='task1-baseline')
    parser.add_argument('--mode', type=str, default='online')
    return parser.parse_args()


def set_up(args):
    PROJECT_NAME = "Class-LT-EVAL"
    SAVE_CODE_LIST = ["arch/vit.py", "dataset.py", "utils/ops.py", "utils/utils.py", "main_task1_eval.py",
                      'job/eval_task1.job']
    TIME = utils.get_time()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID', None)

    if args.slurm_job_id is not None:
        args.name = f'{args.slurm_job_id}-' + args.name
    if args.class_name is not None:
        args.name = args.name + f'-{args.class_name}'
    if args.debug:
        args.name = args.name + '-try'

    run_id = random.randint(10000, 99999)
    model_dir = os.path.join(args.save_dir, f'{TIME}-{PROJECT_NAME}-{args.name}-{run_id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    utils.save_code(model_dir, SAVE_CODE_LIST)
    print(f"Save to {model_dir}")

    return model_dir


def main(args):
    # Setup
    model_dir = set_up(args)
    if args.class_name is not None:
        class_names_train = [args.class_name]
        class_names_pred = [args.class_name]
    else:
        class_names_train = ops.read_txt(args.data_dir, args.names_file)
        class_names_pred = ops.read_txt(args.data_dir, "EVAL_CLASSES.txt")

    n_classes = len(class_names_train)
    pl.seed_everything(args.seed, workers=True)

    # Load dataset
    data_loader = dataset.InferenceDataModule(data_dir=args.data_dir, csv_file=args.csv_file, img_dir=args.img_dir,
                                              batch_size=args.batch_size, img_size=args.img_size)
    data_loader.setup()
    test_loader = data_loader.test_dataloader()

    model = ViT.load_from_checkpoint(model_url=args.model_url, n_classes=n_classes, checkpoint_path=args.ckpt_path,
                                     criterion=ops.get_criterion(args.loss, n_classes), with_lora=args.lora)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt_path}")

    all_names = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, names = batch
            images = images.to(model.device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_names.extend(names)
            all_probs.extend(probs.detach().cpu().numpy())

    # Save to a csv file
    df = pd.DataFrame(all_probs, columns=class_names_train)
    df['dicom_id'] = all_names
    df = df[['dicom_id'] + class_names_pred]  # Pred names may be different from train names
    df.to_csv(os.path.join(model_dir, 'probs.csv'), index=False)


if __name__ == "__main__":
    args = parseargs()
    main(args)
