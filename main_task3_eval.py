# Lightnign model, image classification
# 2024-07-02 by xtc

import os
import pdb
import random
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import pytorch_lightning as pl

import dataset_vlm as dataset
import utils.ops as ops
import utils.utils as utils
from arch.vlm import VLM


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MIMIC/task3_development_starting_kit')
    parser.add_argument('--csv_file', type=str, default='development.csv')
    parser.add_argument('--img_dir', type=str, default='data/mimic-cxr-jpg/2.0.0')
    parser.add_argument('--desc_file', type=str, default=None)  # Or 'description.csv'
    parser.add_argument('--names_file', type=str, default='CLASSES.txt')  # Or 'CLASSES_45.txt'
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--model_url_vision', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--model_url_lang', type=str, default='microsoft/BiomedVLP-CXR-BERT-specialized')
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRA.')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--class_name', type=str, default=None)
    parser.add_argument('--loss', type=str, default='wbce')
    parser.add_argument('--lora_rank', '-rank', type=int, default=4)

    # Usually default hyperparams
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--name', type=str, default='task3-baseline')
    parser.add_argument('--mode', type=str, default='online')
    return parser.parse_args()


def set_up(args):
    PROJECT_NAME = "Class-LT-EVAL"
    SAVE_CODE_LIST = ["arch/vlm.py", "dataset_vlm.py", "utils/ops.py", "utils/utils.py", "main_task3_eval.py",
                      'job/eval_task3.job']
    TIME = utils.get_time()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID', None)

    if args.slurm_job_id is not None:
        args.name = f'{args.slurm_job_id}-{args.name}'
    if args.class_name is not None:
        args.name = f'{args.name}-{args.class_name}'
    if args.debug:
        args.name = args.name + '-try'

    run_id = random.randint(10000, 99999)
    model_dir = os.path.join(args.save_dir, f'{TIME}-{PROJECT_NAME}-{args.name}-{run_id}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    utils.save_code(model_dir, SAVE_CODE_LIST)
    print(f"\nSave to {model_dir}")

    return model_dir


def main(args):
    ## Setup
    model_dir = set_up(args)
    if args.class_name is not None:
        class_names_pred = [args.class_name]
    else:
        class_names_pred = ops.read_txt(args.data_dir, "EVAL_CLASSES.txt")
    pl.seed_everything(args.seed, workers=True)
    sigmoid = torch.nn.Sigmoid()

    # Load model
    model = VLM.load_from_checkpoint(model_url_vision=args.model_url_vision, model_url_language=args.model_url_lang,
                                     n_classes=1,
                                     checkpoint_path=args.ckpt_path, criterion=ops.get_criterion(args.loss),
                                     with_lora=args.lora, lora_rank=args.lora_rank)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt_path}")

    df = pd.DataFrame()

    for class_name in class_names_pred:
        print(f"\nInferencing class: {class_name}")

        # Load dataset
        data_loader = dataset.InferenceDataModule(data_dir=args.data_dir, csv_file=args.csv_file, img_dir=args.img_dir,
                                                  tokenizer_url=args.model_url_lang,
                                                  class_name=class_name, batch_size=args.batch_size,
                                                  img_size=args.img_size, desc_file=args.desc_file)
        data_loader.setup()
        test_loader = data_loader.test_dataloader()

        all_names = []
        all_probs_i = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                images, input_ids, attention_masks, names = batch['images'], batch['input_ids'], batch[
                    'attention_masks'], batch['names']
                images = images.to(model.device)
                input_ids = input_ids.to(model.device)
                attention_masks = attention_masks.to(model.device)
                logits = model(images, input_ids, attention_masks).squeeze()
                probs = sigmoid(logits)
                all_names.extend(names)
                all_probs_i.extend(probs.detach().cpu().numpy())

        df['dicom_id'] = all_names
        df[class_name] = all_probs_i

    # Save to a csv file
    df = df[['dicom_id'] + class_names_pred]
    df.to_csv(os.path.join(model_dir, 'probs.csv'), index=False)


if __name__ == "__main__":
    args = parseargs()
    main(args)
