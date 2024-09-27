# Calculate weights for ensembling models
# Modified from 'MedVQA.medvqa.models.ensemble.py'
# 2024-07-23 by xtc

import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import utils.utils as utils


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MIMIC/task1_development_starting_kit')
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--names_file', type=str, default='CLASSES.txt')  # Or 'CLASSES_45.txt'
    parser.add_argument('--val_labels_file', '-labels', type=str, default='val_df.csv')
    parser.add_argument('--val_probs_files', '-val', type=str, default=None,
                        help='Contains paths to probs files for val set')
    parser.add_argument('--test_probs_files', '-test', type=str, default=None,
                        help='Contains paths to probs files for dev/test set')
    parser.add_argument('--stage', type=str, default='val',
                        help='Which stage to ensemble, choose from [val, pred, both]')
    parser.add_argument('--fuse', type=str, default='max', help='Ensemble method, choose from [max, mean, weighted]')
    parser.add_argument('--weight_file', type=str, default='weights.csv', help='weights file for pred (test) stage')
    return parser.parse_args()


# Load class_names
def read_txt(data_dir, file_name):
    names = pd.read_csv(os.path.join(data_dir, file_name), header=None).values
    names = [name[0] for name in names]
    return names


def load_probs(file_dirs, file_suffix='probs.csv', columns=None):
    all_probs = []
    for file_dir in file_dirs:
        probs = pd.read_csv(os.path.join(file_dir, file_suffix))
        probs = probs[columns].values
        all_probs.append(probs)
    all_probs = np.stack(all_probs)  # [K, N, C], means K models, N samples, C classes
    return all_probs


class AveragePrecision:
    def __init__(self, num_labels=None, average='None'):
        self.num_labels = num_labels
        self.average = average

    def __call__(self, probs, labels):
        if self.num_labels == 1:
            return average_precision_score(labels, probs)
        elif self.num_labels > 1:
            scores = np.array([average_precision_score(labels[:, i], probs[:, i]) for i in range(self.num_labels)])
            if self.average == 'None':
                return scores
            elif self.average == 'macro':
                return np.mean(scores)
            else:
                raise ValueError(f"Invalid average method: {self.average}, please choose from ['None', 'macro']. ")
        else:
            raise ValueError(f"Invalid number of classes: {self.num_labels}")


def get_weights(APs, method='max'):
    # Keep the shape, just assign weights for each model each class
    # APs: [K, C], means K models, C classes
    if method == 'max':
        weights = np.zeros_like(APs)
        for i in range(APs.shape[1]):
            max_idx = np.argmax(APs[:, i])
            weights[max_idx, i] = 1
    elif method == 'mean':
        weights = np.ones_like(APs) / APs.shape[0]
    elif method == 'weighted':
        weights = np.zeros_like(APs)
        for i in range(APs.shape[1]):
            weights[:, i] = APs[:, i] / np.sum(APs[:, i])
    else:
        raise ValueError(f"Invalid weight method: {method}, please choose from ['max', 'mean', 'weighted']. ")
    return weights


def weighted_average(probs, weights):
    # probs: [K, N, C], K models, N samples, C classes
    # weights: [K, C]
    n_models, n_samples, n_classes = probs.shape
    assert weights.shape == (n_models, n_classes)
    probs_weighted = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        for k in range(n_models):
            probs_weighted[:, c] += probs[k, :, c] * weights[k, c]
        probs_weighted[:, c] /= np.sum(weights[:, c])
    return probs_weighted


def set_up(args):
    PROJECT_NAME = "Class-LT-ENSEMBLE"
    SAVE_CODE_LIST = ["utils/utils.py", "ensemble.py", "job/ensemble.job"]
    TIME = utils.get_time()
    args.slurm_job_id = os.getenv('SLURM_JOB_ID', None)

    args.name = f'{args.stage}-{args.fuse}'
    if args.slurm_job_id is not None:
        args.name = f'{args.slurm_job_id}-' + args.name

    run_id = random.randint(10000, 99999)
    model_dir = os.path.join(args.save_dir, f'{TIME}-{PROJECT_NAME}-{args.name}-{run_id}')
    os.makedirs(model_dir, exist_ok=True)
    utils.save_params(model_dir, vars(args))
    utils.save_code(model_dir, SAVE_CODE_LIST)

    if args.val_probs_files is not None:
        utils.save_code(model_dir, [os.path.join(args.data_dir, args.val_probs_files)])
    if args.test_probs_files is not None:
        utils.save_code(model_dir, [os.path.join(args.data_dir, args.test_probs_files)])

    print(f"Save to {model_dir}")
    return model_dir


def main(args):
    # Setup
    result_dir = set_up(args)
    class_names = read_txt(args.data_dir, "EVAL_CLASSES.txt")
    n_classes = len(class_names)
    AP_func = AveragePrecision(n_classes, average='None')

    if args.stage in ['val', 'both']:
        print("\nCalculating weights for each model and each class...")
        val_files = read_txt(args.data_dir, args.val_probs_files)
        all_probs = load_probs(val_files, 'probs.csv', columns=class_names)
        labels = pd.read_csv(os.path.join(args.data_dir, args.val_labels_file))
        labels = labels[class_names].values

        # Calc AP for each model each class
        APs = []
        for k in range(all_probs.shape[0]):
            AP_k = AP_func(all_probs[k], labels)
            print(f'AP of Model {k}: {AP_k}\n  Average: {np.mean(AP_k)}')
            APs.append(AP_k)
        APs = np.array(APs)  # [K, C], K models, C classes
        weights = get_weights(APs, method=args.fuse)
        probs_fused = weighted_average(all_probs, weights)
        AP_fused = AP_func(probs_fused, labels)
        print(f"\nAP of fused model: {AP_fused}\n  Average: {np.mean(AP_fused)}. ")

        # Save weights
        df_save = pd.DataFrame(weights, columns=class_names)
        df_save.to_csv(os.path.join(result_dir, 'weights.csv'), index=False)
        print(f"Saved weights to {os.path.join(result_dir, 'weights.csv')}. ")

    if args.stage in ['pred', 'both']:
        print("\nCreating weighted average probs...")
        if args.stage == 'pred':
            weights = pd.read_csv(args.weight_file)
        else:
            weights = pd.read_csv(os.path.join(result_dir, 'weights.csv'))
        weights = weights[class_names].values
        test_files = read_txt(args.data_dir, args.test_probs_files)
        all_probs = load_probs(test_files, 'probs.csv', columns=class_names)
        probs_fuse = weighted_average(all_probs, weights)

        # Save fused probs
        df_template = pd.read_csv(os.path.join(test_files[0], 'probs.csv'))
        df_save = pd.DataFrame(probs_fuse, columns=class_names)
        df_save['dicom_id'] = df_template['dicom_id']
        df_save = df_save[['dicom_id'] + class_names]
        df_save.to_csv(os.path.join(result_dir, 'probs.csv'), index=False)
        print(f"Saved fused probs to {os.path.join(result_dir, 'probs.csv')}. ")


if __name__ == '__main__':
    args = parseargs()
    main(args)
