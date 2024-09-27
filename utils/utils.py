import os
import json
import datetime
import shutil
import numpy as np
import torch.nn as nn


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        return curr_lr


def get_grad_norm(model):
    """
    Calculate the norm of the gradient of the model.
    """
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                continue
            else:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_grad_norm_parameter(param):
    """
    Calculate the norm of the gradient of a parameter.
    """
    if param.requires_grad:
        if param.grad is None:
            return -1
        else:
            return param.grad.detach().data.norm(2).item()
    return -1  # -1 means no gradient


def save_params(model_dir, params, name='params'):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)


def get_time():
    now = datetime.datetime.now()
    formatted_time = now.strftime('%Y%m%d_%H%M')
    return formatted_time


def save_code(model_dir, code_names):
    save_dir = os.path.join(model_dir, 'backup')
    os.makedirs(save_dir, exist_ok=True)
    for name in code_names:
        shutil.copy(name, save_dir)
    return


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def count_parameters(model):
    params_all = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_all, params_trainable
