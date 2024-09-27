# Operations
# 2024-07-03 by xtc

import os
import pandas as pd
import torch
import torch.nn as nn


class ASLwithClassWeight(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(ASLwithClassWeight, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1 - p)
        self.neg_weights = torch.exp(p)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()

        # Calculating Probabilities
        xs_pos = torch.sigmoid(pred)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss *= weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label
            pt1 = xs_neg * (1 - label)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


def get_criterion(loss_name, n_classes=1):
    if loss_name == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_name == "wbce":
        if n_classes == 1:  # Binary classification
            weight = torch.tensor([10])  # 1 or 10.
        elif n_classes == 40:
            weight = torch.tensor([74.93751833, 2.95972528, 1299.85929648, 60.06888417,
                                   2.46371324, 1539.89880952, 15.84151974, 5.94843784,
                                   69.71046162, 7.73737681, 220.4465355, 91.35497681,
                                   21.37819848, 86.30893761, 63.9450577, 399.72910217,
                                   355.08115543, 24.66382472, 331.73907455, 2005.75193798,
                                   109.72326775, 2.34104695, 47.95442511, 33.37405391,
                                   6.54902018, 2.89860092, 419.24512987, 78.11705379,
                                   366.71448864, 4.54802829, 500.6879845, 17.68025689,
                                   157.71919068, 285.67884828, 28.02466644, 1504.06395349,
                                   125.52541544, 2.00736533, 76.59922062, 123.57699711])  # |neg|/|pos|
        elif n_classes == 45:
            weight = torch.tensor([74.93751833, 2.95972528, 1299.85929648, 60.06888417,
                                   2.46371324, 1539.89880952, 15.84151974, 5.94843784,
                                   69.71046162, 7.73737681, 220.4465355, 91.35497681,
                                   21.37819848, 86.30893761, 63.9450577, 399.72910217,
                                   355.08115543, 24.66382472, 331.73907455, 2005.75193798,
                                   109.72326775, 2.34104695, 47.95442511, 33.37405391,
                                   6.54902018, 2.89860092, 419.24512987, 78.11705379,
                                   366.71448864, 4.54802829, 500.6879845, 17.68025689,
                                   157.71919068, 285.67884828, 28.02466644, 1504.06395349,
                                   125.52541544, 2.00736533, 76.59922062, 123.57699711,
                                   1280.53960396, 628.85644769, 143.2178273, 595.47695853,
                                   69.6525655])
        else:
            raise ValueError(f"Loss WBCE only support 1, 40 or 45 for now. Unsupported: {n_classes}")
        weight = torch.clamp(weight, max=10)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    elif loss_name == "asl":
        if n_classes == 1:
            num_instances = torch.tensor([23534])
        elif n_classes == 40:
            num_instances = torch.tensor([3409, 65376, 199, 4239, 74738, 168, 15371, 37256, 3661,
                                          29628, 1169, 2803, 11568, 2965, 3986, 646, 727, 10087,
                                          778, 129, 2338, 77482, 5288, 7531, 34292, 66401, 616,
                                          3272, 704, 46660, 516, 13858, 1631, 903, 8919, 172,
                                          2046, 86079, 3336, 2078])  # 40
        elif n_classes == 45:
            num_instances = torch.tensor([3409, 65376, 199, 4239, 74738, 168, 15371, 37256, 3661,
                                          29628, 1169, 2803, 11568, 2965, 3986, 646, 727, 10087,
                                          778, 129, 2338, 77482, 5288, 7531, 34292, 66401, 616,
                                          3272, 704, 46660, 516, 13858, 1631, 903, 8919, 172,
                                          2046, 86079, 3336, 2078, 202, 411, 1795, 434, 3664])  # 45
        else:
            raise ValueError(f"Loss ASL only support 1, 40 or 45 for now. Unsupported: {n_classes}")
        num_total = 258871
        criterion = ASLwithClassWeight(num_instances, num_total)
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
    return criterion

def read_txt(data_dir, file_name):
    names = pd.read_csv(os.path.join(data_dir, file_name), header=None).values
    names = [name[0] for name in names]
    return names  # List