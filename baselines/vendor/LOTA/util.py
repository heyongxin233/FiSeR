import math
import random

import numpy as np
import torch


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        cur_lr = lr
    return cur_lr


def cosine_warmup_lr(optimizer, base_lr, min_lr, step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        lr = base_lr * float(step + 1) / float(warmup_steps)
    elif total_steps <= warmup_steps:
        lr = base_lr
    else:
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps - 1, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

from torch import nn

def bceLoss():
    return nn.BCEWithLogitsLoss()


def crossEntropyLoss():
    return nn.CrossEntropyLoss()
def mseLoss():
    return nn.MSELoss()
