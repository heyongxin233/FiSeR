import functools
import math
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt.arch)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.lr = opt.lr
        self.total_train_steps = getattr(opt, 'total_train_steps', 0)
        self.model.to(self.device)


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        if self.total_train_steps:
            warmup_steps = min(self.opt.warmup_steps, self.opt.steps_per_epoch)
            warmup_steps = min(warmup_steps, self.total_train_steps)
            if warmup_steps > 0 and self.total_steps < warmup_steps:
                current_lr = self.opt.lr * float(self.total_steps + 1) / float(max(1, warmup_steps))
            else:
                progress = 0.0
                if self.total_train_steps > warmup_steps:
                    progress = (self.total_steps - warmup_steps) / float(self.total_train_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                current_lr = self.opt.lr * (self.opt.min_lr_ratio + (1 - self.opt.min_lr_ratio) * cosine_decay)
            for group in self.optimizer.param_groups:
                group['lr'] = current_lr
            self.lr = current_lr

        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.total_steps += 1


