import functools
import math
import torch
import torch.nn as nn
from models.clip_models import CLIPModel
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        def _cosine_warmup(step_idx: int, total_steps: int) -> float:
            warmup_steps = min(opt.warmup_steps, opt.steps_per_epoch)
            warmup_steps = min(warmup_steps, total_steps)
            if warmup_steps > 0 and step_idx < warmup_steps:
                return opt.lr * float(step_idx + 1) / float(max(1, warmup_steps))
            progress = 0.0
            if total_steps > warmup_steps:
                progress = (step_idx - warmup_steps) / float(total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return opt.lr * (opt.min_lr_ratio + (1 - opt.min_lr_ratio) * cosine_decay)

        if self.isTrain and not opt.continue_train:
            self.model = CLIPModel()

        if not self.isTrain or opt.continue_train:
            self.model = CLIPModel()

        params = []
        for name, p in self.model.named_parameters():
            if name == "fc.weight" or name == "fc.bias":
                params.append(p)
            else:
                p.requires_grad = False
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

            self.lr = opt.lr
            self.total_train_steps = opt.total_train_steps
            self.lr_schedule = _cosine_warmup

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(self.device)

    def set_input(self, input):
        self.input = input[0].to(self.device, non_blocking=True)
        self.label = input[1].to(self.device, non_blocking=True).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        if hasattr(self, 'lr_schedule'):
            current_lr = self.lr_schedule(self.total_steps, self.total_train_steps)
            for group in self.optimizer.param_groups:
                group['lr'] = current_lr
            self.lr = current_lr

        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.total_steps += 1
