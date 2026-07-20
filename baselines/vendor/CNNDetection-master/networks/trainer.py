import functools
import math
import torch
import torch.nn as nn
from networks.resnet import resnet50
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
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

            self.total_train_steps = opt.total_train_steps
            self.lr_schedule = _cosine_warmup
            if opt.warmup_steps > 0:
                for group in self.optimizer.param_groups:
                    group['lr'] = 0.0
                self.lr = 0.0
            else:
                self.lr = opt.lr

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
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

    def step_scheduler(self):
        return

    def get_current_lr(self):
        return self.lr if hasattr(self, 'lr') else self.optimizer.param_groups[0]['lr']
