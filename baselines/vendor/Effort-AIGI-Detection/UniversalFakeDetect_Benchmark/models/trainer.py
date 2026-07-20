from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_model import BaseModel, init_weights
from models import get_model


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.should_log = (not opt.distributed) or opt.local_rank == 0
        self.precision = opt.precision
        self.autocast_enabled = False
        self.autocast_dtype = None
        self.autocast_device_type = self.device.type
        backbone = get_model(opt.arch, opt)
        self.lr = opt.lr
        torch.nn.init.normal_(backbone.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            params = []
            for name, p in backbone.named_parameters():
                if name == "fc.weight" or name == "fc.bias":
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            if self.should_log:
                print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            params = [p for p in backbone.parameters() if p.requires_grad]

        if not params:
            raise ValueError("No trainable parameters were found for the current configuration.")

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        if self.precision == "bf16-mixed":
            if self.device.type != "cuda":
                if self.should_log:
                    print("bf16 mixed precision requested on a non-CUDA device, falling back to fp32.")
                self.precision = "fp32"
            elif hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                if self.should_log:
                    print("bf16 mixed precision is not supported on this GPU, falling back to fp32.")
                self.precision = "fp32"
            else:
                self.autocast_enabled = True
                self.autocast_dtype = torch.bfloat16

        if self.should_log:
            print(f"Training precision: {self.precision}")

        self.model = backbone.to(self.device)
        if opt.compile:
            if hasattr(torch, "compile"):
                if self.should_log:
                    print(
                        "Compiling model with torch.compile "
                        f"(backend={opt.compile_backend}, mode={opt.compile_mode})"
                    )
                self.model = torch.compile(
                    self.model,
                    backend=opt.compile_backend,
                    mode=opt.compile_mode,
                )
            elif self.should_log:
                print("torch.compile is not available in the current PyTorch build, skipping compile.")
        if opt.distributed:
            self.model = DDP(
                self.model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    def autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(
            device_type=self.autocast_device_type,
            dtype=self.autocast_dtype,
        )

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            self.lr = param_group['lr']
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
        self.optimizer.zero_grad()
        with self.autocast_context():
            self.forward()
            self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss.backward()
        self.optimizer.step()
