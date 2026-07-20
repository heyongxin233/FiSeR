import os

import torch
import torch.nn as nn
from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.lr = getattr(opt, "lr", 0.0)
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = getattr(opt, "device", None)
        if self.device is None:
            self.device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")

    def save_networks(self, epoch):
        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        state = {
            "model": getattr(self.model, "module", self.model).state_dict(),
            "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "total_steps": self.total_steps,
        }
        torch.save(state, save_path)
        print(f"Saving model {save_path}")

    def load_networks(self, epoch):
        load_filename = f"model_epoch_{epoch}.pth"
        load_path = os.path.join(self.save_dir, load_filename)
        print(f"loading the model from {load_path}")
        state = torch.load(load_path, map_location=self.device)
        if hasattr(state, "_metadata"):
            del state._metadata

        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(model_state, strict=True)

        if isinstance(state, dict):
            self.total_steps = state.get("total_steps", self.total_steps)
            if self.isTrain and not self.opt.new_optim and state.get("optimizer") is not None:
                self.optimizer.load_state_dict(state["optimizer"])
                for optim_state in self.optimizer.state.values():
                    for key, value in optim_state.items():
                        if torch.is_tensor(value):
                            optim_state[key] = value.to(self.device)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.opt.lr

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(module.weight.data, 1.0, gain)
            init.constant_(module.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)
