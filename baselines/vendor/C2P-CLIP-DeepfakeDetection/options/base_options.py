import argparse
import os
import time

import torch

import utils.util as util


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "yes", "on", "y", "t", "1"}:
        return True
    if value in {"false", "no", "off", "n", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from {value}.")


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_clip_dir = os.path.join(project_root, "pretrained", "clip-vit-large-patch14")
        clip_default = local_clip_dir if os.path.isdir(local_clip_dir) else "openai/clip-vit-large-patch14"

        parser.add_argument("--mode", default="binary")
        parser.add_argument("--arch", type=str, default="c2p_clip", help="architecture for binary classification")

        parser.add_argument("--rz_interp", default="bilinear")
        parser.add_argument("--blur_prob", type=float, default=0.0)
        parser.add_argument("--blur_sig", default="0.5")
        parser.add_argument("--jpg_prob", type=float, default=0.0)
        parser.add_argument("--jpg_method", default="pil")
        parser.add_argument("--jpg_qual", default="75")

        parser.add_argument("--dataroot", default="./dataset/", help="path to image folders")
        parser.add_argument("--textroot", default="", help="optional path to caption txt files")
        parser.add_argument("--classes", default="", help="classes to use, separated by comma")
        parser.add_argument("--class_bal", action="store_true")
        parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
        parser.add_argument("--loadSize", type=int, default=256, help="resize size")
        parser.add_argument("--cropSize", type=int, default=224, help="crop size")
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids, e.g. 0 or 0,1,2")
        parser.add_argument("--name", type=str, default="c2p_clip", help="experiment name")
        parser.add_argument("--epoch", type=str, default="latest", help="which checkpoint epoch to load")
        parser.add_argument("--num_threads", type=int, default=8, help="number of dataloader workers")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="checkpoint root")
        parser.add_argument("--serial_batches", action="store_true", help="disable random shuffle")
        parser.add_argument("--no_flip", action="store_true", help="disable horizontal flip")
        parser.add_argument("--suffix", type=str, default="", help="customized suffix appended to opt.name")

        parser.add_argument("--delr_freq", type=int, default=10, help="epoch interval for lr decay")
        parser.add_argument("--delr", type=float, default=0.8, help="lr decay factor")
        parser.add_argument("--seed", type=int, default=123, help="random seed")
        parser.add_argument("--clip", type=str, default=clip_default, help="HF model id or local path")
        parser.add_argument("--claloss", type=float, default=8.0, help="classification loss weight")
        parser.add_argument("--cates", nargs="+", default=["Deepfake", "Camera"], help="common prompt words")
        parser.add_argument("--lora_r", type=int, default=6, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=6, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.5, help="LoRA dropout")
        parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
        parser.add_argument(
            "--lmdb_real_label",
            type=int,
            default=1,
            help="raw label value corresponding to real images when LMDB metadata lacks explicit hints",
        )

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: {default}]"
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def _format_train_name(self, opt):
        time_tag = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        return "__".join(
            [
                opt.name,
                time_tag,
                f"Seed_{opt.seed}",
                "cates_" + "-".join(opt.cates),
                f"claloss_{opt.claloss}",
                f"lora_r_{opt.lora_r}",
                f"lora_alpha_{opt.lora_alpha}",
                f"lora_dropout_{opt.lora_dropout}",
                f"lr_{opt.lr}",
            ]
        )

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        opt.imgroot = opt.dataroot

        if opt.isTrain:
            opt.name = self._format_train_name(opt)

        if opt.suffix:
            suffix = "_" + opt.suffix.format(**vars(opt))
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        device_count = torch.cuda.device_count()
        cuda_available = torch.cuda.is_available() and device_count > 0
        opt.gpu_ids = [int(idx) for idx in opt.gpu_ids.split(",") if int(idx) >= 0]
        if cuda_available:
            opt.gpu_ids = [idx for idx in opt.gpu_ids if idx < device_count] or [0]
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                local_rank = int(local_rank)
                if not 0 <= local_rank < device_count:
                    raise ValueError(f"LOCAL_RANK {local_rank} is out of range for {device_count} devices.")
                torch.cuda.set_device(local_rank)
            else:
                torch.cuda.set_device(opt.gpu_ids[0])

        opt.classes = [cls for cls in opt.classes.split(",") if cls]
        opt.rz_interp = opt.rz_interp.split(",")
        opt.blur_sig = [float(item) for item in opt.blur_sig.split(",")]
        opt.jpg_method = opt.jpg_method.split(",")
        opt.jpg_qual = [int(item) for item in opt.jpg_qual.split(",")]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("jpg_qual should contain at most two comma separated values.")

        self.opt = opt
        return self.opt
