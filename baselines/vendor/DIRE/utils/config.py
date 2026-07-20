from __future__ import annotations

import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = ["zhaolian_train"]
    datasets_test = ["adm_res_abs_ddim20s"]
    mode = "binary"
    class_bal = False
    batch_size = 64
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 20
    serial_batches = False
    isTrain = True

    # data augmentation
    rz_interp = ["bilinear"]
    blur_prob = 0.0
    blur_sig = [0.5]
    jpg_prob = 0.0
    jpg_method = ["cv2"]
    jpg_qual = [75]
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    ####### train setting ######
    warmup = True
    warmup_epoch = 3
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 400
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True
    train_lmdb = None
    val_lmdb = None
    lmdb_path = None
    eval_names = None
    corruption_type = "none"
    corruption_value = 0.0
    crop_mode = "center"
    metrics_out = ""
    distributed = False
    local_rank = 0
    rank = 0
    world_size = 1
    master_addr = "127.0.0.1"
    master_port = 29500

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(root_dir, "data")
    exp_root = os.path.join(root_dir, "data", "exp")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list, valid_keys: set | None = None):
    if len(arg_list) % 2 == 0:
        return dict(zip(arg_list[::2], arg_list[1::2]))

    if not valid_keys:
        raise ValueError(
            "Override list has odd length; it must be key/value pairs."
            f" Received: {arg_list}"
        )

    opts = {}
    idx = 0
    while idx < len(arg_list):
        key = arg_list[idx]
        if idx + 1 >= len(arg_list):
            opts[key] = ""
            idx += 1
            continue
        candidate = arg_list[idx + 1]
        if candidate in valid_keys:
            opts[key] = ""
            idx += 1
            continue
        opts[key] = candidate
        idx += 2
    return opts


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default=[0], type=int, nargs="+")
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    valid_keys = {
        fieldkey
        for fieldkey in dir(cfg)
        if not fieldkey.startswith("__")
        and not fieldkey.startswith("_")
        and not callable(getattr(cfg, fieldkey))
    }
    opts = args_list2dict(args.opts, valid_keys=valid_keys)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type is type(None):
            setattr(cfg, k, v)
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

cfg.gpus: list = args.gpus
existing_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if "LOCAL_RANK" not in os.environ:
    if existing_cuda_devices and existing_cuda_devices.strip():
        print(
            "Detected external CUDA_VISIBLE_DEVICES="
            f"{existing_cuda_devices}; skipping override for --gpus={cfg.gpus}."
        )
    else:
        # --gpus uses indices within the visible GPU list (after CUDA_VISIBLE_DEVICES mapping).
        os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
cfg.exp_name = args.exp_name
ckpt_arg = args.ckpt
if os.path.isabs(ckpt_arg):
    cfg.ckpt_path = ckpt_arg
elif os.sep in ckpt_arg or (os.altsep and os.altsep in ckpt_arg):
    cfg.ckpt_path = os.path.normpath(os.path.join(cfg.root_dir, ckpt_arg))
else:
    cfg.ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_arg)

if isinstance(cfg.datasets, str):
    cfg.datasets = cfg.datasets.split(",")
