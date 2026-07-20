import sys
import os
import torch


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)


def resolve_hf_pretrained_path(name: str) -> str:
    if not name or os.path.isdir(name) or os.path.isfile(name):
        return name

    cache_roots = []
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        cache_roots.append(hub_cache)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(os.path.join(hf_home, "hub"))

    cache_roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

    repo_dir_name = "models--" + name.replace("/", "--")
    for cache_root in cache_roots:
        snapshots_dir = os.path.join(cache_root, repo_dir_name, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        snapshot_names = sorted(os.listdir(snapshots_dir))
        for snapshot_name in reversed(snapshot_names):
            snapshot_path = os.path.join(snapshots_dir, snapshot_name)
            if os.path.isdir(snapshot_path):
                return snapshot_path

    return name
