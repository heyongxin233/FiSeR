import copy
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm.auto import tqdm
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        class SummaryWriter:  # type: ignore[override]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

from data import create_dataloader
from models.trainer import Trainer
from options.train_options import TrainOptions


SEED = 0


def is_main_process(opt):
    return (not opt.distributed) or opt.local_rank == 0


def setup_distributed(opt):
    if opt.distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_distributed(opt):
    if opt.distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(rank=0):
    seed = SEED + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_val_opt(opt):
    if not opt.val_lmdb:
        return None

    val_opt = copy.deepcopy(opt)
    val_opt.isTrain = False
    val_opt.serial_batches = True
    val_opt.data_label = "val"
    val_opt.lmdb_path = opt.val_lmdb
    return val_opt


def gather_numpy_array(array, opt):
    if not opt.distributed:
        return array

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, array)
    arrays = [np.asarray(item) for item in gathered if item is not None and len(item) > 0]
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def calculate_acc(y_true, y_pred, threshold=0.5):
    real_mask = y_true == 1
    fake_mask = y_true == 0
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask] > threshold) if real_mask.any() else 0.0
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask] > threshold) if fake_mask.any() else 0.0
    acc = accuracy_score(y_true, y_pred > threshold)
    return real_acc, fake_acc, acc


@torch.no_grad()
def evaluate(model, data_loader, opt):
    if data_loader is None:
        return None

    core_model = model.get_core_model()
    was_training = core_model.training
    core_model.eval()

    prediction_chunks = []
    label_chunks = []
    for images, labels in data_loader:
        images = images.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True).float()
        with model.autocast_context():
            logits = core_model(images).view(-1)
        prediction_chunks.append(torch.sigmoid(logits.float()).cpu())
        label_chunks.append(labels.cpu())

    if was_training:
        core_model.train()

    if not prediction_chunks:
        return None

    y_pred = torch.cat(prediction_chunks).numpy()
    y_true = torch.cat(label_chunks).numpy()
    y_pred = gather_numpy_array(y_pred, opt)
    y_true = gather_numpy_array(y_true, opt)

    if not is_main_process(opt):
        return None

    ap = average_precision_score(y_true, y_pred)
    real_acc, fake_acc, acc = calculate_acc(y_true, y_pred)
    return {
        "ap": ap,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "acc": acc,
    }


def main():
    opt = TrainOptions().parse()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    setup_distributed(opt)

    train_writer = None
    val_writer = None
    try:
        set_seed(opt.local_rank if opt.distributed else 0)

        model = Trainer(opt)
        data_loader = create_dataloader(opt)
        val_opt = build_val_opt(opt)
        val_loader = create_dataloader(val_opt) if val_opt is not None else None

        if is_main_process(opt):
            train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
            if val_loader is not None:
                val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

            print(f"Length of data loader: {len(data_loader)}")
            with open(os.path.join(opt.checkpoints_dir, opt.name, "log.txt"), "a") as handle:
                handle.write(f"Length of data loader: {len(data_loader)}\n")
            model.save_networks("model_epoch_init.pth")

        for epoch in range(opt.niter):
            if hasattr(data_loader.sampler, "set_epoch"):
                data_loader.sampler.set_epoch(epoch)
            if val_loader is not None and hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)

            epoch_total_steps = len(data_loader)
            if opt.max_steps_per_epoch > 0:
                epoch_total_steps = min(epoch_total_steps, opt.max_steps_per_epoch)

            epoch_iterator = data_loader
            progress_bar = None
            if is_main_process(opt):
                progress_bar = tqdm(
                    data_loader,
                    total=epoch_total_steps,
                    desc=f"Epoch {epoch + 1}/{opt.niter}",
                    dynamic_ncols=True,
                )
                epoch_iterator = progress_bar

            for step_idx, data in enumerate(epoch_iterator, start=1):
                model.total_steps += 1
                model.set_input(data)
                model.optimize_parameters()

                if progress_bar is not None:
                    progress_bar.set_postfix(
                        step=model.total_steps,
                        loss=f"{float(model.loss.detach().item()):.4f}",
                        lr=f"{model.lr:.2e}",
                    )

                if is_main_process(opt) and opt.loss_freq > 0 and model.total_steps % opt.loss_freq == 0:
                    loss_value = float(model.loss.detach().item())
                    train_writer.add_scalar("loss", loss_value, model.total_steps)

                if opt.max_steps_per_epoch > 0 and step_idx >= opt.max_steps_per_epoch:
                    break

            if progress_bar is not None:
                progress_bar.close()

            if is_main_process(opt) and epoch % opt.save_epoch_freq == 0:
                print(f"saving the model at the end of epoch {epoch}")
                model.train()
                model.save_networks(f"model_epoch_{epoch}.pth")

            metrics = evaluate(model, val_loader, opt)
            if metrics is not None and is_main_process(opt):
                val_writer.add_scalar("accuracy", metrics["acc"], model.total_steps)
                val_writer.add_scalar("ap", metrics["ap"], model.total_steps)
                print(
                    f"(Val @ epoch {epoch}) acc: {metrics['acc']}; ap: {metrics['ap']}; "
                    f"real_acc: {metrics['real_acc']}; fake_acc: {metrics['fake_acc']}"
                )

            model.train()
    finally:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        cleanup_distributed(opt)


if __name__ == "__main__":
    main()
