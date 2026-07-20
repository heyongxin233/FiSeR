import copy
import os

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from validate import validate


def init_distributed(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False
    opt.device = torch.device('cuda', opt.local_rank) if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    opt = TrainOptions().parse()
    init_distributed(opt)

    train_loader, train_sampler = create_dataloader(opt, lmdb_path=opt.train_lmdb, is_train=True, distributed=opt.distributed)
    dataset_size = len(train_loader.dataset)
    if opt.rank == 0:
        print('#training images = %d' % dataset_size)

    val_opt = None
    if opt.val_lmdb:
        val_opt = copy.deepcopy(opt)
        val_opt.isTrain = False
        val_opt.distributed = False
        val_opt.rank = 0
        val_opt.world_size = 1
        val_opt.lmdb_path = opt.val_lmdb

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train")) if opt.rank == 0 else None

    opt.iters_per_epoch = len(train_loader)
    opt.steps_per_epoch = len(train_loader)
    model = Trainer(opt)
    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    for epoch in range(opt.epoch_count, opt.epoch_count + opt.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{opt.epoch_count + opt.epochs - 1}", dynamic_ncols=True) if opt.rank == 0 else None

        for data in train_loader:
            model.set_input(data)
            model.optimize_parameters()

            if opt.rank == 0:
                current_loss = model.loss.item() if torch.is_tensor(model.loss) else float(model.loss)
                if train_writer:
                    train_writer.add_scalar('loss', current_loss, model.total_steps)
                    train_writer.add_scalar('lr', model.get_current_lr(), model.total_steps)
                if progress_bar is not None:
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{model.get_current_lr():.6f}")
                    progress_bar.update(1)

        if opt.rank == 0 and progress_bar is not None:
            progress_bar.close()

        if opt.rank == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

        if val_opt is not None and opt.rank == 0:
            model.eval()
            metrics, _, _ = validate(model.model if hasattr(model, 'model') else model, val_opt, lmdb_path=opt.val_lmdb, distributed=False)
            log_msg = (
                f"(Val @ epoch {epoch}) acc: {metrics['acc'] * 100:.1f}; auroc: {metrics['auroc'] * 100:.1f}; "
                f"pr_auc: {metrics['pr_auc'] * 100:.1f}; F1: {metrics['F1'] * 100:.1f}; "
                f"avg_recall: {metrics['avg_recall'] * 100:.1f}; pos_recall: {metrics['pos_recall'] * 100:.1f}; "
                f"neg_recall: {metrics['neg_recall'] * 100:.1f}; tpr@fpr: {metrics['tpr_at_fpr'] * 100:.1f}"
            )
            print(log_msg)
            if train_writer:
                for key, value in metrics.items():
                    train_writer.add_scalar(f'val/{key}', value, epoch)
            model.train()

    if opt.distributed:
        dist.destroy_process_group()
