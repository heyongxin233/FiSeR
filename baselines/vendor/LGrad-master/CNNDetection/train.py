import os
import sys
import time
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tqdm import tqdm

from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
import random


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def init_distributed(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.local_rank = int(os.environ.get('LOCAL_RANK', opt.local_rank))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(opt.local_rank)
        opt.distributed = True
    elif opt.distributed:
        dist.init_process_group(backend='nccl')
        opt.rank = dist.get_rank()
        opt.world_size = dist.get_world_size()
        torch.cuda.set_device(opt.local_rank)
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.local_rank = 0
        opt.distributed = False

    opt.device = torch.device('cuda', opt.local_rank) if torch.cuda.is_available() else torch.device('cpu')
    opt.gpu_ids = [opt.local_rank] if torch.cuda.is_available() else []


def build_val_opt(train_opt):
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.batch_size = train_opt.batch_size
    val_opt.num_threads = train_opt.num_threads
    val_opt.distributed = train_opt.distributed
    val_opt.rank = getattr(train_opt, 'rank', 0)
    val_opt.world_size = getattr(train_opt, 'world_size', 1)
    val_opt.device = train_opt.device
    val_opt.gpu_ids = train_opt.gpu_ids
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    opt.epochs = opt.epochs or opt.niter
    start_epoch = opt.epoch_count if opt.epoch_count > 0 else 1
    end_epoch = start_epoch + opt.epochs - 1

    init_distributed(opt)
    print('  '.join(list(sys.argv)))

    train_path = opt.train_lmdb or opt.dataroot
    if train_path is None:
        raise ValueError('Please provide --train_lmdb pointing to a training LMDB.')

    val_opt = None
    if opt.val_lmdb:
        val_opt = build_val_opt(opt)
        val_opt.val_lmdb = opt.val_lmdb

    train_loader, train_sampler = create_dataloader(opt, lmdb_path=train_path, is_train=True, distributed=opt.distributed)
    val_loader = None
    if val_opt:
        val_loader, _ = create_dataloader(val_opt, lmdb_path=val_opt.val_lmdb, is_train=False, distributed=opt.distributed)

    total_epochs = end_epoch - start_epoch + 1
    opt.steps_per_epoch = len(train_loader)
    opt.total_train_steps = len(train_loader) * total_epochs

    if opt.rank == 0:
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val")) if val_loader else None
    else:
        train_writer = None
        val_writer = None

    model = Trainer(opt)
    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(
            model.model, device_ids=[opt.local_rank], output_device=opt.local_rank)
    model.train()

    if opt.rank == 0:
        print(f'cwd: {os.getcwd()}')
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Length of data loader: {len(train_loader)}")

    for epoch in range(start_epoch, end_epoch + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{end_epoch}",
                            dynamic_ncols=True) if opt.rank == 0 else None

        running_loss = 0.0
        last_loss = None

        for i, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

            loss_val = model.loss.item() if torch.is_tensor(model.loss) else float(model.loss)
            running_loss += loss_val
            last_loss = loss_val
            avg_loss = running_loss / float(i + 1)

            if model.total_steps % opt.loss_freq == 0 and opt.rank == 0:
                if train_writer:
                    train_writer.add_scalar('loss', loss_val, model.total_steps)
                    train_writer.add_scalar('lr', model.lr, model.total_steps)

            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{last_loss:.4f}" if last_loss is not None else 'n/a',
                    avg_loss=f"{avg_loss:.4f}",
                    lr=f"{model.lr:.3e}",
                )
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        if opt.rank == 0:
            model.save_networks(epoch)

        if val_loader is not None:
            model.eval()
            metrics, y_true, y_pred = validate(
                model.model,
                val_loader,
                device=opt.device,
                distributed=opt.distributed,
                world_size=opt.world_size,
                rank=opt.rank,
                desc=f"Val {epoch}/{end_epoch}")
            if opt.rank == 0 and metrics is not None:
                if val_writer:
                    val_writer.add_scalar('accuracy', metrics['acc'], model.total_steps)
                    val_writer.add_scalar('pr_auc', metrics['pr_auc'], model.total_steps)
                    val_writer.add_scalar('auroc', metrics['auroc'], model.total_steps)
                tqdm.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} (Val @ epoch {epoch}) "
                    f"acc: {metrics['acc']}; pr_auc: {metrics['pr_auc']}; auroc: {metrics['auroc']}; "
                    f"avg_recall: {metrics['avg_recall']}")
            model.train()

    if opt.rank == 0:
        model.eval()
        model.save_networks('last')

    if opt.distributed:
        dist.destroy_process_group()
