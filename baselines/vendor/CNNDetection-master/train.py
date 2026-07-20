import copy
import os
import time
import torch
import torch.distributed as dist
import torch.nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions


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

    train_loader, train_sampler = create_dataloader(opt, lmdb_path=opt.train_lmdb, is_train=True)
    dataset_size = len(train_loader.dataset)
    if opt.rank == 0:
        print('#training images = %d' % dataset_size)

    if opt.rank == 0:
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    else:
        train_writer = None

    opt.steps_per_epoch = len(train_loader)
    opt.total_train_steps = len(train_loader) * opt.epochs

    model = Trainer(opt)
    if opt.distributed:
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    for epoch in range(opt.epoch_count, opt.epoch_count + opt.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        progress_bar = None
        if opt.rank == 0:
            progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{opt.epoch_count + opt.epochs - 1}",
                                dynamic_ncols=True)

        for i, data in enumerate(train_loader):
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if opt.rank == 0:
                current_loss = model.loss.item() if torch.is_tensor(model.loss) else float(model.loss)
                current_lr = model.get_current_lr()
                if train_writer:
                    train_writer.add_scalar('loss', current_loss, model.total_steps)
                    train_writer.add_scalar('lr', current_lr, model.total_steps)
                if progress_bar is not None:
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.6f}")
                    progress_bar.update(1)

        model.step_scheduler()

        if opt.rank == 0:
            if progress_bar is not None:
                progress_bar.close()
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

    if opt.distributed:
        dist.destroy_process_group()
