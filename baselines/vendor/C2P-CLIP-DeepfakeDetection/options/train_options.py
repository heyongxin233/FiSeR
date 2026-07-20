from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--earlystop_epoch", type=int, default=15)
        parser.add_argument("--data_aug", action="store_true", help="enable extra blur/jpeg augmentation")
        parser.add_argument("--optim", type=str, default="adamw", help="optimizer to use [adam, adamw, sgd]")
        parser.add_argument("--new_optim", action="store_true", help="ignore optimizer state when resuming")
        parser.add_argument("--loss_freq", type=int, default=50, help="logging frequency in steps")
        parser.add_argument("--save_epoch_freq", type=int, default=1, help="checkpoint frequency in epochs")
        parser.add_argument("--continue_train", action="store_true", help="resume from checkpoint")
        parser.add_argument("--epoch_count", type=int, default=1, help="starting epoch count")
        parser.add_argument("--last_epoch", type=int, default=-1, help="scheduler warm start epoch")
        parser.add_argument("--train_split", type=str, default="train")
        parser.add_argument("--val_split", type=str, default="val")
        parser.add_argument("--niter", type=int, default=10, help="number of epochs")
        parser.add_argument("--epochs", type=int, default=None, help="override niter")
        parser.add_argument("--total_steps", type=int, default=0, help="optional max train steps; 0 disables the cap")
        parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for Adam/AdamW")

        parser.add_argument("--train_lmdb", type=str, default=None, help="path to training lmdb")
        parser.add_argument("--val_lmdb", type=str, default=None, help="optional path to validation lmdb")
        parser.add_argument("--distributed", action="store_true", help="enable torch.distributed training")
        parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed launch")
        parser.add_argument("--find_unused_parameters", action="store_true", help="enable DDP unused param detection")
        parser.add_argument("--amp", action="store_true", help="enable mixed precision training")
        parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")

        self.isTrain = True
        return parser
