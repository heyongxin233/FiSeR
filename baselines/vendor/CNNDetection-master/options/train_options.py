from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count; models are saved once per epoch')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--epochs', type=int, default=30, help='number of training epochs to run')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--warmup_steps', type=int, default=2000, help='number of warmup steps (capped at one epoch)')
        parser.add_argument('--min_lr_ratio', type=float, default=0.01, help='final lr ratio for cosine annealing')
        parser.add_argument('--train_lmdb', type=str, required=True, help='lmdb path for training data')
        parser.add_argument('--val_lmdb', type=str, default='', help='optional lmdb path for validation data (not used during training)')

        self.isTrain = True
        return parser
