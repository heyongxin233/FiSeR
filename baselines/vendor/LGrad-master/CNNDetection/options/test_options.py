from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--lmdb_path', type=str, default=None, help='comma separated lmdb paths for evaluation')
        parser.add_argument('--eval_names', type=str, default=None, help='optional comma separated names matching lmdb_path')
        parser.add_argument('--distributed', action='store_true', help='enable distributed evaluation')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed evaluation')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')

        self.isTrain = False
        return parser
