from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--model_path", type=str, required=True, help="checkpoint path")
        parser.add_argument("--lmdb_path", type=str, default=None, help="comma separated lmdb paths for evaluation")
        parser.add_argument("--eval_names", type=str, default=None, help="optional comma separated names for lmdb_path")
        parser.add_argument("--distributed", action="store_true", help="enable distributed evaluation")
        parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed launch")
        parser.add_argument("--no_resize", action="store_true")
        parser.add_argument("--no_crop", action="store_true")
        parser.add_argument("--eval", action="store_true", help="use eval mode during test time")
        parser.add_argument("--amp", action="store_true", help="enable mixed precision inference")
        parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")

        self.isTrain = False
        return parser
