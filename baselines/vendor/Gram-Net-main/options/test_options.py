from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')

        for action in parser._actions:
            if '--lmdb_path' in action.option_strings:
                action.required = True
                action.help = 'lmdb path for evaluation data (comma separated for multiple sets)'
                break
        parser.add_argument('--eval_names', type=str, default='', help='optional comma separated names matching lmdb paths')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
