import argparse
import os


class ConfigurationManager:
    def __init__(self):
        self.config_initialized = False
        self.argument_parser = None

    def define_arguments(self, argument_parser):
        argument_parser.add_argument('--name', type=str, default='lota_wildfake')
        argument_parser.add_argument('--train_lmdb', type=str, default='')
        argument_parser.add_argument('--val_lmdb', type=str, default='')
        argument_parser.add_argument('--lmdb_path', type=str, default='')
        argument_parser.add_argument('--eval_names', type=str, default='')
        argument_parser.add_argument('--model_path', type=str, default='')
        argument_parser.add_argument('--load', type=str, default='')
        argument_parser.add_argument('--save_path', type=str, default='checkpoints')
        argument_parser.add_argument('--results_dir', type=str, default='results')

        argument_parser.add_argument('--batch_size', '--batchsize', dest='batch_size', type=int, default=64)
        argument_parser.add_argument('--val_batch_size', '--val_batchsize', dest='val_batch_size', type=int, default=128)
        argument_parser.add_argument('--epochs', '--epoch', dest='epochs', type=int, default=30)
        argument_parser.add_argument('--lr', type=float, default=1e-4)
        argument_parser.add_argument('--min_lr', type=float, default=0.0)
        argument_parser.add_argument('--weight_decay', type=float, default=0.0)
        argument_parser.add_argument('--scheduler', type=str, default='poly', choices=['poly', 'cosine'])
        argument_parser.add_argument('--warmup_epochs', type=int, default=0)
        argument_parser.add_argument('--num_workers', type=int, default=8)
        argument_parser.add_argument('--seed', type=int, default=42)
        argument_parser.add_argument('--max_steps', type=int, default=0)
        argument_parser.add_argument('--max_eval_batches', type=int, default=0)

        argument_parser.add_argument('--img_height', type=int, default=256)
        argument_parser.add_argument('--bit_mode', type=str, default='scaling', choices=['scaling', 'thresholding'])
        argument_parser.add_argument('--patch_size', type=int, default=32)
        argument_parser.add_argument('--patch_mode', type=str, default='max', choices=['max', 'min', 'random'])
        argument_parser.add_argument('--no_patch', action='store_true')

        argument_parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'])
        argument_parser.add_argument('--no_pretrained', action='store_true')
        argument_parser.add_argument('--pretrained_dir', type=str, default=os.environ.get('TORCH_HOME', ''))

        argument_parser.add_argument('--distributed', action='store_true')
        argument_parser.add_argument('--rank', type=int, default=0)
        argument_parser.add_argument('--world_size', type=int, default=1)
        argument_parser.add_argument('--local_rank', type=int, default=0)
        argument_parser.add_argument('--master_addr', type=str, default='127.0.0.1')
        argument_parser.add_argument('--master_port', type=str, default='29500')
        return argument_parser

    def collect_arguments(self):
        if not self.config_initialized:
            argument_parser = argparse.ArgumentParser(
                description='LOTA configuration',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
            argument_parser = self.define_arguments(argument_parser)
            self.config_initialized = True
            self.argument_parser = argument_parser
        return self.argument_parser.parse_args()

    def display_configuration(self, config):
        print('===== LOTA Configuration =====')
        for parameter, value in sorted(vars(config).items()):
            print(f'{parameter}: {value}')
        print('==============================')

    def parse(self, display_settings=True):
        config = self.collect_arguments()
        config.isTrain = False
        config.isVal = False
        config.use_patch = not config.no_patch
        config.load = config.load or ''
        config.model_path = config.model_path or ''
        config.output_dir = os.path.join(config.save_path, config.name)
        if display_settings:
            self.display_configuration(config)
        self.config = config
        return self.config
