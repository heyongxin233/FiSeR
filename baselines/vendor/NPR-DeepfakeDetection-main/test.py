import os
import time
import copy
import sys
from pathlib import Path
import torch
import logging
import numpy as np
from tqdm import tqdm

from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from test_logger import log_test_run


def setup_logging():
    logging.basicConfig(filename='log_test.log', level=logging.INFO, format='%(asctime)s %(message)s')
    return logging.getLogger()


def load_model(model_path, device):
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')

    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # Handle checkpoints saved from DataParallel by stripping the leading
    # "module." prefix so the keys match the current model definition.
    if isinstance(state_dict, dict):
        needs_stripping = any(key.startswith('module.') for key in state_dict.keys())
        if needs_stripping:
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    opt = TestOptions().parse(print_options=False)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging()

    log_msg = f'Model_path {opt.model_path}'
    print(log_msg)
    logger.info(log_msg)

    lmdb_paths = [p for p in opt.lmdb_path.split(',') if p]
    eval_names = [n for n in opt.eval_names.split(',') if n]
    if eval_names and len(eval_names) != len(lmdb_paths):
        raise ValueError('Number of eval_names must match number of lmdb paths')
    if not eval_names:
        eval_names = [os.path.basename(path.rstrip('/')) for path in lmdb_paths]

    print(f"Testing on {len(lmdb_paths)} LMDB datasets")
    logger.info(f"Testing on {len(lmdb_paths)} LMDB datasets")

    metrics_collection = []
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(current_time)
    logger.info(current_time)

    model = load_model(opt.model_path, opt.device)

    for v_id, (lmdb_path, name) in enumerate(zip(lmdb_paths, eval_names)):
        eval_opt = copy.deepcopy(opt)
        eval_opt.lmdb_path = lmdb_path
        eval_opt.no_resize = True
        eval_opt.isTrain = False
        eval_opt.distributed = False

        metrics, _, _ = validate(model, eval_opt)
        metrics_collection.append(metrics)
        log_msg = (
            f"({v_id} {name:12}) acc: {metrics['acc'] * 100:.2f}; auroc: {metrics['auroc'] * 100:.2f}; "
            f"pr_auc: {metrics['pr_auc'] * 100:.2f}; F1: {metrics['F1'] * 100:.2f}; "
            f"avg_recall: {metrics['avg_recall'] * 100:.2f}; pos_recall: {metrics['pos_recall'] * 100:.2f}; "
            f"neg_recall: {metrics['neg_recall'] * 100:.2f}; tpr@fpr: {metrics['tpr_at_fpr'] * 100:.2f}"
        )
        print(log_msg)
        logger.info(log_msg)
        log_test_run(
            project_root=Path(__file__).resolve().parent,
            project_name=Path(__file__).resolve().parent.name,
            dataset_path=lmdb_path,
            ckpt_path=opt.model_path,
            metrics=metrics,
            cmd=" ".join(sys.argv),
        )

    if metrics_collection:
        mean_metrics = {k: np.mean([m[k] for m in metrics_collection]) for k in metrics_collection[0].keys()}
        log_msg = (
            f"({len(metrics_collection)} {'Mean':10}) acc: {mean_metrics['acc'] * 100:.2f}; auroc: {mean_metrics['auroc'] * 100:.2f}; "
            f"pr_auc: {mean_metrics['pr_auc'] * 100:.2f}; F1: {mean_metrics['F1'] * 100:.2f}; "
            f"avg_recall: {mean_metrics['avg_recall'] * 100:.2f}; pos_recall: {mean_metrics['pos_recall'] * 100:.2f}; "
            f"neg_recall: {mean_metrics['neg_recall'] * 100:.2f}; tpr@fpr: {mean_metrics['tpr_at_fpr'] * 100:.2f}"
        )
        print(log_msg)
        logger.info(log_msg)
    print('*' * 25)
    logger.info('*' * 25)


if __name__ == '__main__':
    main()
