import os
import csv
import sys
import torch
import logging
import time
import copy
from pathlib import Path
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from util import Logger, printSet
import numpy as np
from test_logger import log_test_run

logging.basicConfig(filename='log_test.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Running tests
opt = TestOptions().parse(print_options=False)
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = os.path.basename(opt.model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'acc', 'auroc', 'pr_auc', 'F1', 'avg_recall', 'pos_recall', 'neg_recall', 'tpr@fpr']]

log_msg = f'Model_path {opt.model_path}'
print(log_msg)
logger.info(log_msg)
print("{} model testing on...".format(model_name))

lmdb_paths = [p for p in opt.lmdb_path.split(',') if p]
eval_names = [n for n in opt.eval_names.split(',') if n]
if eval_names and len(eval_names) != len(lmdb_paths):
    raise ValueError('Number of eval_names must match number of lmdb paths')
if not eval_names:
    eval_names = [os.path.basename(path.rstrip('/')) for path in lmdb_paths]

metrics_collection = []
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(current_time)
logger.info(current_time)
for v_id, (lmdb_path, name) in enumerate(zip(lmdb_paths, eval_names)):
    eval_opt = copy.deepcopy(opt)
    eval_opt.lmdb_path = lmdb_path
    eval_opt.no_resize = True    # testing without resizing by default
    eval_opt.isTrain = False
    eval_opt.distributed = False

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(opt.device)
    model.eval()

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
        dataset_path=eval_opt.lmdb_path,
        ckpt_path=opt.model_path,
        metrics=metrics,
        cmd=" ".join(sys.argv),
    )

if metrics_collection:
    mean_metrics = {k: np.mean([m[k] for m in metrics_collection]) for k in metrics_collection[0].keys()}
    log_msg = (
        f"({v_id + 1} {'Mean':10}) acc: {mean_metrics['acc'] * 100:.2f}; auroc: {mean_metrics['auroc'] * 100:.2f}; "
        f"pr_auc: {mean_metrics['pr_auc'] * 100:.2f}; F1: {mean_metrics['F1'] * 100:.2f}; "
        f"avg_recall: {mean_metrics['avg_recall'] * 100:.2f}; pos_recall: {mean_metrics['pos_recall'] * 100:.2f}; "
        f"neg_recall: {mean_metrics['neg_recall'] * 100:.2f}; tpr@fpr: {mean_metrics['tpr_at_fpr'] * 100:.2f}"
    )
    print(log_msg)
    logger.info(log_msg)
print('*' * 25)
logger.info('*' * 25)
