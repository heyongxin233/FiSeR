import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from utils.util import Logger
import random


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(opt.seed)

    Test_dataroot = os.path.join(opt.dataroot,'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    Testopt = TestOptions().parse(print_options=False)
    Test_vals = os.listdir(Test_dataroot)
    data_loader = create_dataloader(opt)
    model = Trainer(opt)
    
    def testmodel(epoch=0):
        print('*'*25);accs = [];aps = [];logs=[f"Testing end of {epoch}"]
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(Test_vals):
            Testopt.dataroot = f"{Test_dataroot}/{val}"
            # Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.loadSize = opt.cropSize
            Testopt.cropSize = opt.cropSize
            Testopt.no_resize = False  
            Testopt.no_crop = False 
            Testopt.classes = ''
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            logs.append("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
            print(logs[-1])
        logs.append("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
        print(logs[-1]);print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        return round(np.array(accs).mean()*100, 4)
    model.train()
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            if model.total_steps > opt.total_steps: break 
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} loss1: {} loss2-cla: {} at step: {} lr {}".format(model.loss, model.loss1, model.loss2, model.total_steps, model.lr))

            # if model.total_steps % opt.eval_freq == 0:
            #     print(os.getcwd())
            #     print(f'==========total_steps {model.total_steps}=================')
            #     model.eval()
            #     testacc = testmodel(epoch)
            #     model.save_networks( f'{str(epoch)}_total_steps_{str(model.total_steps)}_testacc_{str(testacc)}')
            #     model.train()

                
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        model.eval()
        testacc = testmodel(epoch)
        model.save_networks( f'{str(epoch)}_total_steps_{str(model.total_steps)}_testacc_{str(testacc)}')
        print('saving the latest model %s (epoch %d, model.total_steps %d)' %(opt.name, epoch, model.total_steps))
        model.train()
