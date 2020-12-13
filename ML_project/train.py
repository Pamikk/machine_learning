###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
###files
from config import Config as cfg
from dataProcessing import mnist as dataset
from models.network import NetAPI
from trainer import Trainer
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')
def main(args,cfgs):
    #get data config
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config  = cfgs['train']
    val_cfg = cfgs['val']
    train_set = dataset(config)
    test_set = dataset(val_cfg,mode='test')
    if args.bs:
        config.bs = args.bs
    train_loader = DataLoader(train_set,batch_size=config.bs,shuffle=True,pin_memory=False)
    trainval_loader = DataLoader(train_set,batch_size=val_cfg.bs,shuffle=False,pin_memory=False)
    test_loader = DataLoader(test_set,batch_size=val_cfg.bs,shuffle=False,pin_memory=False)
    datasets = {'train':train_loader,'trainval':trainval_loader,'test':test_loader}
    config.exp_name = args.exp
    config.device = torch.device("cuda")
    torch.cuda.empty_cache()
    if args.lr:
        config.lr = args.lr 
        config.adjust_lr = True
    #config.lr /= config.bs
    network = NetAPI(config,args.net)
    torch.cuda.empty_cache()
    det = Trainer(config,datasets,network,(args.resume,args.epochs))
    if args.mode=='test':
        det.test(det.start)
    else:
        det.train()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="start from epoch?")
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    parser.add_argument("--mode",type=str,default='train',help="only validation")
    parser.add_argument("--seed",type=int,default=2333)
    parser.add_argument("--net",type=str,default='SCCM',help="network type:SCCM|SCAN")
    parser.add_argument("--bs",type=int,default=None,help="batchsize")
    parser.add_argument("--lr",type=float,default=None)
    args = parser.parse_args()
    cfgs = {}
    cfgs['train'] = cfg()
    cfgs['trainval'] = cfg('trainval')
    cfgs['val'] = cfg('val')
    cfgs['test'] = cfg('test')
    main(args,cfgs)
    
    

    