import torch
import time
import numpy as np

from tqdm import tqdm
import os
import json

from utils import Logger,save_imgs
#from utils import cal_correct_num,get_sentence
tosave = ['acc']

class Trainer:
    def __init__(self,cfg,datasets,net,epoch):
        self.cfg = cfg
        if 'train' in datasets:
            self.trainset = datasets['train']
        if 'trainval' in datasets:
            self.trainval = datasets['trainval']
        else:
            self.trainval = False
        if 'test' in datasets:
            self.testset = datasets['test']
        self.net = net
        name = cfg.exp_name
        self.name = name
        self.checkpoints = os.path.join(cfg.checkpoint,name)
        self.device = cfg.device
        self.net = self.net
        self.pred = os.path.join(self.checkpoints,'pred')
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        if not(os.path.exists(self.pred)):
            os.mkdir(self.pred)
        start,total = epoch       
        self.total = total
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate
        self.val_every_k_epoch = cfg.val_every_k_epoch

        self.best_acc = 0
        self.best_acc_epoch = 0
        #train setting
        self.movingAvg = 0
        self.bestMovingAvg = 1e9
        self.bestMovingAvgEpoch = 1e9
        self.early_stop_epochs = 50
        self.alpha = 0.5 #for update moving Avg
        self.save_pred = False
        self.set_lr = cfg.adjust_lr
        self.lr_decay = cfg.lr_factor
        self.cmp = min
        self.lr = cfg.lr
        self.ft_keys = cfg.ft_keys
        self.schedule = cfg.schedule
        self.num = cfg.num
        #load from epoch if required
        if start:
            if start=='-1':
                self.load_last_epoch()
            else:
                self.load_epoch(start.strip())
        else:
            self.start = 0
        self.net = self.net.to(self.device)
    def load_last_epoch(self):
        files = os.listdir(self.checkpoints)
        idx = 0
        for name in files:
            if name[-3:]=='.pt':
                epoch = name[6:-3]
                if epoch=='best' or epoch=='bestm':
                  continue
                idx = max(idx,int(epoch))
        if idx==0:
            exit()
        else:
            self.load_epoch(str(idx))
    def save_epoch(self,idx,epoch):
        saveDict = {'net':self.net.state_dict(),
                    'optimizers': [optz.state_dict() for optz in self.net.optimizers],
                    'epoch':epoch,
                    'acc':self.best_acc,
                    'best_epoch':self.best_acc_epoch,
                    'movingAvg':self.movingAvg,
                    'bestmovingAvg':self.bestMovingAvg,
                    'bestmovingAvgEpoch':self.bestMovingAvgEpoch}
        path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        torch.save(saveDict,path)                  
    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            info = torch.load(model_path)
            self.net.load_state_dict(info['net'])
            for i,optz in enumerate(info['optimizers']):
                self.net.optimizers[i].load_state_dict(optz)#might have bugs about device
                for state in self.net.optimizers[i].state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            if self.set_lr:
                    self.reset_lr(self.lr)
            else:
                self.lr = self.get_cur_lr()
            self.start = info['epoch']+1
            self.best_acc = info['acc']
            self.best_acc_epoch = info['best_epoch']
            self.movingAvg = info['movingAvg']
            self.bestMovingAvg = info['bestmovingAvg']
            self.bestMovingAvgEpoch = info['bestmovingAvgEpoch']
            self.cmp = min
            self.wait = 5
        else:
            print('no such model at:',model_path)
            exit()
    def _updateMetrics(self,val,epoch):
        f = self.cmp
        if (epoch==0):
            return
        if self.movingAvg ==0:
            self.movingAvg = val
        else:
            self.movingAvg = self.movingAvg * self.alpha + val*(1-self.alpha)
        if self.movingAvg == f(self.bestMovingAvg,self.movingAvg):
            self.bestMovingAvg = self.movingAvg
            self.bestMovingAvgEpoch = epoch
            self.save_epoch('bestm',epoch)
            self.wait = 5
            print(self.movingAvg)
        else:
            self.wait-= 1
        if self.wait==0:#adjust on plateu
            self.adjust_lr(self.lr_decay)
            self.wait=5
    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))
    def get_cur_lr(self):
        return self.net.optimizers[0].param_groups[0]['lr']
    def lr_schedule_on_epoch(self,epoch):
        if epoch in self.schedule:
            self.adjust_lr(self.lr_decay)
    def adjust_lr(self,lr_factor):
        for optz in self.net.optimizers:
            for param_group in optz.param_groups:
                param_group['lr']*=lr_factor    
    def reset_lr(self,lr):
        for optz in self.net.optimizers:
            for param_group in optz.param_groups:
                param_group['lr']=lr  
    def train_one_epoch(self):
        running_loss ={'all':0.0,'gloss':0.0,'dloss':0.0,'img_loss':0.0,'kld_loss':0.0}
        self.net.train()
        n = len(self.trainset)
        for data in tqdm(self.trainset):
           loss = self.net(data)
           for k in loss:
               if not np.isnan(loss[k]):
                   running_loss[k]+=loss[k]/n
        self.logMemoryUsage()
        return running_loss
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")
        self.lr=self.get_cur_lr()
        print(self.lr)
        epoch = self.start
        stop_epochs = 0
        while epoch < self.total and stop_epochs<self.early_stop_epochs:
            running_loss = self.train_one_epoch()            
            lr = self.get_cur_lr()
            print(lr)
            self.logger.write_loss(epoch,running_loss,lr)
            #step lr
            #self.lr_schedule_on_epoch(epoch)
            self._updateMetrics(running_loss['all'],epoch)
            #if self.lr <= self.cfg.min_lr+1e-16:
                #stop_epochs +=1
            if (epoch+1)%self.save_every_k_epoch==0:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.val_every_k_epoch==0 and (self.val_every_k_epoch !=-1):                
                self.test(epoch)
            print(f"best so far with {self.best_acc} at epoch:{self.best_acc_epoch}")
            epoch +=1                
        print("Best Accuracy: {:.4f} at epoch {}".format(self.best_acc, self.best_acc_epoch))
        self.save_epoch(str(epoch-1),epoch-1)
    def validate(self,epoch,mode):
        self.net.eval()
        print('start Validation Epoch:',epoch)
        if mode=='val':
            valset = self.valset
        else:
            valset = self.trainval
        with torch.no_grad():
            n_gt = 0.0
            n_cor = 0.0
            for data in tqdm(valset):
                inputs,labels = data
                outs = self.net(inputs.to(self.device).float())
                n_gt += len(labels)                 
        metrics={'acc':n_cor/n_gt}        
        return metrics
    def test(self,epoch):
        self.net.eval()
        pd_num = 0
        path = os.path.join(self.pred,f'epoch_{epoch}')
        if not os.path.exists(path):
            os.mkdir(path)
        with torch.no_grad():
            for data in tqdm(self.testset):
                res = self.net(data)
                save_imgs(path,res,pd_num)
                pd_num+=1

        


                


        




