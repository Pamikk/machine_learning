
import numpy as np
import random
import json
#Train Setting
dataset = 'mnist'
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.img_path = f'../dataset/{dataset}' 
        self.checkpoint='../checkpoints'#path to save model weights
        self.bs = 16
        self.exp = 'exp' #default experiment name
        #data Setting
        self.cls_num = 10
        self.size = (28,28)
        self.ratio = 0.8
        self.latent_dim = 100
        self.latent_dim_vae = 32
        self.num = 10 #test generate 
        self.in_channel = 1  
        if mode=='train':
            self.bs = 64 # batch size            
            #augmentation parameter
            #train_setting
            self.lr = 0.1
            self.weight_decay=5e-4
            self.momentum = 0.9
            #lr_scheduler
            self.min_lr = 1e-7
            self.lr_factor = 0.25
            self.patience = 10
            self.schedule = []
            #exp_setting
            self.save_every_k_epoch = 5
            self.val_every_k_epoch = 10
            self.adjust_lr = False
            self.ft_keys = [] #fine tune keys
        
