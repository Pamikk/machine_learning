import torch.utils.data as data
import torch
import json
import numpy as np
import random
import cv2
import os
from torch.nn import functional as F
def valid_scale(src,vs):
    vs = random.uniform(-vs,vs)
    img = src.astype(np.float)
    img*= (1+vs)
    img[img>255] = 255
    img = img.astype(np.uint8)
    return img
def resize(src,tsize):
    dst = cv2.resize(src,(tsize[1],tsize[0]),interpolation=cv2.INTER_LINEAR)
    return dst
def shear(src,shear):
    h,w = src.shape
    sx = random.uniform(-shear,shear)
    sy = random.uniform(-shear,shear)
    mat = np.array([[1,sx,0],[sy,1,0]])    
    dst = cv2.warpAffine(src,mat,(w,h))
    return dst
def rotate(src,ang,scale):
    h,w = src.shape
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, scale)
    dst = cv2.warpAffine(src,mat,(w,h))
    return dst
def load_mnist(path,mode,r=1):
    label_path = os.path.join(path,f'{mode}-labels.idx1-ubyte')
    img_path = os.path.join(path,f'{mode}-images.idx3-ubyte')
    with open(label_path, 'rb') as f:
        f.read(8)
        labels = np.fromfile(f, dtype=np.uint8)

    with open(img_path, 'rb') as f:
        f.read(16)
        images = np.fromfile(f,dtype=np.uint8).reshape(len(labels), 784)
    n = int(len(labels)*r)
    return images[:n], labels[:n]
class mnist(data.Dataset):
    def __init__(self,cfg,mode='train'):
        path = cfg.img_path
        self.cfg = cfg
        self.mode = mode
        self.accm_batch = 0
        self.size = cfg.size
        if mode != 'test':
            self.images,self.labels  = load_mnist(path,'train')
            self.num = len(self.labels)
        else:
            self.images,self.labels = load_mnist(path,'t10k')
            self.images = self.images[:256]
            self.labels = self.images[:256]
        self.mean = self.images.mean()
        self.std = self.images.std()
    def __len__(self):
        return len(self.labels)

    def img_to_tensor(self,img):
        data = torch.tensor(img,dtype=torch.float)
        if data.max()>1:
             data /= 255.0
        data = (data-0.5)*2
        return data.reshape(self.cfg.in_channel,*self.size)
    def __getitem__(self,idx):
        img = self.images[idx].reshape(self.size)
        data = self.img_to_tensor(img)
        #label = torch.tensor([self.labels[idx]],dtype=torch.long)
        return data#,label

                





