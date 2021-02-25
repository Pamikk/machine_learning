import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss(reduction='sum')
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
def NetAPI(cfg,net,init=False):
    networks = {'GAN':GAN,'VAE':VAE}
    network = networks[net](cfg)
    if init:
        network.apply(init_weights)
    return network

class Generator(nn.Module):
    def __init__(self,cfg):
        super(Generator,self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.in_channel = cfg.latent_dim
        layers = []
        channels = [128,256,512]
        for i,channel in enumerate(channels):
            if i>0:
                layers.append(nn.Sequential(nn.Linear(self.in_channel,channel),nn.BatchNorm1d(channel, 0.8),self.relu))
            else:
                layers.append(nn.Sequential(nn.Linear(self.in_channel,channel),self.relu))
            self.in_channel = channel
        self.layers = nn.Sequential(*layers)
        self.size = cfg.size
        self.final = nn.Linear(self.in_channel,int(np.prod(self.size)))
        self.img_channel = cfg.in_channel
        

    def forward(self, z):
        img = self.layers(z)
        img = F.tanh(self.final(img))
        img = img.view(img.shape[0], self.img_channel, *self.size)
        return img
class Disciminator(nn.Module):
    def __init__(self,cfg):
        super(Disciminator,self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.layers= nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(cfg.in_channel*np.prod(cfg.size)), 512),
            self.relu,
            nn.Linear(512,256),
            self.relu,
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)
class GAN(nn.Module):
    def __init__(self,cfg):
        super(GAN,self).__init__()
        self.generator = Generator(cfg)
        self.discriminator = Disciminator(cfg)
        self.latent_dim = cfg.latent_dim
        self.optimizers = [optim.Adam(self.generator.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay),optim.Adam(self.discriminator.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)]
    def forward(self,data=None):  
        bs =data.shape[0]                  
        if not self.training:
            z = torch.tensor(np.random.normal(0,1,(bs,self.latent_dim))).float().cuda() 
            gen_img  = self.generator(z)
            return gen_img
        data =data.float().cuda() 
        z = torch.tensor(np.random.normal(0,1,(bs,self.latent_dim))).float().cuda()  
        optz_g,optz_d = self.optimizers
        real = torch.ones(bs,1,requires_grad=False).float().cuda()
        fake = torch.zeros(bs,1,requires_grad=False).float().cuda()
        
        gen_img = self.generator(z)      
        
        

        optz_d.zero_grad()
        loss_d = (bce_loss(self.discriminator(data),real)+bce_loss(self.discriminator(gen_img.detach()),fake))/2
        loss_d.backward(retain_graph=True)
        optz_d.step() 

        optz_g.zero_grad() 
        loss_g = bce_loss(self.discriminator(gen_img),real)
        loss_g.backward()
        optz_g.step()
        
        
          

        self.optimizers = [optz_g,optz_d]
        res={'gloss':loss_g.item(),'dloss':loss_d.item(),'all':loss_g.item()+loss_d.item()}
        return res
channels = [512,256]
class Encoder(nn.Module):
    def __init__(self,cfg):
        super(Encoder,self).__init__()
        self.in_channel = int(cfg.in_channel*np.prod(cfg.size))
        layers = []
        self.relu = nn.ReLU()
        for channel in channels:
            layers.append(nn.Sequential(nn.Linear(self.in_channel,channel),self.relu))
            self.in_channel = channel
        self.layers = nn.Sequential(*layers)
        self.final_mu = nn.Linear(self.in_channel,cfg.latent_dim_vae)
        self.final_var = nn.Linear(self.in_channel,cfg.latent_dim_vae)
    def forward(self,x):
        x = x.reshape(-1,784)
        x = self.layers(x)
        mu = self.final_mu(x)
        log_var = self.final_var(x)
        return [mu,log_var]
class Decoder(nn.Module):
    def __init__(self,cfg):
        super(Decoder,self).__init__()
        self.in_channel = cfg.latent_dim_vae
        self.relu = nn.ReLU()
        layers = []
        self.size = cfg.size
        for channel in channels[::-1]:
            layers.append(nn.Sequential(nn.Linear(self.in_channel,channel),self.relu))
            self.in_channel = channel
        self.final = nn.Linear(self.in_channel,int(np.prod(self.size)))
        self.layers = nn.Sequential(*layers)
        self.img_channel = cfg.in_channel
        
    def forward(self, z):
        img = self.layers(z)
        img = F.tanh(self.final(img))
        img = img.view(img.shape[0], self.img_channel, *self.size)
        return img
class VAE(nn.Module):
    def __init__(self,cfg):
        super(VAE,self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.optimizers = [optim.Adam(self.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)]
        self.img_channel = cfg.in_channel
        self.bs = cfg.bs
        self.size = cfg.size
        self.latent_dim = cfg.latent_dim_vae
    def reparameterize(self,mu,log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std).float().cuda() 
        return mu+eps*std
    def compute_loss(self,img,data,mu,log_var):
        img_loss = mse_loss(img,data)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        loss = img_loss + kld_loss
        return loss,{'all':loss.item(),'img_loss':img_loss.item(),'kld_loss':kld_loss.item()}
    def forward(self,data=None):
        data = data.float().cuda()
        if self.training:
            optz = self.optimizers[0]
            mu, log_var = self.encoder(data)
            z = self.reparameterize(mu, log_var)
            gen_img = self.decoder(z)
            optz.zero_grad()            
            loss,running_loss = self.compute_loss(gen_img,data,mu,log_var)
            loss.backward()
            optz.step()
            self.optimizers[0] = optz            
            return running_loss            
        else:
            z = torch.tensor(np.random.normal(0,1,(data.shape[0],self.latent_dim))).float().cuda()
            #z = torch.rand(data.shape[0],self.latent_dim).float().cuda()
            gen_img = self.decoder(z)
            return gen_img



    




    




    