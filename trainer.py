import torch
from models import VAE
from torch import nn, optim
from datasets import getLoader
from torchvision.utils import save_image
import os
from torch.nn import functional as F

class Trainer():
    def __init__(self, config):
        
        # Training  
        self.epochs = config.epochs
        self.dataset = config.dataset
        self.img_dir = config.img_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.out_c = 1 if config.dataset=='MNIST' else 3

        # Directories
        self.sample_save_dir = config.sample_save_dir
        self.model_save_dir = config.model_save_dir

        # Model Hyperparameters
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.enc_conv_dim = config.enc_conv_dim
        self.dec_conv_dim = config.dec_conv_dim
        self.z_dim = config.z_dim
        self.lr = config.lr

        # Printing paramters
        self.logStep = config.logStep
        self.sampleStep = config.sampleStep 
        self.modelSaveStep = config.modelSaveStep
        
        self.loadNetworks()
        
    def loadNetworks(self):
        self.vae = VAE(self.img_size, self.enc_conv_dim, 
                       self.dec_conv_dim, self.z_dim, self.out_c)
        self.opt = optim.Adam(self.vae.parameters(), lr=self.lr)
        
        self.vae = self.vae.to(self.device)

    def denorm(self, x):
        x = (x + 1)/2
        return x.clamp_(0,1) 

    def save_models(self,epoch, num_iters):
        path  = os.path.join(self.model_save_dir, '{}-{}-vae.ckpt'.format(epoch,num_iters))
        torch.save(self.vae.state_dict(),  path)
        print("Models saved at {} for {} epoch and {} iter".format(self.model_save_dir,epoch,num_iters))

    def train(self):
        self.vae.train()

        dataLoader = getLoader(self.mode, self.img_size, self.batch_size, self.img_dir,
                               self.dataset, self.num_workers)
        
        for epoch in range(self.epochs):
            for idx, data in enumerate(dataLoader):
                if self.dataset=='MNIST':
                    x , _ = data
                else:
                    x = data 
                x = x.to(self.device)
                x_hat, mu, logvar = self.vae(x)
                
                # Reconstruction Loss
                rec_loss = torch.sum((x_hat - x).pow(2))

                # KL Divergence
                KLD = -0.5 * torch.sum(1+ logvar - mu.pow(2) -logvar.exp())
                loss = rec_loss + KLD
                
                self.opt.zero_grad()
                loss.backward()

                self.opt.step()

                if (idx+1) % self.logStep == 0:
                    print("For Epoch- [{}/{}] - {}/{} \t Loss:{:.6f} ".format(epoch,
                     self.epochs, idx*x.size(0), len(dataLoader.dataset), loss.item()/len(x)))
                
                if (idx + 1) % self.sampleStep == 0:
                    with torch.no_grad():
                        z_sample = torch.randn(64, self.z_dim).view(-1, self.z_dim, 1, 1).to(self.device)
                        x_hat = self.vae.decode(z_sample).cpu()
                        save_image(self.denorm(x_hat.data), 
                            os.path.join(self.sample_save_dir, '{}_sample_{}.png'.format(epoch, idx)))
                
                if (idx + 1) % self.modelSaveStep == 0:
                    self.save_models(epoch, idx)
