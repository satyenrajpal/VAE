import torch.nn as nn
import math
import torch

from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, img_size=32, enc_conv_dim=32, dec_conv_dim=512, z_dim=20, out_c=1):
        super(VAE, self).__init__()

        enc =[]
        enc.append(nn.Conv2d(out_c, enc_conv_dim, 4,2,1))
        enc.append(nn.ReLU(inplace=True))

        # Encoder
        num_steps = int(math.log2(img_size))-1
        curr_dim = enc_conv_dim
        for i in range(num_steps-1):
            enc.append(nn.Conv2d(curr_dim, curr_dim*2, 4, 2, 1))
            enc.append(nn.BatchNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            enc.append(nn.ReLU(True))
            curr_dim = curr_dim*2

        self.encoder = nn.Sequential(*enc)
        self.mu = nn.Conv2d(curr_dim, z_dim, 2, 1)
        self.logvar = nn.Conv2d(curr_dim , z_dim, 2, 1)
        
        # Decoder
        dec = []
        dec.append(nn.ConvTranspose2d(z_dim, dec_conv_dim, 1, 1))
        dec.append(nn.ReLU(True))
        curr_dim = dec_conv_dim
        
        for i in range(num_steps):
            dec.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1))
            dec.append(nn.BatchNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            dec.append(nn.ReLU(True))
            curr_dim = curr_dim//2

        dec.append(nn.ConvTranspose2d(curr_dim, out_c, 4, 2, 1))
        dec.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec)

        self.img_size = img_size

    def encode(self, x):
        h1 = self.encoder(x)
        return self.mu(h1), self.logvar(h1)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x_hat = self.decoder(z)
        assert x_hat.size(2) == x_hat.size(3) == self.img_size
        return x_hat
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
