import argparse
from trainer import Trainer
import os
import torch

def str2bool(v):
    return v.lower() in ('true')

def mkdir_p(dir_):
	if not os.path.exists(dir_):
		os.makedirs(dir_)

def main(config):

    mkdir_p(config.save_dir)

    if config.mode == 'train':
        torch.manual_seed(1234)

        config.sample_save_dir = os.path.join(config.save_dir, 'samples')
        config.model_save_dir = os.path.join(config.save_dir, 'models')
        
        mkdir_p(config.sample_save_dir)
        mkdir_p(config.model_save_dir)

        trainer = Trainer(config)

        with open(os.path.join(config.save_dir,'config.txt'), 'w') as file_:
            for i in config.__dict__:
                file_.write("{} - {} \n ".format(i, config.__dict__[i]))
        
        print("Training...")
        trainer.train()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--dataset', type=str,default='MNIST',help='Which dataset to train/test model',choices=['MNIST','2d', 'Teapots'])
    parser.add_argument('--z_dim', type=int, default=20, help='dimension of noise vector')
    parser.add_argument('--img_size', type=int, default=32, help='Size of images in dataset')
    parser.add_argument('--enc_conv_dim', type=int, default=32, help='start conv dim for FrontEnd')
    parser.add_argument('--dec_conv_dim',type=int,default=512, help = 'start conv dim for Generator')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for Adam optimizer')
    
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test','calc_score'])
    
    # Directories.
    parser.add_argument('--img_dir', type=str, default='dataset')
    parser.add_argument('--save_dir', type=str, default='VAE')
    
    # Step size.
    parser.add_argument('--logStep', type=int, default=10)
    parser.add_argument('--sampleStep', type=int, default=100)
    parser.add_argument('--modelSaveStep', type=int, default=200)
    
    config = parser.parse_args()
    main(config)    