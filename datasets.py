import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.utils.data as data 
import os, glob
from PIL import Image
import torch
import numpy as np

class Teapots(data.Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.dataset = self.preprocess(img_dir)
        self.transform = transform

    def preprocess(self,img_dir):
        cwd = os.getcwd()
        os.chdir(img_dir)

        labels = [filename for filename in glob.glob("*.jpeg")]
        os.chdir(cwd)
        return labels

    def __getitem__(self, index):
        filename = self.dataset[index]
        img = Image.open(os.path.join(self.img_dir,filename))
        if self.transform is not None:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)


def getLoader(mode, img_size, batch_size, img_dir, dataset,num_workers=4):
    
    transform_op=[]
    transform_op.append(T.Resize((img_size, img_size)))
    transform_op.append(T.ToTensor())
    if (dataset=='MNIST'):
        transform_op.append(T.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_op.append(T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
    transform_op = T.Compose(transform_op)

    if dataset == 'MNIST':
        dataset = dset.MNIST(img_dir, transform=transform_op, download=True)
    if dataset == 'Teapots':
        dataset = Teapots(img_dir, transform=transform_op)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=(mode=='train'),
                            num_workers=num_workers)
    return dataloader