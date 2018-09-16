# VAE
My implementation of a Basic [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114). There are a few changes in comparison to this paper - 
 - Instead of upsampling by Nearest Neighbors, Conv Transpose is used
 - Reconstruction loss is a simple l2 loss

The dataset is sourced from [this repo](https://github.com/cianeastwood/qedr).   

Results - 
![](/docs/1_sample_1999.png) | ![](/docs/2_sample_99.png)
<img src="/docs/1_sample_1999.png" width="300" hspace="20" /> <img src="/docs/2_sample_99.png" width="300"/> 