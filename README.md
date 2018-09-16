# VAE
My implementation of a Basic [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114). There are a few changes in comparison to this paper - 
 - Instead of upsampling by Nearest Neighbors, Conv Transpose is used
 - Reconstruction loss is a simple l2 loss

The teapot dataset is sourced from [this repo](https://github.com/cianeastwood/qedr).   

Results - <br>
<img src="/docs/1_sample_1999.png" width="300" hspace="20" /> <img src="/docs/2_sample_99.png" width="300"/> <br>
<img src="/docs/2_sample_199.png" width="300" hspace="20" /> <img src="/docs/2_sample_299.png" width="300"/> <br>

MNIST - <br>
<img src="/docs/6_sample_199.png" width="200" hspace="100" /> <img src="/docs/6_sample_299.png" width="200"/> <br>

To run the script, simply run `python main.py --dataset DATASET[MNIST/Teapots] --img_dir IMAGE_DIRECTORY --save_dir SAVE_DIRECTORY` <br>
Additional arguments are provided in `main.py` for further customization

