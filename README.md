Pix2Pix Conditional GAN Model in Pytorch
========================================

This code is an implementation of the Pix2Pix model in Pytorch.
It is based on the original code at 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
but unlike the original version which includes CycleGAN, 
this code only implements Pix2Pix.

Training:

1) config.py sets the data path and paths for saving the models and logs.
   It also has the training configuration that I used. 
   Please change the paths and configuration to suit your needs.

2) In main.py, set the mode to "train"

3) Run "python main.py"


Testing:

1) In main.py, set the mode to "test"

2) Run "python main.py"
 
