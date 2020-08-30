#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Configuration 
"""

import os


class Config():
    def __init__(self, mode="train"):
        
        self.mode = mode #train/test 
        self.resume = False #set to True to resume training 
        self.gpu_list = None #gpu ids 
       
        
        self.configData()
        self.configModel()
        
        #if self.mode == "train":
        self.configTrain()
        self.configLogs()
        
    def configLogs(self):
        """====== Model save and load paths====="""
        
        self.model_save_freq = 1 #save model once every these many epochs
        self.log_scalar_freq = 500   #log frequency for scalar values like loss
        self.val_freq = 1000   #log frequency for images
        
        #paths for logging results and model checkpoints 
        base_path = "./gan_models/pix2pix_%s/"%self.dataset
        suffix = os.path.join(self.gen_net + "_" + self.disc_net, self.gan_mode + "_" + self.lr_policy + "_L1_50") #+"_5layers")
        self.model_root = os.path.join(base_path, "saved_models", suffix, "checkpoints")
        self.log_path = os.path.join(base_path, "logs", suffix)
        self.log_file = os.path.join(self.log_path, "%s.log"%self.mode)
        
        #pretrained paths
        if self.resume:
            self.ep_cnt = 0
            base_path = "./gan_models/pix2pix_%s/"%self.dataset
            suffix = os.path.join(self.gen_net + "_" + self.disc_net, self.gan_mode + "_" + self.lr_policy+"_L1_10")
            self.load_saved_G = os.path.join(base_path, "saved_models", suffix, "checkpoints/pix2pix_G_ep_%d.pth"%(self.ep_cnt))
            self.load_saved_D = os.path.join(base_path, "saved_models", suffix, "checkpoints/pix2pix_D_ep_%d.pth"%(self.ep_cnt))
            
        return
    
    
    def configData(self):
        """===============Data-related configuration============"""
         
        self.nworkers = 8
        self.img_size = 256 #286 #image size
        self.crop_size = 256  #crop image to this size
        self.max_dataset_size = float("inf") #used in case of smaller training subset
        #img transforms
        self.gray = False #for colorization, this will differ for input and output
        self.crop = False  #crop to crop_size
        self.flip = False #randomly flip images
       
        #data paths
        self.dataset = "edges2handbags" #["edges_shoe_bags"|"edges2handbags"|"edges2shoes"|"facades"]
        self.data_root = "/data_public/image/" #change to your data path
        self.dataset_mode = "aligned"
        self.direction = "AB"
        #TBD: set the input and output channels based on direction
        self.inp_nc = 3    #number of input channels [3: RGB, 1: grayscale]
        self.out_nc = 3   #number of input channels [3: RGB, 1: grayscale]
        return
        
    def configModel(self):            
        """===============Configure model parameters============"""
        
       
        #configuration used in Generator         
        
        self.gen_net = "unet_256"  #Gnet arch: [resnet_6, resnet_9, unet_128, unet_256]
        self.n_blocks = int(self.gen_net.split("_")[1]) #number of resnet blocks
        #number of downsampling layers in Unet. This is the number of layers it takes to
        #reduce the input image size to 1x1. For img_size=256, ndowns = 8; img_size=128, ndowns=7. 
        self.ndowns = 8 if self.n_blocks==256 else 7  
        self.ngf = 64 #number of filters in GNet
        self.pad_type = "reflect"
        
        #configuration used in discriminator
        self.disc_net = "basic"  #Dnet arch: [basic, nlayers, pixel]
        self.ndf = 64 #number of filters in Dnet
        if self.disc_net == "basic":
           self.nlayers_d = 3 #default PatchGAN has 3 layers, used if disc_net = nlayers
        else:
           self.n_layers_d = 5
        return
    
    def configTrain(self):           
        """===============Configure model training parameters ============""" 
       
        
        self.batch_size = 4  #mini batch size for training 
        #specifies the number of epochs with initial learning rate
        self.nepochs = 100 
        self.nepoch_decay = 100  #100 number of epochs over which learn rate has to be decayed to 0 for linear policy
        self.ep_cnt = 1
        
        #losses to be computed:
        #for generator: GAN loss + L! loss
        #for discriminator: binary x-entropy loss
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        
        #weight initialization: [normal|kaiming|orthogonal|xavier]
        self.init_type = "normal" 
        self.init_gain = 0.02 #scaling factor used for xavier, orthogonal, normal init
        #type of normalization layer: #[batch|instance]
        self.norm_layer = "batch"        
        self.use_dropout = False      
        
        
        #learning rate policy: [linear|cosine|plateau|step]
        self.lr_policy = "linear"
        self.lr = 0.0002 #initial learning rate for Adam
        self.beta1  = 0.5 #momentum for Adam optimizer
        self.lr_decay_iters = 50 #multiply by a threshold every lr_decay_iters iterations

        #training objective: GAN_loss (Goodfellow's paper) + lambda_L1*||G(A)-B||_1
        self.gan_mode = "vanilla"  #type of gan loss: [vanilla|lsgan|wgangp]
        self.lambda_L1 = 50.0 #paper:100.0 #weight for L1 loss between pixels of generated image and target image
        self.pool_size = 0 #no buffering for images
        
        return
    
    
class ConfigTest():    
    def __init__(self, mode="test"):
        """===============Configure model testing parameters ============""" 
        
        self.mode = mode #train/test 
        self.gpu_list = None #gpu ids 
         
        self.nworkers = 8
        self.dataset = "edges_shoe_bags" #["edges2handbags"|"edges2shoes"|"facades"]
        self.data_root = "/data_public/image/"
        self.dataset_mode = "aligned"
        self.gray = False
        self.crop = False #avoid cropping for test images
        self.img_size = 256
        self.batch_size = 6
        self.dataset_mode = "aligned"
        self.direction = "AB"
        #TBD: set the input and output channels based on direction
        self.inp_nc = 3    #number of input channels [3: RGB, 1: grayscale]
        self.out_nc = 3   #number of input channels [3: RGB, 1: grayscale]
        self.max_dataset_size = float("inf") #used in case of smaller training subset
        
      
        
        self.norm_layer = "batch"
        self.use_dropout = False
        
        #to load the model
        self.gen_net = "unet_256" #used for logging comment
        self.n_blocks = int(self.gen_net.split("_")[1]) #number of resnet blocks
        #number of downsampling layers in Unet. This is the number of layers it takes to
        #reduce the input image size to 1x1. For img_size=256, ndowns = 8; img_size=128, ndowns=7. 
        self.ndowns = 8 if self.n_blocks==256 else 7  
        self.ngf = 64 #number of filters in GNet
        self.pad_type = "reflect"
        
        #configuration used in discriminator
        self.disc_net = "pixel"  #Dnet arch: [basic, nlayers, pixel]
        self.ndf = 64 #number of filters in Dnet
        self.nlayers_d = 3 #default PatchGAN has 3 layers, used if disc_net = nlayers
        
        #pretrained paths
        self.ep_cnt = 21
        #suffix = "{}_{}/vanilla_linear/".format(self.gen_net, self.disc_net)
        #suffix = "unet_256_pixel/vanilla_linear_L1_100/"
        #suffix = "unet_256_pixel/vanilla_linear/"
        suffix = "unet_256_basic/vanilla_linear_val/"
        #suffix = "unet_256_basic/vanilla_linear/"
        base_model_path = "./gan_models/pix2pix_edges2handbags/"
        self.model_root = os.path.join(base_model_path, "saved_models", suffix, "checkpoints")
        self.load_saved_G = os.path.join(self.model_root, "pix2pix_G_ep_%d.pth"%(self.ep_cnt))
        self.load_saved_D = os.path.join(self.model_root, "pix2pix_D_ep_%d.pth"%(self.ep_cnt))
        self.log_path = os.path.join(base_model_path, "logs", suffix)
        self.log_file = os.path.join(self.log_path, "%s.log"%self.mode)
        
    
        return 
