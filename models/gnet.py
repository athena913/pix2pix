#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import functools

""" Generator and Discriminator networks for Pix2Pix conditional GAN """


class ResnetGen(nn.Module):
    """ Resnet-based Generator network: 
        Contains residual blocks between the downsampling and upsampling layers.
    """
   
    def __init__(self, cfg, inp_nc, out_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        
        super(ResnetGen, self).__init__()
             
        ngf = cfg.ngf
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(inp_nc, ngf, kernel_size=7, padding=0, bias=use_bias), 
                 norm_layer(ngf), 
                 activation]
             
        #downsampling layers
        n_down_sample = 2
        for i in range(n_down_sample):
          mult = 2**i   
          model += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias), 
                    norm_layer(ngf*mult*2), 
                    activation]
          
        #resnet blocks
        mult = 2**n_down_sample
        for i in range(cfg.n_blocks):
           model += [ResnetBlock(ngf*mult, padding_type=cfg.pad_type, norm_layer=norm_layer, use_dropout=cfg.use_dropout, use_bias=use_bias)] 
        
        #upsampling layers - symmetric to downsampling
        for i in range(n_down_sample):
          #skip connection between i-th downsampling layer and (n-i)th upsampling layer  
          mult = 2**(n_down_sample - i)   
          model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 
                                       kernel_size=3, stride=2, 
                                       padding=1, output_padding=1, bias=use_bias), 
                    norm_layer(int(ngf*mult/2)), 
                    activation]
          
        model +=  [nn.ReflectionPad2d(3), 
                   nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0), 
                   nn.Tanh()]
    
        self.model = nn.Sequential(*model)
        
    def forward(self, inp):
        
        out = self.model(inp)
        return out


class ResnetBlock(nn.Module):
    """ Building block for Resnet-based generator """

    def __init__(self, inp_dim, padding_type, norm_layer, use_dropout, use_bias):
        
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(inp_dim, padding_type, norm_layer, use_dropout, use_bias)         
        
    def build_conv_block(self, dim, pad_type, norm_layer, use_dropout, use_bias):    
        """ Resnet  building block """
        
        cblock = [] #conv blocks
        p = 0 #padding
        if pad_type == "reflect":
           cblock += [nn.ReflectionPad2d(1)]
        elif pad_type == "replicate":
           cblock += [nn.ReplicationPad2d(1)] 
        elif  pad_type == "zero": #no padding
            p = 1
        else:
            raise NotImplementedError("unknown padding type %s"%pad_type)  
            

        cblock += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                   norm_layer(dim), nn.ReLU(True)]
        
        if use_dropout: 
           cblock += [nn.Dropout(0.5)] #50% dropout
           
        p = 0
        if pad_type == "reflect":
           cblock += [nn.ReflectionPad2d(1)]
        elif pad_type == "replicate":
           cblock += [nn.ReplicationPad2d(1)] 
        elif  pad_type == "zero": #no padding
            p = 1
        else:
            raise NotImplementedError("unknown padding type %s"%pad_type)  
        
        cblock += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)] 
        
        cblock = nn.Sequential(*cblock)
        return cblock

    def forward(self, inp):
        #add skip connections         
        out = inp + self.conv_block(inp)
        return out

#=================Unet-based Generator ===================
        
class UnetGen(nn.Module):
    """ 
       Defines a Unet-based Generator network
    """
   
    def __init__(self, cfg, inp_nc, out_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
       
        """ recursively construct a Unet generator from innermost to outermost layer
        
            cfg: configuration
            inp_nc: number of channels in input images
            out_nc: number of channels in output images
            norm_layer: normalization layer
        """
        
        super(UnetGen, self).__init__()
        #unet encoder structure: 
        #C64-C128-C256-C512-C512-C512-C512-C512
        #unet decoder structure: 
        #CD512-CD512-CD512-C512-C256-C128-C64  with skip connection becomes
        #CD512-CD1024-CD1024-C1024-C512-C256-C128        
        
        ngf = cfg.ngf
        #innermost layer
        unet_blk = UnetSkipBlock(ngf*8, ngf*8, inp_nc=None, sub_module=None, innermost=True, norm_layer=norm_layer, use_bias=use_bias, use_drop=cfg.use_dropout)
        for i in range(cfg.ndowns-5): #intermediate layers with ngf*8 filters
            unet_blk = UnetSkipBlock(ngf*8, ngf*8, inp_nc=None, sub_module=unet_blk, norm_layer=norm_layer, use_bias=use_bias, use_drop=cfg.use_dropout)
        #reduce the number of filters from ngf*8 to ngf    
        unet_blk = UnetSkipBlock(ngf*4, ngf*8, inp_nc=None, sub_module=unet_blk, norm_layer=norm_layer, use_bias=use_bias)
        unet_blk = UnetSkipBlock(ngf*2, ngf*4, inp_nc=None, sub_module=unet_blk, norm_layer=norm_layer, use_bias=use_bias)
        unet_blk = UnetSkipBlock(ngf, ngf*2, inp_nc=None, sub_module=unet_blk, norm_layer=norm_layer, use_bias=use_bias)
        #outer most layer
        self.model = UnetSkipBlock(out_nc, ngf, inp_nc=inp_nc, sub_module=unet_blk, outermost=True, norm_layer=norm_layer, use_bias=use_bias)
        
    def forward(self, inp):
        
        out = self.model(inp)
        return out

class UnetSkipBlock(nn.Module):
    """ Defines a Unet submodule 
        |downsampling---submodule-----upsampling|
    """ 
        
    def __init__(self, outer_nc, inner_nc, inp_nc=None, sub_module=None, 
                 outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_bias= False, use_drop=False):
        """
          Defines a unet submodule with skip connection
          outer_nc: number of filters in the outer conv layer
          inner_nc: number of filters in the inner conv layer         
          inp_nc: number of channels in the input images/features
          sub_module: unet sub modules
          outermost: True if this block is the outermost block, else False
          innermost: True if this block is the innermost block, else False
          norm_layer: normlization layer
          use_bias: whether to use bias
          use_drop: use dropout or not
        """
        
        super(UnetSkipBlock, self).__init__()        
    
      
        if inp_nc is None:
           inp_nc = outer_nc
        self.outermost = outermost
        
        #downsampling layers in encoder have LeakyReLU   
        down_act = nn.LeakyReLU(0.2, True)         
        down_conv = nn.Conv2d(inp_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias) 
        down_norm = norm_layer(inner_nc) 
                      
        #upsampling layers in decoder have ReLU   
        up_act = nn.ReLU(True)        
        up_norm = norm_layer(outer_nc) 
        
        if outermost:
           up_conv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1) 
           up = [up_act, up_conv,  nn.Tanh()] #final decoder block has Tanh activation
           down = [down_conv] #no batch norm for outermost encoder block
           model = down + [sub_module] + up
        elif innermost:
           up_conv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias) 
           up = [up_act, up_conv,  up_norm] 
           down = [down_act, down_conv] #no batch norm for innermost  block
           model = down +  up
           
        else:#intermediate layers
            up_conv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias) 
            up = [up_act, up_conv, up_norm] 
            down = [down_act, down_conv, down_norm]
           
            if use_drop: 
               model = down + [sub_module] + up + [nn.Dropout(0.5)] #50% dropout
            else: 
               model = down + [sub_module] + up #no dropout
               
        #stacked layers    
        self.model = nn.Sequential(*model)     
        
           
    def forward(self, inp):
        if self.outermost == False:
            #add skip connection         
            out = torch.cat([inp, self.model(inp)], 1)
        else:
            out = self.model(inp)
        return out     
           
           
#==============creator functions ==================        
def createGNet(cfg, norm_layer, use_bias):
    """ Create generator network 
    
        Generator network is based on unet or resnet architecture:
            Unet: unet_128 uses 128x128 image input
                  unet_256 uses 256x256 image input
            Resnet: consists of resnet blocks in between the downsampling and upsampling layers
                    resnet_6 uses 6 resnet blocks between downsampling and upsampling layers
                    resnet_9 uses 9 resnet blocks between downsampling and upsampling layers
    
    """
    
    net = None
   
    if cfg.gen_net.startswith("resnet"):
        net = ResnetGen(cfg, cfg.inp_nc, cfg.out_nc, norm_layer, use_bias)
    elif cfg.gen_net.startswith("unet"):
        net = UnetGen(cfg, cfg.inp_nc, cfg.out_nc, norm_layer, use_bias)
    else:
        raise NotImplementedError("unknown generator net: %s"%cfg.gen_net)
      
    return net
