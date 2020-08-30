#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pix2Pix Discriminator network definitions.
"""

import torch.nn as nn


class NLayerDNet(nn.Module):
    """ PatchGAN discriminator network 
        Classifies if an NxN patch of an image is real or fake
    """
   
    def __init__(self, cfg, inp_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        
        super(NLayerDNet, self).__init__()
       
        ndf = cfg.ndf   
        ks = 4 #4x4 filter
        pw = 1 #pad width
        activation = nn.LeakyReLU(0.2, True)
        layers = [nn.Conv2d(inp_nc, ndf, kernel_size=ks, stride=2, padding=pw), activation]
             
        #discriminator layers 
        #inp channel(3)->64->128->256->512 (3 layers for 70x70 PatchGAN)
        #inp channel(3)->64->128 (1 layer for 16x16 PatchGAN)
        #inp channel(3)->64->128->256->512->512->512 (5 layers for 286x286 PatchGAN)
        mult_prev = 1
        for i in range(1, cfg.nlayers_d): #3 for default PatchGAN
          mult = min(2**i, 8)   
          layers += [nn.Conv2d(ndf*mult_prev, ndf*mult, kernel_size=ks, stride=2, padding=pw, bias=use_bias), 
                    norm_layer(ndf*mult), 
                    activation]
          mult_prev = mult
        
        #add additional layers for a n-layer PatchGAN
        mult_prev = mult 
        mult = min(2**cfg.nlayers_d, 8)   
        layers += [nn.Conv2d(ndf*mult_prev, ndf*mult, kernel_size=ks, stride=1, padding=pw, bias=use_bias), 
                    norm_layer(ndf*mult), 
                    activation]
         
        #final layer results in a 1-dim output
        layers +=  [nn.Conv2d(ndf*mult, 1, kernel_size=ks, stride=1, padding=pw)]
        #stack the layers of a PatchGAN
        self.model = nn.Sequential(*layers)
        
    def forward(self, inp):
        
        out = self.model(inp)
        #print("PatchGan out:", inp.shape, out.shape)
        return out
    
class PixelDNet(nn.Module):
    """ PixelGAN discriminator network (==1x1 PatchGAN)
        Classifies if each pixel of an image is real or fake
    """
   
    def __init__(self, cfg, inp_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        
        super(PixelDNet, self).__init__()
       
        
        ndf = cfg.ndf   
        ks = 1 #1x1 spatial filter to convolve over pixels
        pw = 0 #pad width
        activation = nn.LeakyReLU(0.2, True)
        layers = [nn.Conv2d(inp_nc, ndf, kernel_size=ks, stride=1, padding=pw), activation]
        layers += [nn.Conv2d(ndf, ndf*2, kernel_size=ks, stride=1, padding=pw, bias=use_bias), 
                   norm_layer(ndf*2), 
                   activation]
        #final layer results in a 1-dim output
        layers +=  [nn.Conv2d(ndf*2, 1, kernel_size=ks, stride=1, padding=pw, bias=use_bias)]
        #stack the layers of a PixelGAN
        self.model = nn.Sequential(*layers)
        
    def forward(self, inp):
        
        out = self.model(inp)
        #print("PixelGan out", inp.shape, out.shape)
        return out    
    
    
def createDNet(cfg, norm_layer, use_bias):
    """ Create discriminator network 
    
        3 types of discriminator architecture are supported:
            patch ("PatchGAN"): divides an image into subpatches of a certain size (default 70x70). 
                                Each patch is classified as fake or real. Has fewer params than classifying 
                                the whole image.
            pixel ("PixelGAN"): classifies if each pixel (1x1) is fake or real.
    """
    
    net = None
    #default PatchGAN, divides an image into subpatches of a certain size (default 70x70). 
    #Each patch is classified as fake or real by  DNet. The network can have the basic 3 layers 
    #or multiple layers.
    #input and output channels are concatenated because the DNet processes a pair of aligned images
    #that are concatenated along the channel dim.
    if cfg.disc_net == "basic" or cfg.disc_net == "nlayers": 
        net = NLayerDNet(cfg, cfg.inp_nc+cfg.out_nc, norm_layer, use_bias)
    elif cfg.disc_net == "pixel" : #classifies if each pixel is fake or real
        net = PixelDNet(cfg, cfg.inp_nc+cfg.out_nc, norm_layer, use_bias)
    else:
        raise NotImplementedError("unknown discriminator net: %s"%cfg.disc_net)    
    return net     
