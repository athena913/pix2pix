#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model initialization routines
"""

import functools

import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def initNormLayer(norm_type="instance"):
    """ Get the normalization layer
    
        Params: normalization layer : batch | instance | none
        batch norm: tracks the running stats (mean/std dev) and learns affine params
        instance norm: no learnable affine params; running stats are not tracked  
    """    
    
    norm_layer = None
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError("unknown norm layer: %s"%norm_type)        
    return norm_layer    


def initScheduler(optim, cfg):
    """
      Returns a learning rate scheduler.
      optim: optimizer for the model
      cfg: configuration object. 
      cfg.lr_policy specifies the learning rate scheduler: linear|step|plateau|cosine
    """
    scheduler = None
    
    #for linear policy, keep the learn rate constant for the first cfg.nepochs and then decay the rate 
    #linearly to 0 over the next cfg.nepoch_decay epochs. 
    #For the other policies, reuse the native pytorch scheduler
    if cfg.lr_policy == "linear":
        def lambda_rule(epoch):
            lr_lin = 1.0 - max(0, epoch + cfg.ep_cnt - cfg.nepochs)/float(cfg.nepoch_decay +1)
            return lr_lin 
        scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lambda_rule)
    elif cfg.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optim, step_size=cfg.lr_decay_iters, gamma=0.1)
    elif cfg.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif cfg.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.nepochs, eta_min=0)    
    else:
        raise NotImplementedError("unknown learning rate policy: %s"%cfg.lr_policy)
    return scheduler

def initModelWts(model, init_fn, init_gain=0.02):
    """ Initialize the weights according to the specified initializer"""
    
    
    
    def init_func(m):
        cls = m.__class__.__name__
        #initialize the weights of conv or linear layers
        if hasattr(m, "weight") and  cls.find("Conv") != -1 or cls.find("Linear") != -1:
           if  init_fn == "normal":  #default
              nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
           elif  init_fn == "orthogonal":
              nn.init.normal_(m.weight.data, gain=init_gain)   
           elif init_fn == "xav_uni":
              nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
           elif init_fn == "xav_norm":
              nn.init.xavier_normal_(m.weight.data, gain=init_gain)
           elif init_fn == "kai_uni":  
              nn.init.kaiming_uniform_(m.weight, a=0,  mode="fan_in", nonlinearity="relu")             
           elif init_fn == "kai_norm":    
               nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
           else:
               raise NotImplementedError("unknown weight initializer: %s"%init_fn)    
           
           #initialize the bias
           if  hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, val=0.0)
        
        #initialize batchnorm parameters
        elif isinstance(m, nn.BatchNorm2d):# batchnorm's weight is not a matrix
             nn.init.normal_(m.weight.data, 1.0, init_gain)
             nn.init.constant_(m.bias.data, val=0.)       
           
    torch.manual_seed(256)
    model.apply(init_func)      
    return     
