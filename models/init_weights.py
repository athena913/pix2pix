#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Initialize the model weights """

import numpy as np
import torch
import torch.nn as nn

def initKaiUni(module):
    """ Kaiming uniform weight initialization """
    
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
             nn.init.uniform_(m.weight, a=0, b=1)
             nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):    
             nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
             if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
    return

def initKaiNorm(module):
    """ Kaiming uniform weight initialization """
    
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
             nn.init.normal_(m.weight, 0, 0.01)
             nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):    
             nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
             if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
    return

def initXavUni(module):
    """ Xavier uniform weight initialization """
    
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
             nn.init.uniform_(m.weight, a=0, b=1)
             nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):    
             nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
             if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
    return
            

def initXavNorm(module):
    """ Xavier normal weight initialization """
    
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
             nn.init.uniform_(m.weight, 0, 0.01)
             nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):    
             nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
             if m.bias is not None:
               nn.init.constant_(m.bias, val=0.)
    return

def initNormal(module):
    """ Gaussian distribution weight initialization """
    
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
               nn.init.constant_(m.bias.data, val=0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(m.bias.data, val=0.)
        elif isinstance(m, nn.Linear):    
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
               nn.init.constant_(m.bias.data, val=0.)
    return

def initModelWts(module, init_fn):
    """ Initialize the weights according to the specified initializer"""
    
    torch.manual_seed(256)
    if init_fn == "xav_uni":
       initXavUni(module)
    elif init_fn == "xav_norm":
         initXavNorm(module)
    elif init_fn == "kai_uni":        
         initKaiUni(module)
    elif init_fn == "kai_norm":    
         initKaiNorm(module)
    elif init_fn == "normal":    
         initNormal(module)
    return     
