#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions used in Pix2Pix model.
"""
import torch
import torch.nn as nn

class GANLoss(nn.Module):
    
    def __init__(self, gan_mode, real_label=1.0, fake_label=0.0):
        """ Initialize the GANLoss class
        
            gan_mode: type of gan loss 
            
        
        """
    
        super(GANLoss, self).__init__()
        
        #register the labels as buffers. These labels will be saved and restored in the 
        #model dictionary but no gradients will be computed for them.  
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
       
        self.gan_mode = gan_mode
        if gan_mode == "vanilla":  #Goodfellow's paper
           self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan": #least squares
           self.loss = nn.MSELoss()
        elif gan_mode == "wgangp":#wasserstein GAN
           self.loss = None
        else:
            raise NotImplementedError("unknown gan mode %s"%gan_mode)
        
    def getLabelTensor(self, pred, targ_is_real):
         """ Return the real label or fake label as a tensor 
             with the same dimension as the prediction (from the discriminator)
         """ 
         
         if targ_is_real:
             target = self.real_label
         else:
             target = self.fake_label
         target = target.expand_as(pred)
         return target
     
    def __call__(self, pred, targ_is_real):
        """
          Calculate loss for the real or fake predictions from discriminator, wrt the labels. 
        """
        
        if self.gan_mode in ["lsgan", "vanilla"]:
           target = self.getLabelTensor(pred, targ_is_real)
           loss = self.loss(pred, target)
        elif self.gan_mode == "wgangp": 
            if targ_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()
                
        return loss   


def compGradPenalty(dnet, real_data, fake_data, dev, dtype="mixed", const=1.0, lambda_gp=10.0):   
    """
    Returns the gradient penalty loss (ref: WGAN-GP paper)
       dnet: discriminator net
       real_data, fake_data: inputs to Disc net
       dev: device type (gpu/cpu)
       dtype: [real|fake|mixed]
       const: constant in the wgan_gp formula
       lambda_gp: weight for this loss
    """
    
    if lambda_gp > 0.0:
        if dtype == "real":
           data = dnet(real_data)
        elif dtype == "fake":
           data = dnet(fake_data) 
        elif dtype == "mixed":
            #interpolation of real and fake data 
            alpha = torch.rand(real_data.shape[0], 1, device=dev)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement()//real_data.shape[0]).contiguous().view(*real_data.shape)
            data = alpha*real_data + (1-alpha) * fake_data
        else:
            raise NotImplementedError("unknown data type %s"%dtype)    
    
        #requires gradients to be computed 
        data.requires_grad(True)
        disc_out = dnet(data) #discriminator output    
        grads = torch.autograd.grad(outputs=disc_out, inputs=data, 
                                    grad_outputs=torch.ones(disc_out.size()).to(dev),
                                    create_graph=True, retain_graph=True, only_inputs=True)
        
        grads = grads[0].view(real_data.size(0), -1) #flatten the gradients
        #compute weighted mean square 
        gp = (((grads + 1e-16).norm(2, dim=1) - const)**2).mean()*lambda_gp
        return gp, grads #return grad penalty and gradients
    
    else: # if the weight for this loss is 0, ignore this loss
        return 0.0, None
