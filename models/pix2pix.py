import sys
import time
import logging

import functools 
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import gnet
import dnet 
import init_model as im
import losses

""" Pix2Pix model """



class Pix2Pix():

    def __init__(self, cfg, logger):

        torch.manual_seed(144)
        
        self.cfg = cfg  #model configuration
        self.logger = logger #Tensorboard logging
        
        self.metric = 0   
        norm_layer = im.initNormLayer(cfg.norm_layer)
        #bias is not needed in case of BatchNorm which has affine params 
        if type(norm_layer) == functools.partial:
           use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
           use_bias = norm_layer == nn.InstanceNorm2d
           
        print("creating Gnet")
        self.gnet = gnet.createGNet(cfg, norm_layer, use_bias)
        print("initializing Gnet model")
        self.gnet  = self.initModel(self.gnet, ["G"])
       
        if cfg.mode == "train":
           
           print("creating Dnet")
           self.dnet = dnet.createDNet(cfg, norm_layer, use_bias)
           print("initializing Dnet model")
           self.dnet = self.initModel(self.dnet, ["D"])

           self.initTrain()
        

    def initModel(self, model, mnames):
        """ 
            Initialize the model
            - port to model to the cpu/gpu device 
            - initialize the model weights 
        """
        
        #port to the cpu/gpu device
        self.use_cuda = torch.cuda.is_available() # check if GPU exists
        self.device = torch.device("cuda" if self.use_cuda else "cpu") # use GPU or CPU
        model.to(self.device)
        
        self.gpu_list = self.cfg.gpu_list #gpu ids 
        if self.use_cuda:
            if self.gpu_list is None: 
               self.gpu_list = list(range(torch.cuda.device_count())) 
            if len(self.gpu_list) > 0: #use multi-gpus
               print("Using gpus:{}".format(self.gpu_list))
               model = torch.nn.DataParallel(model, device_ids=self.gpu_list)
            else:
               print("Using %d gpus"%len(self.gpu_list))  
       
        #load pretrained weights
        if self.cfg.mode == "test":
            self.loadWeights(mnames)
        elif self.cfg.resume: #resume training from a previous checkpoint
            print("Resuming training")
            logging.info("Resuming training")
            self.loadWeights(mnames)    
        else:
            #initialize the model weights for training   
           im.initModelWts(model, self.cfg.init_type, self.cfg.init_gain)     
        return model      
           
    def initTrain(self):
        self.optimizers = []
        if self.cfg.mode == "train":
            self.loss_gan = losses.GANLoss(self.cfg.gan_mode).to(self.device)
            self.loss_L1 = torch.nn.L1Loss()
            
            #initialize the optimizers. 
            self.dopt = optim.Adam(self.dnet.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))
            self.gopt = optim.Adam(self.gnet.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))
            self.optimizers.append(self.gopt)
            self.optimizers.append(self.dopt)
            #configure the learning rate schedulers for the optimizers  for gnet and dnet
            self.schedulers = [im.initScheduler(optim, self.cfg) for optim in self.optimizers]
           
        return
    
    def loadWeights(self, nets):
        """ Load pretrained networks """
       
        for n in nets:
            if n == "G":
               net = self.gnet 
               model_wts = self.cfg.load_saved_G
            elif n == "D":
               net = self.dnet 
               model_wts = self.cfg.load_saved_D   
                        
#            if isinstance(net, torch.nn.DataParallel):
#                net = net.module
            print("Loading model weights from %s"%model_wts)  
                      
            #first deserialize the state dictionary by calling torch.load()  
            sd = torch.load(model_wts, map_location=self.device)
            #CPU or single GPU 
            if not isinstance(net, torch.nn.DataParallel):    
                #sd = torch.load(model_wts, map_location='cpu')
                new_sd = {}
                #trained model uses DataParallel. 
                #so remove "module" from the model parameters when machine has only cpu.
                for k, v in sd.items():
                        import re
                        #remove "module" in state dictionary keys
                        k = re.sub("module\.", "", k)
                        #print(k, v)
                        new_sd[k] = v
                net.load_state_dict(new_sd, strict=True)          
            else:            
               net.load_state_dict(sd, strict=True) 
            
#            state_dict = torch.load(model_wts, map_location=self.device)
#            if hasattr(state_dict, '_metadata'):
#                    del state_dict._metadata
#            net.load_state_dict(state_dict, strict=True)            
            

        return
        
        
    def updateLearnRate(self):
        """
        Update learning rate for all networks.
        """
        
        for scheduler in self.schedulers:
            if self.cfg.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:    
                scheduler.step()
       
        for o in self.optimizers:         
          lr = o.param_groups[0]["lr"]
          logging.info("updated learn rate=%f"%lr)         
        return 
    
    
    def getInput(self, batch):
      """ Generate  input from batches"""

      
      #aligned images has image pairs
      if self.cfg.dataset_mode == "aligned":
          ab = self.cfg.direction == "AB"
          
          realA = batch["A" if ab else "B"].to(self.device)
          realB = batch["B" if ab else "A"].to(self.device)
      else: #unaligned, single image
          realA = batch["A"].to(self.device)
          realB = None
      img_paths = batch["img_path"]
      
      return realA, realB, img_paths
  
    def setReqGrads(self, nets, req_grads):
        """ Set if gradients are required for the networks """
        
        if not isinstance(nets, list):
            nets = [nets]
        for n in nets: 
            if n is not None:
               for param in n.parameters():
                   param.requires_grad = req_grads
        return           
        
    
    
    def backwardD(self):
        """ calculate GAN loss for discriminator """
      
        #reset performance logs at the start of each epoch
        batch_perf = {}
        #train dnet on fake data 
        pair = torch.cat((self.realA, self.fakeB), 1)
        #train Dnet on fake data. detach to avoid backprop to G.
        pred_fake = self.dnet(pair.detach())
        loss_D_fake = self.loss_gan(pred_fake, targ_is_real=False)
        
        #Train dnet on real data
        pair = torch.cat((self.realA, self.realB), 1)
        pred_real = self.dnet(pair)
        loss_D_real = self.loss_gan(pred_real, targ_is_real=True)
        #combine loss and backprop
        loss_D = (loss_D_real + loss_D_fake)*0.5
        
        #log performance for display
        batch_perf["loss_D_fake"] = loss_D_fake.item()
        batch_perf["loss_D_real"] = loss_D_real.item()
        batch_perf["loss_D"] = loss_D.item()
        batch_perf["pred_fake"] = pred_fake
        batch_perf["pred_real"] = pred_real
        
        loss_D.backward()
                
#        print("backward_D: Fake_pred={}, Real_pred={}, Fake_loss={}, Real_loss={}, Total_loss={}".format(self.pred_fake, 
#                self.pred_real, self.loss_D_fake, self.loss_D_real, self.loss_D))
        return batch_perf
    
    def backwardG(self):
        """ calculate GAN loss + L1 loss for the generator """
        
        #reset performance logs at the start of each epoch
        batch_perf = {}
        
        pair = torch.cat((self.realA, self.fakeB), 1)
        #train Dnet on fake data
        pred_fake = self.dnet(pair)
        loss_G_gan = self.loss_gan(pred_fake, targ_is_real=True)
        loss_G_L1 = self.loss_L1(self.fakeB, self.realB) * self.cfg.lambda_L1
        #combine loss and backprop
        loss_G = loss_G_gan + loss_G_L1
        
        #log performance for display
        batch_perf["loss_G_gan"] = loss_G_gan.item()
        batch_perf["loss_G_l1"] = loss_G_L1.item()
        batch_perf["loss_G"] = loss_G.item()
        
        loss_G.backward()
#        print("backward_G: Fake_pred={}, gan_loss={}, L1_loss={}, Total_loss={}".format(pred_fake, 
#                self.loss_gan_G, self.loss_L1_G, self.loss_G))
        return batch_perf

    def forward(self):
        """ forward pass: used during training and testing """
        
        fake = self.gnet(self.realA)
        return fake 
    

    def trainBatch(self):    
        #Train discriminator and generator networks
       
        #self.gnet.train()
        #self.dnet.train()
            
        #generate fake images 
        self.fakeB =  self.gnet(self.realA) #forward pass G(A)
        #print("Fake data", self.fakeB.shape)
        #update D
        self.setReqGrads(self.dnet, True)  #enable backprop for D
        #reset grads
        self.dopt.zero_grad() #set D's grads to 0
        batch_D = self.backwardD()   #calculate grads for D
        self.dopt.step() #update weights for D
        
        #update G
        self.setReqGrads(self.dnet, False)  #D does not require grads when optimizing G
        #reset grads
        self.gopt.zero_grad() #set G's grads to 0
        batch_G = self.backwardG()      #calculate grads for G
        self.gopt.step()      #update weights for G
        
        #merge D and G performance for logging/display
        batch_D.update(batch_G)

        return batch_D

    

    def trainModel(self, data_loader):

        num_batch = len(data_loader["train"])
       
        
        start_time = time.time()
        print("{}: Start model training".format(start_time) )
        logging.info("{}: Start model training".format(start_time))

        #train for N epochs
        nepochs = self.cfg.nepochs  #+ self.cfg.nepoch_decay + 1
        for epoch in range(nepochs):
        
           epoch_perf = {}
           #train over the whole dataset 
           for n_batch, real_batch in enumerate(data_loader["train"]):
               #to avoid duplicate logging at the end of an epoch
               logged_scalar = False 
               logged_img = False 
               #get paired data for training the translation model A->B
               self.realA, self.realB, self.img_paths = self.getInput(real_batch)
               #print("(real batch, real_data):", self.realA.shape, self.realB.shape)
               batch_perf = self.trainBatch()
               #update cumulative performance. This is done after validation, so psnr can be recorded.
               epoch_perf = updateEpochPerf(epoch_perf, batch_perf)
               
               #log train performance periodically, after updating epoch performance. 
               if (n_batch % self.cfg.log_scalar_freq) == 0:
                   logged_scalar = True 
                   self.logger.log_train(batch_perf, epoch, nepochs, n_batch, num_batch, "batch")                 
                   #self.logger.display_status(epoch, nepochs, n_batch, num_batch,
                   #        batch_perf["loss_D"], batch_perf["loss_G"], batch_perf["pred_real"], batch_perf["pred_fake"]) 
                                      
               #validate generated images periodically    
               if (n_batch % self.cfg.val_freq) == 0:
                   logged_img = True
                   test_imgs = {}
                   test_imgs["realA"] = self.realA.data.cpu()
                   test_imgs["realB"] = self.realB.data.cpu()
                   test_imgs["fakeB"] = self.fakeB.data.cpu()
                   self.logger.log_cgan_images(test_imgs, epoch, n_batch, num_batch, "train")
                   #validate periodically
                   self.testModel(data_loader["test"], epoch, n_batch, num_batch)
                   # save model periodically  
                   self.logger.save_models(self.gnet, self.dnet, epoch)
               
           #end of an epoch         
           #save model checkpoints periodically        
           if epoch % self.cfg.model_save_freq==0: 
              if logged_scalar == False: 
                  #performance of last batch in the epoch 
                  self.logger.log_train(batch_perf, epoch, nepochs,  n_batch, num_batch, "batch")
              if logged_img == False:    
                   self.testModel(data_loader["test"], epoch, n_batch, num_batch)
              #cumulative performance for entire epoch
              self.logger.log_train(epoch_perf, epoch, nepochs, n_batch, num_batch, "epoch") 
             
             
              #checkpoint after each epoch
              self.logger.save_models(self.gnet, self.dnet, epoch)
#              torch.save(self.gnet.state_dict(), "%s/pix2pix_G_ep_%d.pth"%(self.cfg.model_root, epoch))
#              torch.save(self.dnet.state_dict(), "%s/pix2pix_D_ep_%d.pth"%(self.cfg.model_root, epoch))
              
           #update learning rate after every epoch   
           self.updateLearnRate()  
           logging.info("==========End of epoch {}=========".format(epoch)) 
           
        time_elapsed = (time.time() - start_time)/60.0 
        logging.info("Training completed in {} minutes".format(time_elapsed)) 

    def testModel(self, data_loader, ep, n_batch=None, num_batch=None):
        """ Evaluate the model using test data """
        
        from math import log10
        
        
        mse_loss = nn.MSELoss().to(self.device)
        avg_psnr = 0.0
        
        num_test_batch = len(data_loader)
      
        #print("Testing data length = %d"%num_test_batch)
        #self.gnet.load_state_dict(torch.load(self.cfg.load_saved_G), strict=True)
        #put the model in eval mode, affects batchnorm and dropout layers.
        self.gnet.eval()
        test_imgs = {}
        for n_test_batch, real_batch in enumerate(data_loader):
            #get paired data for training the translation model A->B
            realA, realB, img_paths = self.getInput(real_batch)
            #no backprop or gradient computation. speeds up computation.
            with torch.no_grad(): 
                   fakeB = self.gnet(realA)
                   mse = mse_loss(fakeB, realB)
                   avg_psnr += 10 * log10(1.0/mse.item())
                   
                   #display generated images  
                   #if n_test_batch == num_test_batch/2: #first batch of images
                   test_imgs["realA"] = realA.data.cpu()
                   test_imgs["realB"] = realB.data.cpu()
                   test_imgs["fakeB"] = fakeB.data.cpu()
#                  
                   #print("(realA, realB, fakeB):", n_batch, num_batch, realA.shape, realB.shape, fakeB.shape)
                   tag = "test_bag" if n_test_batch < 2 else "test_shoes" #split display images into shoes and bags
                   if n_batch is not None: #validation
                      self.logger.log_cgan_images(test_imgs, ep, n_batch, num_batch, tag)
                   else:
                      self.logger.log_cgan_images(test_imgs, ep, n_test_batch, num_test_batch, tag) 
                   
        avg_psnr = avg_psnr/num_test_batch   
        if n_batch is None: #for test-only mode, use test-batch number
           n_batch = n_test_batch
           num_batch = num_test_batch

        
        self.logger.log(avg_psnr, ep, n_batch, num_batch, "test_psnr")
       
        return avg_psnr
    
def updateEpochPerf(dest, src):
    """ update epoch performance based on the latest batch performance """

    for k, v in src.items():
          if type(v) == torch.Tensor: #for predictions, use mean
          #if k.startswith("pred"): #mean of the predictions              
              v = v.data.mean().item() #cpu().numpy() #.cpu().squeeze().numpy()
#             x = v.data.mean().cpu().numpy() 
#          else: #if isinstance(v, torch.autograd.Variable):
#             x = v.data.cpu().numpy()
          if k in dest.keys(): #accumulate   
             dest[k] += v
          else: #new entry
              dest[k] = v         
    return dest    