import os
import sys
import glob

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils


""" Defines  dataloader  and dataset for img2img translation using conditional GAN """
 
class ImgPairDataset():

   def __init__(self, conf, data_list, trans):
        
        self.conf = conf
        self.vis_trans = trans #visual transformations

        #loads the data list
        self.data_list = data_list
       
  
   def __len__(self):
       
       return min(len(self.data_list), self.conf.max_dataset_size)

   def __getitem__(self, ind):
       
       img_path = self.data_list[ind]
      
       x, y = self.processAlignedImg(img_path)
       if y is not None:
         sample = {"A": x, "B": y, "img_path": img_path}
       else: #single, unpaired image
         sample = {"A": x, "img_path": img_path} 
       return sample

   def processAlignedImg(self, img_path):
       
       #the left half of image contains the source image
       #the right half of image contains the target image
       
       img = Image.open(img_path).convert("RGB")
       if self.conf.dataset_mode == "aligned":
           w, h = img.size
           mid = int(w/2)
           x = img.crop((0, 0, mid, h))
           y = img.crop((mid, 0, w, h))
       else: #single image
           x = img
           y = None
           
       #transorm input images    
       if self.vis_trans is not None:
          x = self.vis_trans(x)
          if y is not None:
            y = self.vis_trans(y)
       return x, y
    
def getTransforms(cfg, mode):
    
        trans = []    
        if cfg.gray:
            trans += [transforms.Grayscale(1)]
            
        if mode == "train":  
            trans += [transforms.Resize((cfg.img_size, cfg.img_size), Image.BICUBIC)]
            if cfg.crop:
                trans += [transforms.RandomResizedCrop(cfg.crop_size)]
            if cfg.flip:
                trans += [transforms.RandomHorizontalFlip()]
        
        trans +=  [transforms.ToTensor()]
        if cfg.gray:
           trans += [transforms.Normalize(mean=(0.5, ), std=(0.5, ))]
        else:    
           trans += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
          
            
        data_transforms = transforms.Compose(trans)
        
         
        return data_transforms


def getDataList(conf):
    """ Get the files in the data path """
       
    dlist = []
    #mixed training datasets
    if conf.dataset == "edges_shoe_bags" and conf.mode == "train":
        dpath = [os.path.join(conf.data_root, "edges2handbags", conf.mode), 
                     os.path.join(conf.data_root, "edges2shoes", conf.mode)] 
        for d in dpath:
            if os.path.exists(d):
              print(d)  
              #get atmost 50K  from each dataset 
              dlist.extend(sorted(glob.glob(d + "/*"))[:50000])
    else:
      dpath = os.path.join(conf.data_root, conf.dataset, conf.mode) 
      print(dpath)
      if os.path.exists(dpath):
          dlist =  sorted(glob.glob(dpath + "/*"))
    print(len(dlist), dlist[0])    
    return dlist   
  
def getDataLoader(conf, mode):
    """Get dataloaders for different modes"""
 
    torch.manual_seed(144) 
    print("Loading %s data from %s"%(mode, conf.data_root))
    #get image transforms for the mode
    trans = getTransforms(conf, mode)
    #get the list of  files
    data_list = getDataList(conf)
    #get the dataset per mode
    data = ImgPairDataset(conf, data_list, trans)
    do_shuffle = True if mode == "train" else False #dont shuffle test  data
    dl = DataLoader(data, batch_size=conf.batch_size, shuffle= do_shuffle, num_workers=conf.nworkers)
               
    return dl
