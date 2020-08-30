import sys
import os
import logging
import time
import matplotlib.pyplot as plt


sys.path.append("data_utils")
sys.path.append("models")
sys.path.append("../..")
import config
import data_loader as dl
import log_utils 
import pix2pix


def printConfig(conf):

    conf = conf.__dict__
    mdl_root = conf["model_root"]

    logging.info("===========Starting at time: {}======".format(time.ctime()))
    for k, v in conf.items():
        logging.info("{} = {}".format(k, v))

    print("===========Starting at time: {}======".format(time.ctime()))
    print("Model dir = %s"%(mdl_root))
    print("Log file = %s"%(conf["log_file"]))
    return

def testDataLoad(dl):
    """ simple test to ensure data loader is correct """
    
    for k  in dl.keys():
      tr = dl[k]
      print("===Length of %s dataset:%d==="%(k, len(tr.dataset)))

      for sample in tr:
           a = sample["A"]
           plt.imshow(a[0].permute(1,2,0))
           plt.show() 
           if "B" in sample.keys():             
               b = sample["B"]           
               print(a.size(), b.size(), sample["img_path"])          
               plt.imshow(b[0].permute(1,2,0))
               plt.show()
           break
    return       



if __name__ == "__main__":

   #mode = "train"
   mode = "test"

   cfg = config.Config(mode) if mode == "train" else config.ConfigTest(mode)
   #logger = log_utils.Logger(model_name="pix2pix_logs", data_name="edges2handbags", log_path=cfg.log_path)
   log_name = "test_logs_ep%d_shoe_bags"%(cfg.ep_cnt) if mode == "test" else "train_logs"
   logger = log_utils.Logger(cfg, log_path=os.path.join(cfg.log_path, log_name), model_name="pix2pix")
   if mode == "train":
       if not os.path.exists(cfg.model_root):
          os.makedirs(cfg.model_root)  
       if not os.path.exists(cfg.log_path):   
          os.makedirs(cfg.log_path)
      
   log_utils.setLogger(cfg.log_file)
   print("Logging file", cfg.log_file)
   printConfig(cfg)

      
   #load data 
   dloader = {}
   if mode == "train":
     dloader["train"] = dl.getDataLoader(cfg, "train")
     val_cfg = config.ConfigTest("test")
     dloader["test"] = dl.getDataLoader(val_cfg, "test")
   else: #test mode
     dloader["test"] = dl.getDataLoader(cfg, "test") 
   #testDataLoad(dloader)
   #print(cfg.load_saved_D, cfg.load_saved_G)
   #exit(0)
   
   # Train
   pix_gan = pix2pix.Pix2Pix(cfg, logger)
   if mode == "train":
      pix_gan.trainModel(dloader)
   else:
      pix_gan.testModel(dloader["test"], cfg.ep_cnt) 
      
   print("Logging file = %s, Saved models = %s"%(cfg.log_file, cfg.model_root))

