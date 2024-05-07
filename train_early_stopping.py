# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:42:36 2023

@author: tc922
idea: try bigger patch_size,
try add the bottom layers after ViT with different patch_size and block layers combination
try add headers with 16 patch size
try model list: ViT patch8/16/32, repvit_m1.dist_in1k, efficientvit, fastvit,maxvit_base_tf_512.in1k,tiny_vit_21m_512.dist_in22k_ft_in1k
label smoothing
"""

import timm
import sys
import os
sys.path.append('./')
#print(os.listdir('/content/drive/MyDrive/ViT-repadapter-1'))
from model_manipulation import *
from repadapter import set_RepAdapter, set_RepWeight
from torch.optim import AdamW
#from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from timm.data import Mixup
from timm.loss import  SoftTargetCrossEntropy
from ImageLoader import get_split
from torchsummary import summary
from result_presentation import *
import time
import pdb
from model_manipulation import callback
import gc


class args_train:
  """
  This args class is for store training information
  """
  def __init__(self,seed = 32,
               model_path = './ViT-B_16.npz',crop_model= False, layer_num = 12,
               trainable_layers= [''],
               root_path = './',rebuild_headers= False, insert_adapter= False,
               mode = 'train',dataset_name = 'COVID_balanced', 
               dataset = 'COVID-19_Radiography_Dataset',
               class_num = 4,epoch = 200,
               verbose= True, early_stopping_patience = 30,
               lr = 2*(1e-6),wd = 0.02,batch_size= 16,
               proc_method = 'Normal',
               method = 'vit_base_patch8_r224_label_smoothing',    #stored folder name
               model_name = 'vit_base_patch16_224',
               img_size = (224,224)):
    self.seed = seed
    self.layer_num = layer_num
    self.lr = lr
    self.wd = wd
    self.model = model_name
    self.model_path = model_path
    self.root_path = root_path
    self.method = method    #this will decide the model/logs will be stored under which folder
    self.mode = mode
    self.class_num = class_num
    self.dataset = dataset
    self.dataset_name = dataset_name
    self.batch_size = batch_size
    self.verbose = verbose
    self.trainable_layers= trainable_layers
    self.rebuild_headers = rebuild_headers
    self.insert_adapter = insert_adapter
    self.crop_model = crop_model
    self.img_size = img_size
    self.early_stopping_patience = early_stopping_patience
    self.epoch = epoch
    self.proc_method = proc_method
    
def train(config, model, dl, opt, scheduler, test_dl, epoch,
          criterion=nn.CrossEntropyLoss(),execution_name= '',callback= None):
    start_time = time.time() 
    model.train()
    model = model.cuda()
    start_time = time.time()
    best_time = 0   #record the time used to reach  the best validaiton accuracy epoch
    test_loss_function = nn.CrossEntropyLoss()  #loss function for calcualte the validaiton loss
    for ep in tqdm(range(epoch)):
        epoch_offset = 0
        model = model.cuda()
        acc_train = 0
        loss_train = 0
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            #pdb.set_trace()
            out = model(x)
            acc_train += train_acc(out,y)
            loss = criterion(out, y)
            loss_train += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        #pdb.set_trace()
        acc_train = float(acc_train/(i+1))  #calculate the train accuracy
        loss_train = float(loss_train/(i+1))
        acc, loss_valid = test(model, test_dl,validation_loss_function = test_loss_function)
        model_dir = os.path.join(args.root_path, 'models')
        log_dir = os.path.join(args.root_path, 'logs')
        if acc > config['best_acc']:
            config['best_acc'] = acc
            best_time = time.time() - start_time
            config['best_time'] = best_time
            save_model(model_dir,log_dir,config['method'], config['name'] +
                execution_name, model, acc, ep + epoch_offset,
                acc_train = acc_train,loss_train=loss_train, loss_valid = loss_valid)
        if ep % 10 == 0:
            save_model(model_dir,log_dir,config['method'], config['name']+'_usual_'
             + execution_name + str(ep+epoch_offset), model, acc, ep + epoch_offset
             )
        save_model(model_dir,log_dir,config['method'], config['name']+'_usual_'
         + execution_name + str(ep+epoch_offset), model, acc, ep + epoch_offset,
         model_save=False,acc_train = acc_train,loss_train=loss_train, loss_valid = loss_valid)
        if callback != None:
            early_stop_state = callback.early_stopping(acc)
            if early_stop_state == True: # the no-improvement continous epochs exceed the maximum
                print('early stopping!')
                best_e, best_a = callback.get_best_value()
                print('best epoch: %d, best accuracy: %.3f'%(best_e,best_a))
                model = model.cpu()
                overall_time = time.time() - start_time
                return model, overall_time, best_time
    model = model.cpu()
    overall_time = time.time() - start_time
    return model, overall_time, best_time

def train_acc(out,y):
    """
    this function is to calculate the training accuracy
    params:
    out: the prediction from model size: [batch_size,class_num]
    y: the ground truth size: [class_num]
    """
    #convert out in to the same size as y
    prediction = torch.argmax(out,dim=1)
    #convert into boolean matrix
    #pdb.set_trace()
    truth = y == prediction
    count = torch.sum(truth == True)
    return count/len(y)  #return acc in this batch

@torch.no_grad()
def test(model, dl, validation_loss_function = None):
    model.eval()
    #acc = Accuracy()
    #pbar = tqdm(dl)
    model = model.cuda()
    true_counts = 0
    test_img_counts = 0
    validation_loss = 0
    for i,batch in enumerate(dl):  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        predict=out.argmax(dim=1).view(-1)
        if validation_loss != None:
            vloss = validation_loss_function(out,y)
            validation_loss += vloss.item()
        predict = torch.Tensor(predict)
        test_img_counts += len(predict)
        y = torch.Tensor(y)
        truth = predict == y
        count = torch.sum(truth == True)
        true_counts += count.item()
    acc = true_counts/test_img_counts
    if validation_loss_function != None: 
        validation_loss = float(validation_loss/(i+1))
        return acc,validation_loss
    else:
        return acc


if __name__ == '__main__':

    load_trained_model_to_train = False
    trained_model_path = ''
    image_dataset_generation = False

    args_temp = args_train()
    config = {'name':args_temp.dataset_name}
    
    
    """
    config_list = [args_train(method = 'efficientvit_b3_r224_in1k',model_name = 'efficientvit_b3.r224_in1k',
               lr = 5*(1e-6),wd = 0.1,batch_size= 32),
                   args_train(method = 'vit_base_patch32_r224',model_name = 'vit_base_patch32_224.augreg_in21k_ft_in1k',
                              lr = 1*(1e-5),wd = 0.1,batch_size= 64),
                    args_train(method = 'vit_base_patch16_r224',model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k',
                               lr = 1*(1e-5),wd = 0.1,batch_size= 64),
                   args_train(method = 'vit_base_patch8_r224',model_name = 'vit_base_patch8_224.augreg_in21k_ft_in1k')                   
        ]
    """
    
    
    """ # proposed model setting
    config_list = [args_train(method = 'vit_base_patch16_r224_block10_wd5e-3', lr = 1*(1e-5),batch_size= 64,
                             model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k',layer_num= 10,crop_model = True,wd = 5e-3)]
    """
    
    
    """ #optimized settings for each model:
    config_list = [args_train(method = 'Final-EffcientNet-B0',model_name = 'efficientnet_b0.ra_in1k',
                              lr = 1*(1e-4),wd = 1e-5,batch_size= 16),
                   args_train(method = 'Final-EfficientNet-B5',model_name = 'efficientnet_b5.sw_in12k',
                              lr = 1*(1e-4),wd = 1e-5,batch_size= 16),
                   args_train(method = 'Final-EfficientViT-B3',model_name = 'efficientvit_b3.r224_in1k',
                              lr = 2*(1e-6),wd = 1e-2,batch_size= 32),
                   args_train(method = "Final-MViT2",model_name = "mvitv2_base_cls.fb_inw21k",
                              lr = 2*(1e-6), wd = 1e-2, batch_size = 16),
                   args_train(method = "Final-ViT-Base-patch8",model_name = "vit_base_patch8_224.augreg_in21k_ft_in1k",
                              lr = 2*(1e-6), wd = 1e-1, batch_size = 16),
                   args_train(method = "Final-ViT-Base-patch16",model_name = "vit_base_patch16_224.augreg_in21k_ft_in1k",
                              lr = 1*(1e-5), wd = 1e-2, batch_size = 64),
                   args_train(method = "Final-ViT-Base-patch32",model_name = "vit_base_patch32_224.augreg_in21k_ft_in1k",
                              lr = 1*(1e-5), wd = 1e-1, batch_size = 64)]
    """
    
    config_list = [args_train(method = 'Just For Test', lr = 1*(1e-5),batch_size= 64,
                             model_name = 'vit_base_patch16_224.augreg_in21k_ft_in1k',layer_num= 10,crop_model = True,wd = 5e-3)]
    
    if image_dataset_generation == True:  #check whether to create new dataest dictionary
        split_images(args_train().root_path,args_train().dataset,random_seed= 128,verbose= True
        ,balance_training = False)
    
    for args in config_list:

        train_dl, test_dl = get_split(args.root_path +'processed',
            batch_size = args.batch_size,evaluation= True, img_size = args.img_size,
            proc_method = args.proc_method, test_loader = True)   #get the dataset
        

        #mkdir for logs and models
        log_dir = os.path.join(args.root_path, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(os.path.join(log_dir,args.method)):
            os.mkdir(os.path.join(log_dir,args.method))
        saved_model_dir = os.path.join(args.root_path, 'models')
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        if not os.path.exists(os.path.join(saved_model_dir,args.method)):
            os.makedirs(os.path.join(saved_model_dir,args.method))
        
        model = create_model(args.model, pretrained= True)
        
        model.cuda()
        
        #start model manipulation
        if args.insert_adapter == True:
            #insert adapters
            set_RepAdapter(model, args.method, dim=args.dim, s= args.scale, args=args) 
                    
        if args.crop_model == True:
            model = model_rebuild(model,args.layer_num)
            
        #choose whether to rebuild the headers: one linear layer -> tow FC layers with ReLU activation and softmax function/ just output change
        if args.rebuild_headers == False:  
            model.reset_classifier(args.class_num)
        else:
            model = model_rebuild(model,add_header= True, class_num=args.class_num)

        if load_trained_model_to_train == True:   #check whether to continue training on previously trained weights
            st = torch.load(trained_model_path)
            model.load_state_dict(st,False)

        model.cuda()
        config['best_acc'] = 0
        config['method'] = args.method
        
        total=0
        trainable = []
        if args.insert_adapter == True:
            for n, p in model.named_parameters():  #only enable adapters to be trainable
                for trainable_layer in args.trainable_layers:
                    if trainable_layer in n:
                        trainable.append(p)
                        total+=p.nelement()  #record the trainable parameters
                        p.requires_grad = True
                        break
                    else:
                        p.requires_grad = False
        else:
            args.trainable_layers = 'This is full training'
            for n,p in model.named_parameters():
                trainable.append(p)
                total += p.nelement()
                p.trainable = True
        print('  + Number of trainable params: %.2fK' % (total / 1e3))  # million as 1 unit
        
        #configure the optimizer and loss function
        opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        scheduler = CosineLRScheduler(opt, t_initial=100,
                                      warmup_t=10, lr_min=1e-7, warmup_lr_init=1e-7, cycle_decay=0.1)

        criterion = torch.nn.CrossEntropyLoss()
        
        callback_early_stoping = callback(stop_patience= args.early_stopping_patience)
        
        #print out some model information
        if args.verbose == True:
          for n,p in model.named_parameters():
              if p.requires_grad == True:
                  print('%s is trainble'%n)
          #print(model)
          H,W = args.img_size
          summary(model, (3, H, W))
          
        #pdb.set_trace()

        
        
        # record configuration
        specification = f'This is optimized training (final) with efficientnetb0, efficientnetb5, effcientvit \
        mvit2, vit-base-p8, vitbase-p16, vit-base-p32 on imbalanced COVID-19 CXR dataset\n \
            img_size = {args.img_size} train/test/validation with {args.model}'
        specif_path = os.path.join(log_dir,args.method,'specification_'+'.txt')
        configuration_record(specif_path,args,(args.batch_size*len(train_dl)
            ,args.batch_size*len(test_dl)),(args.trainable_layers,total),specification)
        
        
        
        #start training data
        model, overall_time,best_time = train(config, model, train_dl, opt, scheduler, test_dl,epoch=args.epoch,
                      criterion = criterion, execution_name = 'paper1_', 
                      callback= callback_early_stoping)
        
        specification = specification + f'\n Time for reach the best validation epoch: {best_time} Time for finish earlystopping: {overall_time}'
        configuration_record(specif_path,args,(args.batch_size*len(train_dl)
            ,args.batch_size*len(test_dl)),(args.trainable_layers,total),specification)
        
        print(config['best_acc'])
        model = 0        
        gc.collect()
        torch.cuda.empty_cache()
