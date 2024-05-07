# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:46:55 2023

@author: tc922
"""

import torch
import timm
from ImageLoader import *
from torch.optim import Adam
from torch import nn
import time
from tqdm import tqdm
from model_manipulation import *
import os
from timm.scheduler.cosine_lr import CosineLRScheduler
from result_presentation import *
from torchsummary import summary
import pdb
import gc


def train(config,model,opt,loss_function,train_dl,test_dl,args,callback= None,appendency = '_'):
    #enable the model to be trainabel (make sure)
    
    save_interval = 10
    model.train()
    model = model.cuda()    #let the model be trained on GPU
    best_acc = 0
    best_time = 0
    test_loss_function = nn.CrossEntropyLoss()
    start_time = time.time()
    for ep in tqdm(range(args.epoch)):  #start going through each epoch
        loss_train = 0
        acc_train = 0  # record the train accuracy
        model.cuda()
        for i,batch in enumerate(train_dl):
            model.train()
            x, y = batch[0].cuda(), batch[1].cuda() #also let tensors transfer to GPU
            train_predict = model(x)
            opt.zero_grad()  # zero the gradient for every batch
            #pdb.set_trace()
            acc_train += train_acc(train_predict,y)
            loss = loss_function(train_predict,y)
            loss_train += loss.item()
            loss.backward()
            opt.step()
            
        acc_train = float(acc_train/(i+1))  #calculate the train accuracy
        loss_train = float(loss_train/(i+1))
        acc, loss_valid = test(model,test_dl,test_loss_function)  #calculate the validation accuracy
        
        log_dir = os.path.join(args.root_path,'logs')
        model_dir = os.path.join(args.root_path,'models')
        if acc >= best_acc:  #if current acc is greater than previous one, save current model and log.
            best_acc = acc
            config['best_acc'] = acc
            best_time = time.time() - start_time 
            save_name = args.dataset_name + '_best_' + appendency #name for save as model and log
            save_model(model_dir,log_dir,args.method,save_name,model,acc,ep,model_save = True, 
                       acc_train = acc_train,loss_train = loss_train, loss_valid = loss_valid)
            
        if ep%save_interval == 0:  #regular saving models
            save_name = args.dataset_name + '_usual_' + str(ep) + appendency
            save_model(model_dir,log_dir,args.method,save_name,model,acc,ep,model_save = True, 
                       acc_train = acc_train,loss_train = loss_train, loss_valid = loss_valid)
        #save log for each epoch
        save_name = args.dataset_name + '_usual_' + str(ep) + appendency
        save_model(model_dir,log_dir,args.method,save_name,model,acc,ep,model_save = False, 
                   acc_train = acc_train,loss_train = loss_train, loss_valid = loss_valid)
        
        if callback != None:
            early_stop_state = callback.early_stopping(acc)
            if early_stop_state == True:
                print()
                print('Early Stopping!')
                best_ep, best_ac = callback.get_best_value()
                print('Best Epoch: %d, Best Accuracy: %.2f'%(best_ep,best_ac))
                overall_time = time.time() - start_time
                return model, best_time, overall_time
        
    model = model.cpu()
    overall_time = time.time() - start_time
    return model,best_time, overall_time

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
        
class args:
  """
  This args class is for store training information
  """
  def __init__(self,seed = 32, dataset = 'COVID-19_Radiography_Dataset',
               trained_model_path = './models/EfficientNetb0/EfficientNetb0__99.pt',
               root_path = './',  #under the same directory
               mode = 'train',epoch= 200, verbose= True,early_stopping_patience = 20,
               class_num = 4, dataset_name = 'COVID19_CXR_4',bottom_layer_in_features= 1280,
               lr = 1e-3,wd = 0.016 ,batch_size = 16, img_size = (224,224),
               proc_method = 'Normal',
               model_name = 'efficientnet_b0.ra_in1k',
               method = 'EfficientNetb5_r224'
               ):
    """
    This class is to store configurations.
    params:
    method: where the models and logs are saved, often named by the model name + special technique used
    model: the full name of the (pretrained) model in timm
    verbose: True or False to decide whether to display model structure, etc
    """
    self.seed = seed
    self.lr = lr
    self.wd = wd
    self.model = model_name
    self.root_path = root_path
    self.method = method
    self.mode = mode
    self.class_num = class_num
    self.dataset = dataset
    self.epoch = epoch
    self.trained_model_path = trained_model_path
    self.batch_size = batch_size
    self.dataset_name = dataset_name  #nick name for dataset
    self.verbose = verbose
    self.img_size = img_size
    self.bottom_layer_in_features = bottom_layer_in_features
    self.early_stopping_patience = early_stopping_patience
    self.proc_method = proc_method
    
from timm.models import create_model
class EfficientNetTl(nn.Module):
    def __init__(self,class_num= 4,bottom_layer_in_features= 1280,model_name = 'efficientnet_b5.sw_in12k',add_bottom_layer = False):
        super().__init__()
        #build the pre_trained model efficientnet-b0 from timm library
        self.base_model = create_model(model_name = model_name,pretrained = True)
        if add_bottom_layer == True:
            # bottom_layer is the position where we want to add after the base_model (pretrained)
            self.bottom_layers = nn.Sequential(
                #BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)  
                #for the nn, the momentum is reverse to tensorflow: 1- 0.99 = 0.01
                nn.BatchNorm1d(bottom_layer_in_features,momentum= 0.01,eps= 1e-3), 
                #Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                # bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
                nn.Linear(bottom_layer_in_features, 256,bias= True),
                nn.ReLU(),
                #Dropout(rate= 0.45, seed= 123),
                nn.Dropout(p= 0.45),
                #Dense(class_count, activation= 'softmax')
                nn.Linear(256, class_num),
                nn.Softmax()
            )
            self.base_model.classifier = nn.Identity()
        else:
            self.base_model.reset_classifier(class_num)
    def forward(self,x):
        x = self.base_model(x)
        return x
    
def get_trainable(model):
    num_trainable_params = 0
    trainanle_params = []
    for n,p in model.named_parameters():
        num_trainable_params += p.nelement()
        trainanle_params.append(p)
        p.requires_grad = True
    return num_trainable_params, trainanle_params




if __name__== '__main__':
    
    args_train = [args(model_name= 'efficientnet_b5.sw_in12k', method = 'EfficientNetb5_r224_wd1e-5_no_bottom_rebuld_gamma_-08',bottom_layer_in_features= 1280,proc_method= 'Gamma',wd = 1e-5)
                  ,args(model_name= 'efficientnet_b5.sw_in12k', method = 'EfficientNetb5_r224_wd1e-5_no_bottom_rebuld',bottom_layer_in_features= 1280,wd = 1e-5)]
    
    for args in args_train:
        model = EfficientNetTl(class_num = args.class_num,model_name= args.model,
            bottom_layer_in_features = args.bottom_layer_in_features)
        
        if args.mode == 'train':
            config = {}
            #create/examine the dirs for storing logs and models
            log_dir = os.path.join(args.root_path,'logs')
            model_dir = os.path.join(args.root_path,'models')
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            if not os.path.exists(os.path.join(log_dir,args.method)):
                os.mkdir(os.path.join(log_dir,args.method))
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            if not os.path.exists(os.path.join(model_dir,args.method)):
                os.mkdir(os.path.join(model_dir,args.method))
            # test_dl here is the validation set
            total, trainable = get_trainable(model)
            if args.verbose == True:
                H,W = args.img_size
                summary(model, (3,H,W))
                print(f'Trainable Params: {total}')
            train_dl,test_dl = get_split(args.root_path+'/processed',evaluation = True,
                batch_size=args.batch_size,proc_method= args.proc_method, test_loader= False)
            
            print('There are totoal %.2fk trainabel parameters'%(total/1000))
            opt = Adam(trainable, lr = args.lr, weight_decay= args.wd) 
            #scheduler = CosineLRScheduler(opt, t_initial=100,
             #                             warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
            loss_function = torch.nn.CrossEntropyLoss()
            
            callback_early_stopping = callback(stop_patience = args.early_stopping_patience)
            
            specification = 'This is weight_decay1e-5 removed bottom_layer gamma correction -0.8 training on inbalanced COVID-19 CXR '
            specif_path = os.path.join(log_dir,args.method,'specification_.txt')
            configuration_record(specif_path,args,(args.batch_size*len(train_dl)
                ,args.batch_size*len(test_dl)),(None,total),specification)
            
            model, best_time, overall_time = train(config,model,opt,loss_function,train_dl,test_dl,args,callback=callback_early_stopping)
            #record related specification
            specification = specification + f'\n Time for reach the best validation epoch: {best_time} Time for finish earlystopping: {overall_time}'
            configuration_record(specif_path,args,(args.batch_size*len(train_dl)
                ,args.batch_size*len(test_dl)),(None,total),specification)
            #print(model)
            print(config['best_acc'])
            model = 0        
            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            model = load_model(model, args.trained_model_path)
            model.eval()
            train_dl, test_dl = get_split(args.root_path+ '/processed',evaluation= False, test_loader= True)
            acc = test(model,test_dl)
            summary(model,(3,224,224))
            for class_name in ['COVID/','Normal/','Lung_Opacity/','Viral Pneumonia/']:  #to iterate through each class folder name
                sample_path = sample_path.replace(prev, class_name)  #replace the name of folder
                prev = class_name
                print(sample_path)
                img_path = [os.path.join(sample_path,path) 
                             for path in os.listdir(sample_path)]  #get the img paths into a list
                test_img_path += (img_path[:10])  #only take first 10 img paths in each folder and concatenate them
            for category_value in range(4):  #draw the Grad-CAM w.r.t. the four categories (classes)
                cam_display = GradCAM_eval(model,model.blocks[11],test_img_path,
                                       use_vit_model = True,category_value = category_value,
                                       save= True,display= False,folder = args.grad_cam_folder)
                cam_display()
            print(acc)
