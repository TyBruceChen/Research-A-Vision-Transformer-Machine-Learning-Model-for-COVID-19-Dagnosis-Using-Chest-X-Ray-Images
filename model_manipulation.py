# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:45:09 2023

@author: tc922
"""
from torch import nn
import torch
from efficientnet import EfficientNetTl

def model_rebuild(model, layer_num=-1, add_header= False, class_num= 0):
    """
    layer_num: set the number of blocks that will be keeped after cropping
    add_header: set the header(BN,LN(ReLU),Dropout,LN(Softmax)) for the model
    """
    if 'vision_transformer' in str(type(model)):
        if layer_num != -1:
            block_length = len(model.blocks)
            if layer_num > block_length: # check if the cropping layer exceed the max layer
                print('Invalid! the maximum length of the blocks in model is %d'%block_length)
                return 'Failure'
            new_model = model
            for i in range(layer_num,block_length):
                new_model.blocks[i] = nn.Identity()
        if add_header == True:
            model.head = nn.Sequential(nn.Linear(768, 256,bias= True),
                                       nn.ReLU(),
                                       nn.Dropout(p = 0.45),
                                       nn.Linear(256, class_num),
                                       nn.Softmax())
            new_model = model
    return new_model

class callback:
    """
    This class is to realize early stopping function, which means it will stop training
    the model when there's no improvement after specific continous epochs
    """
    def __init__(self,stop_patience= 20,threshold= 0.0005):
        """
        patience: number of no-improvement epochs
        parameters:
        stop_patience : the maximum patience, when the class is initialized, 
            patience= stop_patience as well as there's an improvement.
        threshold : define how much increse than previous one is an improvement.
            *this value should range from 0 to 1
        variables:
        patience : this is the variable to record the left patience(times/epochs) that 
            model can have no improvement.
        best_epoch : DESCRIPTION record the best epochs which has the best performance til now.
        best_accuracy: record the best accuracy, so that it can compare with the input one

        """
        self.patience = stop_patience   # patience should be intialized to the maximum
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.best_epoch = 0
        self.best_accuracy = 0
        self.epoch = 0
    def early_stopping(self,acc):
        """
        acc: the input (accuracy) from current training
        """
        if acc < self.best_accuracy + self.threshold:
            self.patience = self.patience -1
            self.epoch += 1
        else:
            self.patience = self.stop_patience
            self.best_epoch = self.epoch
            self.epoch += 1
            self.best_accuracy = acc
        if self. patience <=0:
            return True
        else:
            return False
    def get_best_value(self):
        return self.best_epoch,self.best_accuracy


def save_model(model_dir,log_dir,method, save_name, model, acc, ep, 
               model_save = True, save_entire_model = True, acc_train = None, 
               loss_train = None, loss_valid = None):
    """
    model_dir/log_dir: where the parent dir for all models/logs' path, like './model'
    
    """
    model = model.cpu()
    trainable = {}
    if model_save == True:
        if save_entire_model == True:
            torch.save(model, model_dir+ '/%s/%s.pt'%(method, save_name))
        else:
            for n, p in model.named_parameters():
                if p.requires_grad == True:
                    trainable[n] = p.data
            torch.save(trainable, model_dir+ '/%s/%s.pt'%(method, save_name))
    
    if acc_train != None:
        content =  f'{str(ep)} valid_acc {str(acc)}\n valid_loss {loss_valid}\n train_loss {loss_train}\n train {str(acc_train)}'
    else:
        content = f'{str(ep)} valid_acc {str(acc)}\n valid_loss {loss_valid}\n train_loss {loss_train}'
    
    with open(log_dir + '/%s/%s.log'%(method, save_name), 'w') as f:
        f.write(content)

def load_model(path,model = None,state_dict=False, key_match = True):
    if state_dict == True:  #when the params are saved with the correspding nn.Module, not who model
        state_dict = torch.load(path)
        #for n,p in model.named_parameters():
            #print(n)
        model.load_state_dict(state_dict,key_match)  # True means the state_dict must perfectly match with model structure.
        return model
    else:
        loaded_model = torch.load(path)
        return loaded_model
