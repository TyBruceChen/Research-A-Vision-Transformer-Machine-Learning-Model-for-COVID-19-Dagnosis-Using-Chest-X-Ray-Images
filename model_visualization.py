# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:42:31 2023

@author: tc922
"""


import sys
import os
from dataset import *
from model_manipulation import *
from tqdm import tqdm
from timm.models import create_model
from torch import nn
from sklearn.metrics import recall_score,precision_score,confusion_matrix,f1_score
from ImageLoader import get_split
from result_presentation import *
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gc
import pandas as pd

@torch.no_grad()
def test(model,test_dl, verbose = True, save_path = ''):
    """
    when verbose is true, it will generate and save precision, recall score.
    """
    model.eval()
    model = model.cuda()
    true_count = 0
    test_length_count = 0
    y_pred = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()
    for batch in tqdm(test_dl):
        x, y =batch[0].cuda(), batch[1].cuda()
        print(out)
        out = model(x)
        test_length_count += len(y)
        predict = out.argmax(dim=1).view(-1)
        y = torch.Tensor(y)

        predict = torch.Tensor(predict)
        truth = y==predict
        count = torch.sum(truth == True)
        true_count += count.item()
        if verbose == True:
            y_pred = torch.cat((y_pred,predict))
            y_true = torch.cat((y_true,y))
    print(y)
    print(predict)
    acc = true_count /test_length_count
    if verbose == True:
        recall, precision,f1,recall_w,precision_w,f1_w = calculate_recall_precision(y_true,y_pred,method = None )
        draw_confusion_matrix(y_true,y_pred,save_path = save_path)
        return acc, recall, precision,f1,recall_w,precision_w,f1_w        
    else:
        return acc

def calculate_recall_precision(y_true,y_pred,method = 'micro'):
    """
    y_true and y_pred should be a 1-d array-like variable.
    method = 'micro'  #how the recall and precision score will be calculated
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    recall = recall_score(y_true,y_pred,average = method)
    precision = precision_score(y_true,y_pred,average = method)
    f1 = f1_score(y_true,y_pred,average = method)
    method = 'weighted'
    recall_w = (recall_score(y_true,y_pred,average = method))
    precision_w = (precision_score(y_true,y_pred,average = method))
    f1_w = (f1_score(y_true,y_pred,average = method))
    return recall, precision,f1,recall_w,precision_w,f1_w

def draw_confusion_matrix(y_true,y_pred,label_list = [0,1,2,3],save_path = None):
    """
    1-d for true and pred.
    label is a list of numbers which contains all numbers appeared in the prediction
    method: what number will be displayed on CM: 'None': the number of this prediction block
    'pred': the percentage in prediction column
    'true': the percentage in ground truth row
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    cm_num = confusion_matrix(y_true,y_pred, labels = label_list, normalize = None)
    cm_prob = confusion_matrix(y_true,y_pred, labels = label_list, normalize = 'true')
    print(cm_num)
    H,W = cm_num.shape
    class_labels = ['COVID','Lung Opacity','Normal','Pneumonia']
    
    plt.figure(figsize = (25,25))
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 40})
    plt.imshow(cm_num, interpolation='nearest', cmap=plt.cm.Blues)  # Use a blue color map
    for i,cm_value_row in enumerate(cm_num): 
        for j,cm_value in enumerate(cm_value_row):
            cm = str(cm_value) + '\n ' + str(round(cm_prob[i][j]*100,2))+'%'
            plt.text(j,i,cm,ha = 'center',va = 'center',fontstyle = 'normal')
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction',color = 'red')
    plt.ylabel('Ground Truth',color = 'red')
    
    plt.colorbar()
    
    if save_path != None:
        plt.savefig(os.path.join('./test_results', save_path + '_confusion_matrix.png'))
    plt.show()

class args_test:
  """
  This args class is for store training information
  """
  def __init__(self,seed = 32,
               folder_path = 'vit_base_patch16_fd',
               method = 'vit',
               class_num = 4,
               dataset = 'COVID-19_Radiography_Dataset',
               model = 'vit_base_patch16_224.augreg_in21k_ft_in1k',
               proc_method = 'Normal',
               model_path = 'COVID19_4_16patch_384px_2e6r12_.pt',
               image_dataset_generation = False,grad_cam_folder = '',
               img_size = (224,224)):
    self.seed = seed
    
    self.method = method
    self.class_num = class_num
    self.dataset = dataset
    self.model = model
    self.image_dataset_generation = image_dataset_generation
    self.grad_cam_folder = grad_cam_folder
    self.img_size = img_size
    self.folder_path = folder_path
    self.save_path = folder_path
    self.trained_model_path = os.path.join('./models',folder_path,model_path)
    self.proc_method = proc_method
    
if __name__ == '__main__':
    
    TEST_LOADER = True
    args = args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd5e-3',model_path = 'COVID_balancedpaper1_.pt')
    train_dl, test_dl = get_split('./' + '/processed',evaluation= False, test_loader= TEST_LOADER, 
            img_size= args.img_size,proc_method=args.proc_method)
    model = load_model(args.trained_model_path, state_dict=False)
    img_paths_format = ['./COVID-19_Radiography_Dataset/COVID/images/COVID-NUM.png',
                      './COVID-19_Radiography_Dataset/Lung_Opacity/images/Lung_Opacity-NUM.png',
                      './COVID-19_Radiography_Dataset/Normal/images//Normal-NUM.png',
                      './COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-NUM.png']
    img_paths = [[],[],[],[]]
    for i in range(1,20):
        img_paths[0].append(img_paths_format[0].replace('NUM',str(i)))
        img_paths[1].append(img_paths_format[1].replace('NUM',str(i)))
        img_paths[2].append(img_paths_format[2].replace('NUM',str(i)))
        img_paths[3].append(img_paths_format[3].replace('NUM',str(i)))
    
    
    for i, test_img_paths in enumerate(img_paths):
        # i is to specify each class for each tested each img group
        CAM = GradCAM_eval(model, model.norm, test_img_paths,category_value = i,save= True,display=False, use_vit_model= True, folder = 'OPM')
        CAM()
        