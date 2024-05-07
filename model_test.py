# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:11:00 2023

@author: tc922
"""

import sys
import os
from dataset import *
from model_manipulation import *
from tqdm import tqdm
from timm.models import create_model
from torch import nn
from sklearn.metrics import recall_score,precision_score,confusion_matrix,f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
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
    class_labels = ['COVID-19','Lung Opacity','Normal','Pneumonia']
    model.eval()
    model = model.cuda()
    true_count = 0
    test_length_count = 0
    y_pred = torch.tensor([]).cuda()
    out_pred = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()
    for batch in tqdm(test_dl):
        x, y =batch[0].cuda(), batch[1].cuda()
        out = model(x)
        print(out)
        test_length_count += len(y)
        predict = out.argmax(dim=1)
        y = torch.Tensor(y)
        predict = torch.Tensor(predict)
        print(predict)
        print(y)
        truth = y==predict
        count = torch.sum(truth == True)
        true_count += count.item()
        if verbose == True:
            y_pred = torch.cat((y_pred,predict))
            y_true = torch.cat((y_true,y))
            out_pred = torch.cat((out_pred,out))
    print(y)
    print(predict)
    acc = true_count /test_length_count
    if verbose == True:
        recall, precision,f1,recall_w,precision_w,f1_w = calculate_recall_precision(y_true,y_pred,method = None )
        cm_num = draw_confusion_matrix(y_true,y_pred, class_labels = class_labels, save_path = save_path)
        TP,TN,FP,FN = f_concepts_cm(cm_num, 0)
        draw_roc_curve(out_pred.cpu(), y_true.cpu(), classes_num = [0,1,2,3], class_labels = class_labels)
        return acc, recall, precision,f1,recall_w,precision_w,f1_w        
    else:
        return acc

import pdb

def draw_roc_curve(y_pred_prob, y_true, classes_num, class_labels,
                   save_path = None, softmax_process = True, one_hot_process = True,
                   mode = ""):
    man_thresholds = [np.inf, 1.0000000e+00, 9.9999988e-01, 9.9999976e-01, 9.9999964e-01,
    9.9999952e-01, 9.9999940e-01, 9.9999928e-01, 9.9999917e-01, 9.9999893e-01,
    9.9999869e-01, 9.9999809e-01, 9.9999785e-01, 9.9999678e-01, 9.9999654e-01,
    9.9999321e-01, 9.9999309e-01, 4.4194379e-01, 5.2198684e-10]
    
    man_thresholds = [np.inf, 9.9999988e-01, 9.9999046e-01, 1.0989213e-09]
    """
    This function draws the ROC curve and calculate AUC of All specified class(es).
    
    y_pred_prob: a 2-d array with (sample, probabilities), where probabilities 
        should be summed equal to 1 in each sample
    y_true: the ground truth value default is not one-hot and needed to be one-hot encoded
    classe_nums: the list of lables from the y_true, e.x.: [0,1,2,3] for 4-class classification
    
    softmax_process: if your y_predict is logits (sum is not equal to 1), enable it
    one_hot_process: if the input ground truth is not one-hot encoded, enable it
    
    moode: option: "Normal" -> use sklearn.metrics.roc_curve() to automatically set thresholds
        to draw ROC. "Manual" -> set the self defined thresholds
    
    ROC: different threshold settings for TPR/FPR, reflect the confidence and 
        precision of the model about the prediction it made to the specfic class.
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    """
    if softmax_process == True:
        softmax = nn.Softmax(dim = -1)
        y_pred_prob = softmax(y_pred_prob)
    
    plt.figure(figsize = (15,15))
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 36})
    
   
    for i, class_idx in enumerate(classes_num):
        if one_hot_process == True:
            y_true = label_binarize(y_true, classes = classes_num) #from sklearn.preprocessing import label_binarize
        if mode == "Normal":
            fpr, tpr, _thresh = roc_curve(y_true[:,class_idx], y_pred_prob[:,class_idx])  #from sklearn.metrics
        else:
            TPR = []
            FPR = []
            
            for thresh in man_thresholds:
                y_threshed = []
                #regnerate a prediction list based on the threshold:
                for pred in y_pred_prob[:,class_idx]:
                    if thresh <= float(pred):
                        y_threshed.append(1)
                    elif float(pred) <= 1- thresh:
                        y_threshed.append(0)
                    else:
                        y_threshed.append(-1)
                
                #compare y_threshed with y_ture:
                TP = FN = FP = TN = 0
                for j in range(len(y_threshed)):
                    if y_true[j,class_idx] == 1:
                        if y_true[j,class_idx] == y_threshed[j]:
                            TP += 1
                        elif y_threshed[j] == 0:
                            FN += 1
                    else:
                        if y_threshed[j] == 0:
                            TN += 1
                        elif y_threshed[j] == 1:
                            FP += 1
                #pdb.set_trace()
                #calculate the TPR and FPR
                TPR.append(float(TP/(TP + FN)))
                FPR.append(float(FP/(FP + TN)))
                
            fpr = FPR
            tpr = TPR
            _thresh = man_thresholds
                
                       
        AUC = round(auc(fpr,tpr),4)
        
        plt.plot(fpr, tpr, label = class_labels[i] + f' (AUC: {AUC})')
        print(f'TPR: {tpr},FPR: {fpr}, Threshold Set: {_thresh}')
   
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    
    #draw the reference diagonal line (random classifier result):
    x = [0,1]
    y = [0,1]
    plt.plot(x, y, linestyle = 'dashed')
    
    plt.rcParams.update({'font.size': 1})
    plt.locator_params(axis='x', nbins = 6)  # Increase division on x-axis
    plt.locator_params(axis='y', nbins = 6)  # Increase division on y-axis
   
   
    plt.show()
    
    #print(y_pred_prob[:,class_idx])
    
    if save_path != None:
        plt.savefig(os.path.join('./test_results', save_path + '_roc_' + str(class_idx) + '.png'))
    

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

def draw_confusion_matrix(y_true,y_pred,label_list = [0,1,2,3], class_labels = None, save_path = None):
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
    
    plt.figure(figsize = (25,25))
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 56})
    plt.imshow(cm_num, interpolation='nearest', cmap=plt.cm.Blues)  # Use a blue color map
    for i,cm_value_row in enumerate(cm_num): 
        for j,cm_value in enumerate(cm_value_row):
            cm = str(cm_value) + '\n ' + str(round(cm_prob[i][j]*100,2))+'%'
            if i == j:
                plt.text(j,i,cm,ha= 'center',va= 'center',fontstyle = 'normal',color = 'orange',weight = 'bold',fontsize = 48)
            else:
                plt.text(j,i,cm,ha = 'center',va = 'center',fontstyle = 'normal',color = 'black',weight = 'bold',fontsize = 48)
    #plt.title('Confusion Matrix')
    plt.xlabel('Prediction',color = 'black',weight = 'bold')
    plt.ylabel('Ground Truth',color = 'black',weight = 'bold')
    
    plt.colorbar()
    
    if save_path != None:
        plt.savefig(os.path.join('./test_results', save_path + '_confusion_matrix.png'))
    plt.show()
    return cm_num

def f_concepts_cm(cm_num,class_idx, R = False, P = False, F1  = False, Sp = False):
    """
    This function calculates and turns TP,TN,FP,FN values of specified class.
    cm_num: 2-d tuple like, where the predictions are aligned in column, ground-truth 
    are aligned in row.
    class_idx: the class number you want to calculate with
    """
    length = len(cm_num)
    TP = TN = FP = FN = 0
    
    pass
    for i in range(length):
        for j in range(length):
            if i == class_idx:  #TP and FN
                if i == j:  #TP
                    TP = cm_num[i][j]
                else:   #FN
                    FN += cm_num[i][j]
            else:
                if i == j:  #TN
                    TN += cm_num[i][j]
                elif j == class_idx: #FP
                    FP += cm_num[i][j]
                    
    #print (str(TP/(TP+FP)))
    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    
    return TP,TN,FP,FN

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
    self.save_path = None
    self.trained_model_path = os.path.join('./models',folder_path,model_path)
    self.proc_method = proc_method
    
if __name__ == '__main__':
    
    TEST_LOADER = True
    
    Result_dict = {'Model': [],'Accuracy':[],'Recall(COVID-19)':[],'Precision(COVID-19)':[],'F1 score(COVID-19)':[]
                   ,'Recall(weighted)':[],'Precision(weighted)':[],'F1 score(weighted)':[]}
    
    """
    train_dl, test_dl = get_split('./' + '/processed',evaluation= False, test_loader= TEST_LOADER, 
                                  img_size= args_test().img_size,proc_method=args_test().proc_method)
    """
    """
    args_list = [args_test(folder_path = 'EfficientNetb0_r224',model_path = 'COVID19_CXR_4_best__.pt')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_06',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_0.6')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_08',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_0.8')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_-1',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-1')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_-1_wd',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-1')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_-1_wd1e-4',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-1')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_-1_wd1e-5',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-1')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_-06',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-0.6')
                 ,args_test(folder_path = 'EfficientNetb0_r224_gamma_-08',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-0.8')
                 ,args_test(folder_path = 'EfficientNetb0_r224_wd1e-5_no_bottom_rebuld',model_path = 'COVID19_CXR_4_best__.pt')
                 ,args_test(folder_path = 'EfficientNetb0_r224_wd1e-5_no_bottom_rebuld_gamma_-08',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-0.8')
                 ,args_test(folder_path = 'EfficientNetb0_r224_wd1e-5',model_path = 'COVID19_CXR_4_best__.pt')
                 ,args_test(folder_path = 'EfficientNetb5_r224',model_path = 'COVID19_CXR_4_best__.pt')
                 ,args_test(folder_path = 'EfficientNetb5_r224_gamma_06',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_0.6')
                 ,args_test(folder_path = 'EfficientNetb5_r224_gamma_08',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_0.8')
                 ,args_test(folder_path = 'EfficientNetb5_r224_wd1e-5_no_bottom_rebuld',model_path = 'COVID19_CXR_4_best__.pt')
                 ,args_test(folder_path = 'EfficientNetb5_r224_wd1e-5_no_bottom_rebuld_gamma_-08',model_path = 'COVID19_CXR_4_best__.pt',proc_method = 'Gamma_-0.8')
                 ,args_test(folder_path = 'efficientvit_b3_r224_in1k',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'efficientvit_b3_r224_wd1e-2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'efficientvit_b3_r224_wd1e-3',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'efficientvit_b3_r224_wd1e-4',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'efficientvit_b3_r224_wd1e-2_lr1e-6',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'efficientvit_b3_r224_wd1e-2_lr2e-6',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'efficientvit_b3_r224_wd1e-2_lr2e-7',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch8_r224',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch8_r224_block7',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block5_normal',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block5_gamma_-04',model_path = 'COVID_balancedpaper1_.pt',proc_method = 'Gamma_-0.4')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block7_normal',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block7_gamma_-04',model_path = 'COVID_balancedpaper1_.pt',proc_method = 'Gamma_-0.4')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block7_gamma_-1',model_path = 'COVID_balancedpaper1_.pt',proc_method = 'Gamma_-1')
                 ,args_test(folder_path = 'vit_base_patch32_r224',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block8_normal_wd1e-3',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block8_normal_wd1e-2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block9_normal_wd1e-2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd1e-2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd1e-2_2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd5e-3',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block10_gamma_-0.8_wd5e-3',model_path = 'COVID_balancedpaper1_.pt',proc_method= 'Gamma_-0.8')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block11_normal_wd1e-2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block11_normal_wd1e-2_2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd1e-2_3',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_r224_normal_wd1e-2_2',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'vit_base_patch16_normal_r224_wd5e-3',model_path = 'COVID_balancedpaper1_.pt')
                 ,args_test(folder_path = 'mvit2_base_wd1e-2',model_path = 'COVID_balancedpaper1_.pt')]
    
    
    args_list = [args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd5e-3',model_path = 'COVID_balancedpaper1_.pt')]
    """
    
    args_list = [args_test(folder_path = 'vit_base_patch16_r224_block10_normal_wd5e-3',model_path = 'COVID_balancedpaper1_.pt')]
    prev_proc_method = ''
    for args in args_list:  #for adapter-added vit models
        if args.proc_method != prev_proc_method: 
            train_dl, test_dl = get_split('./' + '/processed',evaluation= False, test_loader= TEST_LOADER, 
                    img_size= args.img_size,proc_method=args.proc_method)
            prev_proc_method = args.proc_method
        model = load_model(args.trained_model_path, state_dict=False)
        # use GradCAM display the extracted features

        # test accuracy
        
        acc,recall,precision,f1,recall_w,precision_w,f1_w = test(model,test_dl, verbose= True, save_path= args.save_path)
        
        Result_dict['Model'].append(args.save_path)
        Result_dict['Accuracy'].append(acc)
        Result_dict['Recall(COVID-19)'].append(recall[0])  #R, P, F1 score for COVID-19 
        Result_dict['Precision(COVID-19)'].append(precision[0])
        Result_dict['F1 score(COVID-19)'].append(f1[0])
        
        Result_dict['Recall(weighted)'].append(recall_w)  #R, P, F1 score in weighted
        Result_dict['Precision(weighted)'].append(precision_w)
        Result_dict['F1 score(weighted)'].append(f1_w)
        
        print(f'Acc: {acc}, Recall: {recall}, Precision: {precision}, F1_score: {f1}')
        model = 0        
        gc.collect()
        torch.cuda.empty_cache()
        
    #df_test = pd.DataFrame(Result_dict)
    #df_test.to_csv('./test_results/' + 'Test_Results' + '.csv')
    
    
    