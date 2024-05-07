# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:30:21 2023

@author: tc922
"""
import matplotlib.pyplot as plt
import os

def get_epoch(list):
    return list[0]
def result_presentation(log_path, log_name,devider):
    """
    log_path: the path where you store the log files
    log_name: the name of same series of logs you want to display (same name or critical name)
    """
    log_path_list = []
    acc_list = []
    epoch_list = []
    result = []
    for log in  os.listdir(log_path):
        if log_name in log:
            log_path_list.append(os.path.join(log_path,log))
    for log in log_path_list:
        f = open(log,'r')
        line = f.read()
        epoch, acc = line.strip().split(devider)
        result.append((int(epoch),float(acc)))
        acc_list.append(float(acc))
        epoch_list.append(int(epoch))
    result = sorted(result, key = get_epoch)
    max_acc = max(acc_list)
    max_acc_epoch = acc_list.index(max_acc)
    acc_list = []
    epoch_list = []
    for _ in result:
        epoch_list.append(_[0])
        acc_list.append(_[1])
    plt.figure(figsize = (10,10))
    
    plt.plot(epoch_list,acc_list,'r',label = 'Validation Accuracy')
    best_acc = f'best_acc_epoch:{str(max_acc_epoch)}: {str(max_acc)}'
    plt.scatter(max_acc_epoch, max_acc, s = 150,c= 'blue',label = best_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def configuration_record(log_path,config,dataset_length,trainable_param,specification= ' '):
    """
    log_path: the location where the specification is stored
    config: the object that store the hyper-parameter information
    trainble_param: (trainble_names, trainable_nums)
    specification: the extra specification for this training
    """
    f = open(log_path,'w')
    train_length, test_length = dataset_length
    trainable_name, trainable_num = trainable_param
    content = f'Dataset:  {config.dataset}, train:  {train_length}  test:  {test_length}\n \
        learning_rate = {config.lr}  batch_size = {config.batch_size} weight_decay = {config.wd}\n \
        pre-process method: {config.proc_method}  \
        model:  {config.model}\n \
        traible: {trainable_name} numbers: {trainable_num/1000}K\n \
        early stopping patience: {config.early_stopping_patience}\n \
        specification: {specification}'
    f.write(content)
    f.close()


import pytorch_grad_cam
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import numpy as np
import cv2
import torch

class GradCAM_eval:
    def __init__(self,model,target_layer,test_img_paths,
                 category_value = 2,save= False,use_vit_model= False,display= True,folder = ''):
        """
        target_layer: the layer you want to see the weights of featurs [layer]
        test_img_paths: the list of img path that you want to let the cam to extract to display
        category_value: this can be tricky, it can be negative, but its same as MAX-it
        """
        self.model = model
        self.target_layer = [target_layer]
        self.test_img_paths = test_img_paths
        self.category_value = category_value
        self.save = save
        self.vit_model = use_vit_model
        self.display = display
        self.folder = folder
    def preprocess_image(self,img, mean=[
            0.5, 0.5, 0.5], std=[
                0.5, 0.5, 0.5]):
        """
        build tensor to feed into the model
        """
        preprocessing = Compose([
            ToTensor(), #ToTensor is designed for covert PIL type to tensor, (H,W,C) -> (C,H,W) with 
                 #[0,255] to [0.0,1.0] torch.FloatTensor type. To reverse this operation, use ToPILImage
            Normalize(mean=mean, std=std)
        ])
        return preprocessing(img.copy()).unsqueeze(0).cuda() # torch.unsqueeze(dim) 
            #just expands in the 'dim' channel for 1 dimmension 
    
    def PIL2Tensor_img(self,img_path):
        img = np.asarray(Image.open(img_path).convert('RGB').resize((224,224)))
        img = np.float32(img) / 255  # the cam display function show_cam_on_image only
          #receive the img in np.float32 format and pixel value range between [0,1]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return self.preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), img
    
    def reshape_transform(self,tensor,height=7,width=7):
        """
        For ViT model, the (feature map)output from target layer needs to be processed:
            (batch, patches, feature_nodes) -> (batch, feature_nodes, height, width)
        """
        result = tensor[:,:49,:].reshape(tensor.size(0),height,width,tensor.size(2))
        result = result.transpose(2,3).transpose(1,2)
        return result
    
    def create_extraction_img(self,img_path,mask_path,save_root_path = './'):
        img = Image.open(img_path).convert('RGB').resize((224,224))
        mask = Image.open(mask_path).convert('RGB').resize((224,224))
        extraction = cv2.bitwise_and(np.array(img), np.array(mask))
        extraction = Image.fromarray(extraction)
        extraction.save(os.path.join(save_root_path,img_path.split('.')[0]+'-extraction.png'))
    class similarityToConceptLayer():
        """
        waiting for future develop: here this funtion compares the cosine similarity bewtween two 
        images, one is from the last layer of model (concept features) but not include FC layer 
        and the input image for it should just contain the object you want to highlight,
        another is the target output layer. U can use this output to replace category, which will 
        show the CAM on the object you want to see.
        """
        def __init__(self,features):
            self.features = self.features
        def __call__(self,model_output):
            cos = torch.nn.CosineSimilarity(dim= 0)
            return cos(model_output, self.features)
    
    def __call__(self):
        for img_path in self.test_img_paths:
            input_tensor,img = self.PIL2Tensor_img(img_path)
            model = self.model.cuda()
            model.eval()
            if self.vit_model == True:
                cam = pytorch_grad_cam.GradCAM(model= model,target_layers = self.target_layer,
                    reshape_transform= self.reshape_transform,use_cuda = True)
            else:
                cam = pytorch_grad_cam.GradCAM(model= model,target_layers = self.target_layer,
                                               use_cuda = True)
            targets = [pytorch_grad_cam.utils.model_targets.ClassifierOutputTarget(self.category_value)]
            #ex_tensor, ex = self.PIL2Tensor_img('./1-extracted.png')
            #lung_concept_features = model()
            #lung_similarity = similarityToConceptLayer()
            grayscale_cams =  cam(input_tensor= input_tensor,targets = targets)
            cam_image = pytorch_grad_cam.utils.image.show_cam_on_image(img, grayscale_cams[0,:],use_rgb= True)
            cam = np.uint8(255*grayscale_cams[0, :])
            cam = cv2.merge([cam, cam, cam])
            images = np.hstack((np.uint8(255*img), cam , cam_image))
            img = Image.fromarray(images)
            if self.save != False:
                img_path = img_path.split('/')[-1]
                folder = './GradCAM/'+self.folder +'_'+str(self.category_value)
                if os.path.exists(folder) == False:
                    os.mkdir(folder)
                img.save(f'{folder}/{img_path}.jpg')
            if self.display == True:
                img.show()
