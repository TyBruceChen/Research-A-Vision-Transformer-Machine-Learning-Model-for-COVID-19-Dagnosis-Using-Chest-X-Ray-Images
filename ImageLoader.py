# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:37:25 2023

@author: tc922
"""

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import random
import os
import shutil
from PIL import ImageOps
import math

def open_img(img_path, img_size):
    """
    get the img path and return the img pixel value
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size)
    return img  

                
        

class ImgFileLoader:
    def __init__(self,file_path,img_size,transform,imgloader=open_img,load_method ='tensor',
                 test_loader= False,proc_method= 'Normal',show_img= True,factor_a = 1):
        """
        parameters:
        fiel_path: the path of the txt file store the images' path and labels
        img_size: (width,height)
        transform: the transform class built in torch to transform numpy to tensor
        imgloader: specify the function to read the imgs
        load_method: the way to load/store dataset: 'tensor' -> store the img pixel value
            not 'tensor' -> store the img path. The former will take more time to load before training
            and more RAM and GPU memory. But it will be much faster than latter after training begin.
        proc_method: the processing on img that may be implemented before loading imgs
            'Normal' -> no processing, 'Mask' -> add masking above the original which will read another
            set of mask imgs and only display the critical part, 'Gamma' -> adjust the gamma value 
            before processing ...
        test_loader: if true, the dataset will only load 100 images
        """
        self.load_method = load_method
        self.img_size = img_size
        self.imgloader = imgloader
        
        self.transform = transform
        f = open(file_path,'r')
        lines = [line for line in f]
        if test_loader == True:
            lines = lines[:100]
        self.img_list = []
        self.proc_method = proc_method
        if load_method == 'tensor':
            for line in lines:
                img_path, label = line.strip().split('  ')
                img = self.imgloader(img_path,self.img_size)
                
                if proc_method == 'Mask':
                    mask_path = img_path.replace('images','masks')
                    mask = self.imgloader(mask_path,self.img_size)
                    img = cv2.bitwise_and(np.array(img), np.array(mask))
                   
                if proc_method == 'Mask_half':
                    mask_path = img_path.replace('images','masks')
                    mask = np.asarray(self.imgloader(mask_path,self.img_size))
                    h,w,c = mask.shape
                    h = int(0.7*h)  #unmask the bottom 30% region
                    mask_half = np.array(mask)
                    mask_half[h:,:,:] = 255
                    img = cv2.bitwise_and(np.array(img), mask_half)
                    
                if 'Gamma' in proc_method:  #the input format of gamma preprocess method should be "Gamma_value" 
                    #the factor value should range from [-1,1]
                    _, factor_a = proc_method.split('_')
                    factor_a = float(factor_a)
                    #factor_a = -0.8  #weight factor
                    eta = 1e-5  #keep the numerical stability
                    img_np = np.array(img, dtype = np.float32)
                    midpoint = np.median(img_np)
                    phi = math.pi*img_np/(2*midpoint + eta)
                    f1 = factor_a*np.cos(phi)
                    gamma = 1 + f1 +eta
                    gamma_correction = 255*((img_np/255)**(1/gamma))
                    img = gamma_correction/255
                img = self.transform(img)
                self.img_list.append((img,label))
        else:
            for line in lines:
                img_path, label = line.strip().split('  ')
                self.img_list.append((img_path,label))
                if proc_method != 'Normal':
                    print('Error! the image processing can be only appended in tensor load mode')
                    print('As a result, the images will not do any processing')
        f.close()
        if show_img == True:
            img,_ = self.__getitem__(3)
            print(type(img))
            plt.imshow(img.permute(1,2,0))
        f.close()
    def __getitem__(self,idx):
        if self.load_method == 'tensor':
            img,lable = self.img_list[idx]
            return img,int(lable)
        else:
            img_path, label = self.img_list[idx]
            img = self.imgloader(img_path,self.img_size)
            img = self.transform(img)
            return img, int(label)
    def __len__(self):
        return len(self.img_list)
    
    
def get_split(root_path,img_size=(224,224),batch_size=16,evaluation=True, proc_method = 'Normal',
              test_loader = False):
    """
    This function pass the img_paths to create DataLoader
    param:
    root_path: the path of .txt file which stores imgs' path 
    evaluation: if true, it provides data for training (train,validation)
        if false, it provides data for testing (train,test)
    """
    train_path = root_path + '/train.txt'
    valid_path = root_path + '/val.txt'
    trainval_path = root_path + '/trainval.txt'
    test_path  = root_path + '/test.txt'
    if evaluation == True:
        test_path = valid_path
        
    transform = transforms.Compose([transforms.ToTensor()])  #ToTensor convert PIL imgs range from [0,255] to tensor range from [0,1]
    train_dl = torch.utils.data.DataLoader(ImgFileLoader(train_path,img_size,transform,proc_method= proc_method
        , test_loader = test_loader),batch_size = batch_size,shuffle = True, num_workers = 0, 
        drop_last = True,pin_memory = True)
    
    test_dl = torch.utils.data.DataLoader(ImgFileLoader(test_path,img_size,transform,proc_method= proc_method
        ,test_loader = test_loader),batch_size = batch_size,shuffle = True, num_workers = 0, 
        drop_last = True,pin_memory = True)
    
    return train_dl,test_dl
    



def split_images(root_path,dataset_name,random_seed= 42,train_ratio= 0.65,validation_ratio= 0.15,
                 test_ratio= 0.2,evaluate= True, verbose= True,balance_training= False):
  """
  This function is to handel the splitted images (separated into corresponding folders) to
  record their abosolute path (in current working dir) and label into .txt files: train,
  validation, test and (may) trainval .txt.
  Default split ratio: train:0.65, validation:0.15, test:0.2
  """
  if os.path.exists(root_path+'/processed') != True:
      #shutil.rmtree(root_path+'/processed')
      os.mkdir(root_path+'/processed') #make the root directory

  random.seed(random_seed)
  dirs = os.listdir(os.path.join(root_path,dataset_name))

  dirs = [dir for dir in dirs if '.' not in dir]  # here dirs stores the names of class folders
  print(dirs)

  img_path_list_t = []  # this store the full path of all images
  img_label_list_t = [] # this store the class (label) of all images

  class_list = [] # this is to store recorded classes
  class_dataset_length = []
  class_num_relation = []
  for label_number,class_name in enumerate(dirs):
      #full path of few folders which store different class of images
      #here also appedn 'images' folder after the class name
    class_path = os.path.join(root_path,dataset_name,class_name,'images')
    imgs = os.listdir(class_path)      # get the image list in one class folder
    print(class_path)
    img_class_label_list =[]
    img_class_path_list = []
    class_num_relation.append((class_name,label_number))
    for img in imgs:
      img_path = os.path.join(class_path,img) #concatenate class folder's path with each image
      # old version:
      #label = img.split('-')[0]  # this is the modified name, because the img name contain label
      #label, class_list = class_converter(label,class_list) # convert classes to digital names.
      img_class_label_list.append(label_number)
      img_class_path_list.append(img_path)  #save all full path of img in img_path_list
    # these *_t list should shape like (number_of_classes,)     
    img_label_list_t.append(img_class_label_list)
    img_path_list_t.append(img_class_path_list)  #save all full path of img in img_path_list
    class_dataset_length.append(len(img_class_label_list)) # record the length of each class list
      # for latter balance training (if requested)
  img_path_list = []  # this store the full path of all images
  img_label_list = [] # this store the class (label) of all images
  if balance_training == True: # when balance training, the length should be same
    min_length = min(class_dataset_length) -1 # get the length of the smallest class dataset
  for class_label_list in img_label_list_t:
    if balance_training == True: # when balance training, the length should be same
      img_label_list += class_label_list[:min_length]
    else:
      img_label_list += class_label_list
  for class_path_list in img_path_list_t:
    if balance_training == True: # when balance training, the length should be same
      img_path_list += class_path_list[:min_length]
    else:
      img_path_list += class_path_list  
  zip_dataset = list(zip(img_path_list, img_label_list))

  random.shuffle(zip_dataset) #shuffle (full images path, img name)

  dataset_length = len(zip_dataset)  # use the total length of dataset to separate sets

  # save train images and train.txt file to record the image name,type
  f = open(root_path+'/processed/'+'train.txt','w')
  f_mask = open(root_path+ '/processed/'+'train_mask.txt','w')
  for zip_data in zip_dataset[:int(train_ratio*dataset_length)]:
      img_path, label = zip_data
      line = img_path + '  ' + str(label) +'\n'	 # separate the relative path with 2 spaces
      f.write(line)
      f_mask.write(img_path.replace('images', 'masks')+'\n')
  f.close()
  f_mask.close()
  
  
  # save validation images and train.txt file to record the image name,type
  f = open(root_path+'/processed/'+'val.txt','w')
  f_mask = open(root_path+'/processed/'+'val_mask.txt','w')
  for zip_data in zip_dataset[int(train_ratio*dataset_length):
                              int(train_ratio*dataset_length+test_ratio*dataset_length)]:
      img_path, label = zip_data
      line = img_path + '  ' + str(label) +'\n'	 # separate the relative path with 2 spaces
      f.write(line)
      f_mask.write(img_path.replace('images', 'masks')+'\n')
  f.close()
  f_mask.close()

  # save train images and test.txt file to record the image name,type
  f = open(root_path+'/processed/'+'test.txt','w')
  f_mask = open(root_path+'/processed/'+'test_mask.txt','w')
  for zip_data in zip_dataset[int(train_ratio*dataset_length+test_ratio*dataset_length):
                                int(dataset_length)]:
      img_path, label = zip_data
      line = img_path + '  ' + str(label) +'\n'	 # separate the relative path with 2 spaces
      f.write(line)
      f_mask.write(img_path.replace('images', 'masks')+'\n')
  f.close()
  f_mask.close()

  # if evaluation is enable, concatenate the train and validation set together
  if evaluate == True:
      f1 = open(root_path+'/processed/'+'train.txt','r')
      content1 = f1.read()
      f2 = open(root_path+'/processed/'+'val.txt','r')
      content2 = f2.read()
      combined_content = content1 + content2
      f3 = open(root_path+'/processed/'+'trainval.txt','w')
      f3.write(combined_content) 
      f1.close()
      f2.close()
      f3.close()
      
      f1_mask = open(root_path+ '/processed/'+'train_mask.txt','r')
      f2_mask = open(root_path+'/processed/'+'val_mask.txt','r')
      f3_mask = open(root_path+'/processed/'+'trainval_mask.txt','w')
      mask1 = f1_mask.read()
      mask2 = f2_mask.read()
      mask3 = mask1 + mask2
      f3_mask.write(mask3)
      f1_mask.close()
      f2_mask.close()
      f3_mask.close()
      
      
  if verbose == True:
      print('Dataset_length: %d'%dataset_length)
      print('Train percentage: %.2f, Validation percentage: %.2f, Test percentage" %.2f'
        %(train_ratio,validation_ratio,test_ratio))
      print('Class names with corresponding numbers:')
      f4 = open(root_path+'/processed/'+'num_class_relation.txt','w')
      f4.write('Dataset_length: %d'%dataset_length)
      for label, num in class_num_relation:
        print(label + ': ' + str(num),end=',')
        f4.write(label + ': ' + str(num))
      f4.close()
      print()
      if balance_training == True:
        print('Balance mode, each class lenght is: %d'%min_length)
      else:
        print('length of each class: ')
        for length in class_dataset_length:
          print(length)

    