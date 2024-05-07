#!/usr/bin/env python
# coding: utf-8

# In[179]:


result_presentation(['logs/vit_base_patch8_r224','logs/vit_base_patch16_r224',
                     'logs/vit_base_patch32_r224','logs/efficientvit_b3_r224_in1k'],
                    ['COVID_balanced_usual_paper1_','COVID_balanced_usual_paper1_',
                     'COVID_balanced_usual_paper1_','COVID_balanced_usual_paper1_'],
                    'ViT-Base_p8_16_32_r224, EfficientViT-B3_r224',
                    True,
                    label_list = ['vit_patch8_r224','vit_patch16_r224','vit_patch32_r224','efficientvit-B3'])


# In[381]:


result_presentation(['logs/vit_base_patch16_r224','logs/vit_base_patch16_r224_block5_gamma_-04','logs/vit_base_patch16_r224_block7_gamma_-04',
                    ,'logs/vit_base_patch16_r224_block7'],
                    ['COVID_balanced_usual_paper1_','COVID_balanced_usual_paper1_','COVID_balanced_usual_paper1_','COVID_balanced_usual_paper1_'],
                    'ViT-Base_p8_16_32_r224, EfficientViT-B3_r224',
                    True,
                    label_list = ['vit_base_patch16','vit_base_patch16_block','vit_base_patch16_block5'
                        ],
                    save_df = True,save_fig = True,fig_save_name = 'ViT_block_cropping_gamma_correction')


# In[179]:


result_presentation(['logs/mvit2_base_wd1e-2'],
                    ['COVID_balanced_usual_paper1_'],
                    'ViT-Base_p8_16_32_r224, EfficientViT-B3_r224',
                    True,
                    label_list = [''], plot_selection = 'Loss',
                    save_df = False,save_fig = True,fig_save_name = 'Loss_MViT')


# In[178]:


# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:08:55 2023
@author: tc922
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import random
def get_epoch(list):
    return list[0]

def result_presentation(log_path_list, log_name, title_name, formal=False, 
        label_list=[],save_df = True,save_fig = False,plot_selection = None,
        fig_save_name = str(int(random.random()*1000))):
    """
    log_path: the list of path where you store the log files
    log_name: the name of same series of logs you want to display (same name or critical name)
    verbose: if verbose is true, the train loss and acc of one model during training will also be printed out.
             This will be used for single model performance display.
    """
    plt.figure(figsize=(10, 10))
    for i, log_path in enumerate(log_path_list):
        #for sets in log_name:
        if True:
            dict_log = {}
            sets = log_name[i]
            log_path_list = []
            test_acc_list = []
            test_loss_list = []
            train_acc_list = []
            train_loss_list = []
            epoch_list = []
            result = []

            for log in os.listdir(log_path):
                if sets in log:
                    log_path_list.append(os.path.join(log_path, log))

            for log in log_path_list:
                f = open(log, 'r')
                line = f.read()
                # when the content of the reading file changes, just change the string_extraction function
                epoch, valid_acc,valid_loss, train_loss, train_acc = string_extraction(line)
                result.append((int(epoch), float(valid_acc),float(valid_loss), float(train_loss), float(train_acc)))

            # sort the order according to epoch value
            result = sorted(result, key=get_epoch)

            for eval_item in result:
                epoch_list.append(eval_item[0])
                test_acc_list.append(eval_item[1])
                test_loss_list.append(eval_item[2])
                train_loss_list.append(eval_item[3])
                train_acc_list.append(eval_item[4])
            max_acc = max(test_acc_list)
            max_acc_idx = test_acc_list.index(max_acc)
            max_acc_epoch = epoch_list[max_acc_idx]
            
            min_loss = min(test_loss_list)
            min_loss_idx = test_loss_list.index(min_loss)
            min_loss_epoch = epoch_list[min_loss_idx]

            plt.rcParams.update({'font.size': 35})
            matplotlib.rcParams['font.family'] = 'Times New Roman'
            if plot_selection == 'Acc' or None:
                if plot_selection == None:
                    plt.subplot(2,1,1)
                plt.plot(epoch_list, train_acc_list, label = label_list[i] + ' Train Acc')
                plt.plot(epoch_list, test_acc_list, label = label_list[i] + ' Validation Acc ')
                best_acc = f'Best {str(max_acc_epoch)}: {str(round(max_acc, 2))}'
                plt.scatter(max_acc_epoch, max_acc, s=150, label=f'Max accuracy: {best_acc}')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
            if plot_selection == 'Loss' or None:
                if plot_selection == None:
                    plt.subplot(2,1,2)
                plt.plot(epoch_list, train_loss_list, label = label_list[i] + ' Train Loss')
                plt.plot(epoch_list, test_loss_list, label = label_list[i] + ' Validation Loss')
                best_loss = f'Best {str(min_loss_epoch)}: {str(round(min_loss, 2))}'
                plt.scatter(min_loss_epoch, min_loss, s=150, label=f'Min loss: {round(min_loss, 4)}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
            if save_df == True:
                dict_log['epoch'] = epoch_list
                dict_log['valid_acc'] = test_acc_list
                dict_log['train_acc'] = train_acc_list
                dict_log['valid_loss'] = test_loss_list
                dict_log['train_loss'] = train_loss_list
                df_log = pd.DataFrame(dict_log)
                df_log.to_csv('train_results/' + label_list[i] + '.csv')

    plt.legend()
    
    if formal == False:
        plt.title(title_name + 'ACC',fontsize = 25)
    
    plt.legend()
    if formal == False:
        plt.title(title_name + 'LOSS',fontsize = 25)
        
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    if save_fig == True:
            plt.savefig('train_results/' + fig_save_name + '.png')
    plt.show()


# In[201]:


model_path = 'models/vit_base_patch16_r224/COVID_balancedpaper1_.pt'
model = torch.load(model_path)
#def model_test(model, dl)

@torch.no_grad()
def test(model,test_dl):
    model.eval()
    model = model.cuda()
    true_count = 0
    test_length_count = 0
    for batch in tqdm(test_dl):
        x, y =batch[0].cuda(), batch[1].cuda()
        out = model(x)
        test_length_count += len(y)
        predict = out.argmax(dim=1).view(-1)
        y = torch.Tensor(y)
        predict = torch.Tensor(predict)
        truth = y==predict
        count = torch.sum(truth == True)
        true_count += count.item()
    print(y)
    print(predict)
    acc = true_count /test_length_count
    return acc


# In[33]:


def string_extraction(txt):
    """
    input format:[epoch] valid_acc []\n valid_loss []\n train_loss []\n train []
    """
    sub_string_list = txt.split('\n')
    ep,valid_acc = sub_string_list[0].split(' valid_acc ')
    _,valid_loss = sub_string_list[1].split(' valid_loss ')
    _,train_loss = sub_string_list[2].split(' train_loss ')
    _,train_acc = sub_string_list[3].split(' train ')
    return ep, valid_acc, valid_loss, train_loss, train_acc

#test_the string_extraction
f = open('logs/vit_base_patch16_r224/COVID_balanced_usual_paper1_4.log','r')
line = f.read()
print(string_extraction(line))
epoch, valid_acc, valid_loss, train_loss, train_acc = string_extraction(line)
dict_log = {'epoch': [],'valid_acc': [], 'valid_loss': [], 'train_acc': [], 'train_loss': []}
dict_log['epoch'].append(epoch)
dict_log['valid_acc'].append(valid_acc)
dict_log['train_acc'].append(train_acc)
dict_log['valid_loss'].append(valid_loss)
dict_log['train_loss'].append(train_loss)
dict_log


# In[29]:


def string_extraction(txt,verbose = True):
    """
    input format:[epoch] Accuracy [test_acc]\n train_loss []\n train []
    """
    sub_string_list = txt.split('\n')
    ep, test_acc = sub_string_list[0].split(' Accuracy ')
        
    if (len(sub_string_list)==2):
        train_acc = sub_string_list[1].split('train ')
        _, train_acc = sub_string_list[1].split('train ')
        train_loss = -1
     
    elif (len(sub_string_list)>2):
        train_loss = sub_string_list[1].split('train_loss ')
        if (len(train_loss)>=2):
            _, train_loss = train_loss       
            
        train_acc = sub_string_list[2].split('train ')
        if (len(train_acc)>=2):
            _, train_acc = sub_string_list[2].split('train ')
    else:
        train_loss = -1
        train_acc = -1
    if verbose == True:
        return ep,test_acc,train_loss,train_acc
    else:
        return ep,test_acc
    
#test_the string_extraction
f = open('logs/1/vit_base_patch16_384_fd/COVID19_4_usual_vit_384px_2e6r12_27.log','r')
line = f.read()
print(string_extraction(line))


# In[14]:


df


# In[222]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

plt.figure(figsize = (15,15))
csv_path = './test_results/Accuracy_models.csv'
df = pd.read_csv(csv_path)

colors = ['red', 'green', 'blue', 'yellow', 'orange','purple','gray','brown']
model_names = ['EfficientNet-B0','EfficieNet-B5','EfficientViT-B3','MViTv2','ViT-Base-patch8','ViT-Base-patch16','ViT-Base-patch32','Our proposed model']
bar_idx = np.arange(len(colors))


plt.barh(bar_idx,df['Accuracy']*100, color = colors)


plt.yticks(np.arange(len(model_names)),model_names)
plt.xlim(92,96)
#plt.xticks()


plt.ylabel('Model type')
plt.xlabel('Accuracy (%)')
#plt.title('Accuracy comparison between models')
matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.show()


# In[201]:


#show the 4 types of images
import matplotlib.pyplot as plt
from PIL import Image
plt.figure()
path_list = ['./COVID-19_Radiography_Dataset/COVID/images/COVID-4.png'
            ,'./COVID-19_Radiography_Dataset/Lung_Opacity/images/Lung_Opacity-4.png'
            ,'./COVID-19_Radiography_Dataset/Normal/images/Normal-4.png'
            ,'./COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-4.png']
label = ['COVID','Lung Opacity','Normal','Viral Pneumonia']
for i,img_path in enumerate(path_list):
    img = Image.open(img_path)
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(left = 0.01,right = 0.9, top = 1.2,bottom = 0.1,wspace= 0.1,hspace = 0.1)
    plt.imshow(img,cmap = 'gray')
    plt.text(x=img.size[1]*0.99,y=img.size[0]*0.95,s = label[i],
            color = 'orange',fontsize=14,ha = 'right',va = 'center',weight = 'bold')
    plt.axis('off')
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.show()


# In[224]:


#display 8 loss 
import matplotlib.pyplot as plt
from PIL import Image
plt.figure(figsize=(200,100))
path_list = ['./train_results/Loss_EfficientNetB0.png',
             './train_results/Loss_EfficientNetB5.png',
             './train_results/Loss_EfficientViT.png',
             './train_results/Loss_MViT.png',
             './train_results/Loss_ViT-8.png',
             './train_results/Loss_ViT-16.png',
             './train_results/Loss_ViT-32.png',
             './train_results/Loss_proposed_model.png']
label = ['EfficientNet-B0','EfficientNet-B5','EfficientViT-B3','MViTv2','ViT-Base-patch8','ViT-Base-patch16','ViT-Base-patch32'
        ,'Our proposed model']
for i,img_path in enumerate(path_list):
    img = Image.open(img_path)
    plt.subplot(2,4,i+1)
    plt.subplots_adjust(left = 0.01,right = 0.1, top = 0.1,bottom = 0,wspace= 0.01,hspace = 0.1)
    plt.imshow(img,cmap = 'gray')
    plt.text(x=img.size[1]*0.5,y=img.size[0]*1.05,s = label[i],
            color = 'black',fontsize=16,ha = 'center',va = 'center',weight = 'bold')
    plt.axis('off')
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.show()


# In[41]:


img.size[1]


# In[231]:


import torch

y_true = torch.tensor([1,0,0,2,1,2,3])
y_pred = torch.tensor([1,0,1,0,2,2,3])

import sklearn 
import numpy as np
from sklearn.metrics import recall_score,precision_score,confusion_matrix,f1_score

method = 'micro'  #how the recall and precision score will be calculated

recall = recall_score(y_true,y_pred,average = method)
precision = precision_score(y_true,y_pred,average = method)
f1 = f1_score(y_true,y_pred,average = method)

cm_prob = confusion_matrix(y_true,y_pred, labels = [0,1,2,3],normalize = 'pred')
cm_num = confusion_matrix(y_true,y_pred, labels = [0,1,2,3],normalize = None)

print(recall)
print(precision)
print(f1)
print(cm_prob)
H,W = cm_prob.shape
class_labels = ['COVID','Lung Opacity','Normal','Pneumonia']


plt.figure(figsize = (20,20))
plt.xticks(np.arange(len(class_labels)), class_labels)
plt.yticks(np.arange(len(class_labels)), class_labels)
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.imshow(cm_prob, interpolation='nearest', cmap=plt.cm.Blues)  # Use a blue color map
for i,cm_value_row in enumerate(cm_num): 
    for j,cm_value in enumerate(cm_value_row):
        cm = str(cm_value)+ '\n' +str(cm_prob[i][j])
        plt.text(j,i,cm,ha = 'center',va = 'center',fontstyle = 'normal')

        
plt.title('Confusion Matrix')
plt.xlabel('Prediction',color = 'black',weight = 'bold',fontsize = '36')
plt.ylabel('Ground Truth',color = 'black',weight = 'bold',fontsize = '36')

plt.colorbar()
plt.show()


# In[316]:


y_pred = torch.tensor([])
y = torch.cat((y_pred,y_true))
y


# In[286]:


np.arange(-0.5,3,0.5)


# In[235]:


import torch
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

factor_a = -1  #weight factor
eta = 1e-5  #keep the numerical stability
img_size = (299,299)
img_path = 'COVID-19_Radiography_Dataset/COVID/images/COVID-1.png'

img_gray = Image.open(img_path).convert('RGB')


img_np = np.array(img_gray)
print(img_np.shape)
midpoint = np.median(img_np)
phi = math.pi*img_np/(2*midpoint)
f1 = factor_a*np.cos(phi)
gamma = 1 + f1 +eta
gamma_correction = 255*((img_np/255)**(1/gamma))


#gamma_correction = np.repeat(gamma_correction[:, :, np.newaxis], 3, axis=2)
print(gamma_correction.shape)
#Image.fromarray(gamma_correction)

gc = torch.from_numpy(gamma_correction)


print(type(gc))
plt.subplot(1,2,1)
plt.imshow(gamma_correction/255)
plt.subplot(1,2,2)
plt.imshow(img_gray)
print(gc.size())
gc = np.asarray(gc)
print(gamma_correction.min())


# In[110]:


gamma_correction.min()


# In[347]:


import torch
from PIL import Image
import numpy as np
path = 'models/EfficientNetb0_r224/COVID19_CXR_4_best__.pt'

from timm.models import create_model
class EfficientNetTl(nn.Module):
    def __init__(self,class_num= 4,bottom_layer_in_features= 1280,model_name = 'efficientnet_b5.sw_in12k'):
        super().__init__()
        #build the pre_trained model efficientnet-b0 from timm library
        self.base_model = create_model(model_name = model_name,pretrained = True)
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
    def forward(self,x):
        x = self.base_model(x)
        return x
#in order to reload the model which you saved and 
    #its structure contains layers where your program do not have, you need to import the self-define layers
model = torch.load(path)  
for n,p in model.named_parameters():
    print(n)
img = Image.open('COVID-44.png').resize((224,224)).convert('RGB')
img = np.asarray(img,dtype = np.float32)
img = torch.tensor(img).permute(2,0,1).view(1,3,224,224)
y_pred = model(img)


# In[352]:


a = ['a','b','c']
b = ['hello','world','!']
dt = {'x': a, 'y':b}
pd.DataFrame(dt)


# In[134]:


import timm 

timm.list_models(pretrained= True)[200:]


# In[70]:


from ignite.metrics import ConfusionMatrix
from ignite.engine import *

# create default evaluator for doctests
def eval_step(engine, batch):
    return batch



def confusion_matrix(class_nums,out,ground_truth, softmax_out = True):
    """
    output the confusion matrix in metrics form
    params:
    class_num: the total number of class types
    out: the output from model, when softmax_out = True, 
        it should be shaped like: (batch_size,class_nums).
        When False, it should be (batch_size)
    """
    batch_size = int(out.size(0))
    if softmax_out == True:
        #convert the softmax_output from 2d into 1d
        predictions = torch.argmax(out,dim= 1)
        #construct a all zeor metrics, which have the same shape as input
        one_hot =  torch.zeros_like(out)  
    else:
        predicitons = out
        # just for practice zeros and zeros_like
        one_hot = torch.zeros([batch,class_nums])
    
    one_hot[torch.arange(batch_size), predictions] = 1 #row and column indicators are both metrics
    
    # from https://pytorch.org/ignite/generated/ignite.metrics.confusion_matrix.ConfusionMatrix.html
    default_evaluator = Engine(eval_step)
    
    #construct confusion matrix object
    metrics = ConfusionMatrix(num_classes = class_nums)
    metrics.attach(default_evaluator, 'cm')
    state = default_evaluator.run([[one_hot, ground_truth]])
    print(state.metrics['cm'])
    return state.metrics['cm']


# In[73]:


get_ipython().system('pip install plotnine')


# In[42]:


import torch

# Example softmax output tensor
softmax_outputs = torch.tensor([[0.1, 0.3, 0.6],
                                [0.8, 0.1, 0.1],
                                [0.2, 0.5, 0.3]])

# Find the indices of the max elements
max_indices = torch.argmax(softmax_outputs, dim=1)

# Create a zero tensor of the same shape
one_hot = torch.zeros_like(softmax_outputs)

# Set the max elements to 1
one_hot[torch.arange(softmax_outputs.size(0)), max_indices] = 1

print(softmax_outputs.size(0))


# In[32]:


import tensorflow as tf
#tf.config.list_physical_devices('GPU')
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
print(tf.config.list_physical_devices('GPU'))


# In[15]:


a=[1,2,3,4,5,5,5,6,6]
a.index(6)


# In[19]:


def path_file_reading(path):
    line_list = [line for line in open(path,'r')]
    path_list = []
    label_list = []
    for line in line_list:
        path, label = line.strip().split('  ')
        path_list.append(path)
        label_list.append(label)
    return path_list,label_list
p,l = path_file_reading('./processed/trainval.txt')
print(len(p))
print(len(l))


# In[76]:


class MyClass:
    def __init__(self):
        self.a =1
for i in range(4):
    c = MyClass()
    print(i)


# In[59]:


from timm.models import create_model
import timm
a = timm.list_models('*densnent*')
a
avail_pretrained_models = timm.list_models(pretrained=True)
len(avail_pretrained_models), avail_pretrained_models
model = create_model(model_name = 'efficientnet_b0.ra_in1k')
print(model)


# In[25]:


from timm.models import create_model
import torch.nn as nn
class EfficientNetTl(nn.Module):
    def __init__(self,class_num= 4,bottom_layer_in_features= 1280):
        super().__init__()
        self.base_model = create_model(model_name = 'efficientnet_b0.ra_in1k',pretrained = True)
        # bottom_layer is the position where we want to add after the base_model (pretrained)
        self.bottom_layers = nn.Sequential(
            #BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)  
            #for the nn, the momentum is reverse to tensorflow: 1- 0.99 = 0.01
            nn.BatchNorm2d(bottom_layer_in_features,momentum= 0.01,eps= 1e-3), 
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
        self.base_model.classifier = self.bottom_layers
        self.model = self.base_model
    def forward(self,x):
        x = self.model(x)
        return x


# In[26]:


model = EfficientNetTl()
#print(model)
model


# In[71]:


img_size = (224,224)
train_dl,test_dl =get_split('./processed',img_size)


# In[70]:


from PIL import Image
import torch
from torchvision import transforms
def open_img(img_path, img_size):
    """
    get the img path and return the img pixel value
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size)
    return img  
class ImgFileLoader:
    def __init__(self,file_path,img_size,transform,imgloader=open_img):
        self.img_size = img_size
        self.imgloader = imgloader
        self.transform = transform
        f = open(file_path,'r')
        lines = [line for line in f]
        self.img_list = []
        for line in lines:
            img_path, label = line.strip().split('  ')
            self.img_list.append((img_path,label))
    def __getitem__(self,idx):
        img_path, label = self.img_list[idx]
        img = self.imgloader(img_path,self.img_size)
        img = self.transform(img)
        return img, int(label)
    def __len__(self):
        return len(self.img_list)
def get_split(root_path,img_size=(224,224),batch_size=16,evaluation=True):
    train_path = root_path + '/train.txt'
    valid_path = root_path + '/val.txt'
    trainval_path = root_path + '/trainval.txt'
    test_path  = root_path + '/test.txt'
    if evaluation == True:
        train_path = trainval_path
    else:
        test_path = valid_path
    transform = transforms.Compose([transforms.ToTensor()])
    train_dl = torch.utils.data.DataLoader(ImgFileLoader(train_path,img_size,transform),
        batch_size = batch_size,shuffle = True, num_workers = 0, 
        drop_last = True,pin_memory = True)
    
    test_dl = torch.utils.data.DataLoader(ImgFileLoader(test_path,img_size,transform),
        batch_size = batch_size,shuffle = True, num_workers = 0, 
        drop_last = True,pin_memory = True)
    
    return train_dl,test_dl
    
    


# In[3]:


import matplotlib.pyplot as plt
import os

result_presentation(['./logs/vit_full_tain_patch8','./logs/vit_full_tain_patch8_mask','./logs/vit_full_train_patch16',
                     './logs/vit_full_tain_patch16_7partial_mask','./logs/vit_full_tain_patch16_mask',
                     './logs/vit_full_tain_patch32','./logs/vit_full_tain_patch32_mask'],
                    ['_usual_'],
                   'Vision Transformer Test Accuracy with COVID-19 CXR Dataset\nEarly stopping after 30 epochs without  0.0001 change',
                    formal = True, verbose = True
                   label_list = ['patch8','patch8 (with masked dataset)','patch16','patch16 (with 70% masked dataset)',
                                 'patch16 (with masked dataset)','patch32','patch32 (with masked dataset)'])

result_presentation(['./logs/EfficientNetb0'],
                    ['EfficientNetb0__'],
                   'EfficientNetb0 Test Accuracy',
                    formal = True, verbose = True
                   label_list = ['EfficientNet-B0'])

