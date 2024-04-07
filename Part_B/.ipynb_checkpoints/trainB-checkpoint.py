#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --upgrade wandb
import cv2
import torch
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision.models import googlenet
import glob
from pandas.core.common import flatten
import random
from tqdm import tqdm 
import numpy as np


# In[2]:


import os
def check_os():
    if os.name == 'nt':
        return 'Windows'
    else:
        return 'Linux'
operatingSystem = check_os()


# In[3]:


#Custom dataset class for inaturalist dataset
from PIL import Image

def numpy_to_pil(image):
    return Image.fromarray(np.uint8(image)).convert('RGB')
    
class iNaturalist(Dataset):
    def __init__(self, image_paths, class_to_idx, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx= class_to_idx
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        # print(image_filepath)
        if(operatingSystem=='Windows'):
            label = image_filepath.split('\\')[-2]
        else:
            label = image_filepath.split('/')[-2]
        # print(label)
        label = self.class_to_idx[label]
        
        PIL_image = numpy_to_pil(image)
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        PIL_image = self.transform(PIL_image)

        return PIL_image, label


# In[4]:


def create_data(data_type, data_path,  data_aug, image_shape, b_size):
    #Defining transformations when data_aug=True  [used when data_type='train' and data_aug=True]
    if(data_aug):
        transforms = T.Compose([T.Resize((image_shape)),
                              T.RandomRotation(degrees=15),
                              T.RandomHorizontalFlip(p=0.5),
                              T.RandomGrayscale(p=0.2),
                              T.ToTensor()])
    else:
    #Defining transformations when data_aug=False
        transforms = T.Compose([T.Resize((image_shape)),
                               T.ToTensor()])
    image_paths=[] # List to store image paths
    classes= [] # List to  store class values
    #get all the paths from data_path and append image paths and class to to respective lists
    cnt=0
    for curr_data_path in glob.glob(data_path + '/*'):
        if(operatingSystem=='Windows'):
            classes.append(curr_data_path.split('\\')[-1])
        else:
            classes.append(curr_data_path.split('/')[-1])
        image_paths.append(glob.glob(curr_data_path+'/*'))
    image_paths = list(flatten(image_paths))
    
    #Creating dictionary for class indexes
    idx_to_class={}
    class_to_idx={}
    for i in range(len(classes)):
        idx=i
        cls=classes[i]
        idx_to_class[idx]=cls
        class_to_idx[cls]=idx
    
    if (data_type != 'test'):
        random.shuffle(image_paths)
        # 80% training data and 20% validation data
        train_image_paths = image_paths[:int(0.8*len(image_paths))]
        valid_image_paths = image_paths[int(0.8*len(image_paths)):] 
        #Using custom class for getting train and validation dataset
        
        train_dataset = iNaturalist(train_image_paths,class_to_idx,transforms)
        validation_dataset = iNaturalist(valid_image_paths,class_to_idx,transforms)  
          
        #using Dataloader to load train and valid dataset according to batch size
        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=b_size, shuffle=True)
        return train_loader,validation_loader
    else:
        #Using custom class for getting test dataset
        transforms = T.Compose([T.Resize((image_shape)),
                               T.ToTensor()])
        test_dataset= iNaturalist(image_paths,class_to_idx,transforms)
        #using Dataloader to load test dataset according to batch size
        test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=True)
        return test_loader


# In[5]:


def cal(data,device,model,criterion):
    loss=0
    corr_sample=0
    tot_sample=0
    for x,y in tqdm(data,total=len(data)):
        x=x.to(device=device)
        y=y.to(device=device)
        out = model(x)
        loss+=criterion(out,y).item()
        _,pred=out.max(1)
        corr_sample+=(pred==y).sum().item()
        tot_sample+=pred.size(0)
    return corr_sample,tot_sample,loss
    
def evaluate(device, loader, model):
    ''' Function to calculate accuracy to see performance of our model '''
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct_samples,total_samples,loss=0,0,0
    with torch.no_grad():
        correct_samples_,total_samples_,loss_ = cal(loader,device,model,criterion)
        correct_samples+=correct_samples_
        total_samples+=total_samples_
        loss+=loss_
           
    acc = round((correct_samples / total_samples) * 100, 4)
    return acc, loss/total_samples 
 


# In[6]:


def cal_train(data,device,model,criterion,optimizer):
    train_loss=0
    corr_train=0
    tot_sample=0
    for idd,(x,y) in enumerate(tqdm(data)):
        x=x.to(device=device)
        yt=y.to(device=device)
        optimizer.zero_grad()
        out = model(x)
        loss=criterion(out,yt) 
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        _, pred = torch.max(out, 1) 
        corr_train+=(pred==yt).sum()
        tot_sample+=pred.size(0)
    return train_loss,tot_sample,corr_train


# In[8]:


from torchvision.models import googlenet
model = googlenet(pretrained=True)


# In[9]:


def train_fine_tune(model,epochs=5,strategy=0):
    if strategy == 0:
        # Freeze all layers except the final classification layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
    elif strategy == 1:
        # Unfreeze all layers and train the entire model
        for param in model.parameters():
            param.requires_grad = True
            
    elif strategy == 2:
        # Unfreeze and fine-tune only a subset of layers (e.g., only top layers)
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.inception5b.parameters():
            param.requires_grad = True
            
    torch.cuda.empty_cache()
    image_shape = (1,3,224,224) # All the images of dataset will get resized to this image shape
    test_data_path = 'inaturalist_12K/val/'
    train_data_path = 'inaturalist_12K/train/'
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.00001)
    for epoch in range(epochs):
        model.train()
        # test_loader = create_data("test",test_data_path,args.data_aug, image_shape[2:], args.batch_size)
        train_loader, valid_loader = create_data("train",train_data_path,False,image_shape[2:], 64)
        
        train_loss,total_samples,train_correct = cal_train(train_loader,device,model,criterion,optimizer)
        train_loss /= total_samples
        train_acc = round((train_correct / total_samples).item()  * 100, 4)
        
       
        
        # Calculating accuracy and loss for test and validataion data
        # val_acc, val_loss = evaluate(device, valid_loader, model)
        # test_acc, test_loss = evaluate(device, test_loader, model)
        print('\nEpoch ', epoch+1, 'train_acc', train_acc, 'train_loss', train_loss)
        



# In[ ]:


model = googlenet(weights=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # Number of classes in iNaturalist dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
epochs = 5

import argparse
parser = argparse.ArgumentParser(description = "Fine tuning on google net model")
parser.add_argument('-s','--strategy',default=1, required=False,metavar="",type=int,help='[0:freeze_all, 1:fine_tuning_all, 2:layer_wise_fine_tuning]')
args = parser.parse_args()
train_fine_tune(model,epochs, args.strategy)







