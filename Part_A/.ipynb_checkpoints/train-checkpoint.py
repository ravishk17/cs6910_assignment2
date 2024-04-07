#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Q2


# In[1]:


#!pip install --upgrade wandb
import cv2
import glob
import random
import torch
import torchvision
torch.manual_seed(7)
torch.cuda.empty_cache()
from torch import  nn,optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.common import flatten
from tqdm import tqdm 
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import wandb
from wandb.keras import WandbCallback


# In[ ]:


#get_ipython().system('wget --header="Host: storage.googleapis.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9" --header="Referer: https://wandb.ai/" "https://storage.googleapis.com/wandb_datasets/nature_12K.zip" -c -O \'nature_12K.zip\'')


# In[ ]:


#get_ipython().system('unzip "nature_12K.zip"')


# In[2]:


import os
def check_os():
    if os.name == 'nt':
        return 'Windows'
    else:
        return 'Linux'
operatingSystem = check_os()
# print(operatingSystem)


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


def forwardSlash(l):
    a=[]
    for ele in l:
        a.append(ele.replace('\\','/'))
    return a
    
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
    


# In[7]:


from pytorch_lightning import LightningModule
class ConvolutionBlocks(LightningModule):
    def __init__(self, activation, batch_norm, size_filters, filter_organization, number_filters):
        super(ConvolutionBlocks, self).__init__()
        
        self.activation=activation
        self.num_filters=[number_filters]
        self.batch_norm=batch_norm
        
        for i in range(1,5):
          self.num_filters.append(int(self.num_filters[i-1]*filter_organization))
            
        if(self.batch_norm):  
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=self.num_filters[0],kernel_size=size_filters[0],stride=(1, 1),padding=(1, 1),bias=False),
                nn.BatchNorm2d(self.num_filters[0]),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                
                nn.Conv2d(in_channels=self.num_filters[0],out_channels=self.num_filters[1],kernel_size=size_filters[1],stride=(1, 1),padding=(1, 1),bias=False),
                nn.BatchNorm2d(self.num_filters[1]),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
                nn.Conv2d(in_channels=self.num_filters[1],out_channels=self.num_filters[2],kernel_size=size_filters[2],stride=(1, 1),padding=(1, 1),bias=False),
                nn.BatchNorm2d(self.num_filters[2]),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
                nn.Conv2d(in_channels=self.num_filters[2],out_channels=self.num_filters[3],kernel_size=size_filters[3],stride=(1, 1),padding=(1, 1),bias=False),
                nn.BatchNorm2d(self.num_filters[3]),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
                nn.Conv2d(in_channels=self.num_filters[3],out_channels=self.num_filters[4],kernel_size=size_filters[4],stride=(1, 1),padding=(1, 1),bias=False),
                nn.BatchNorm2d(self.num_filters[4]),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=self.num_filters[0],kernel_size=size_filters[0],stride=(1, 1),padding=(1, 1),bias=False),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                
                nn.Conv2d(in_channels=self.num_filters[0],out_channels=self.num_filters[1],kernel_size=size_filters[1],stride=(1, 1),padding=(1, 1),bias=False),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
                nn.Conv2d(in_channels=self.num_filters[1],out_channels=self.num_filters[2],kernel_size=size_filters[2],stride=(1, 1),padding=(1, 1),bias=False),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
                nn.Conv2d(in_channels=self.num_filters[2],out_channels=self.num_filters[3],kernel_size=size_filters[3],stride=(1, 1),padding=(1, 1),bias=False),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
                nn.Conv2d(in_channels=self.num_filters[3],out_channels=self.num_filters[4],kernel_size=size_filters[4],stride=(1, 1),padding=(1, 1),bias=False),
                self.activation,
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
            

    def forward(self, x):
        return self.conv_layers(x)
     


# In[8]:


class Model(nn.Module):
    def __init__(self, number_initial_filters , neurons_in_dense_layer, image_shape,dropout , activation, batch_norm, size_filters, filter_organization):
        super().__init__()
        
        if(activation=='relu'):
            self.activation = nn.ReLU()
        elif(activation=='gelu'):
            self.activation = nn.GELU()
        elif(activation=='silu'):
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Mish()
            

        self.conv_blocks = ConvolutionBlocks(self.activation, batch_norm, size_filters, filter_organization, number_initial_filters)
        sz=self.conv_blocks(torch.zeros(*(image_shape))).data.shape
        fc1_in_channels = sz[1] * sz[2] * sz[3]
        self.output= nn.Linear(neurons_in_dense_layer,10,bias=True)   

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc1_in_channels,neurons_in_dense_layer,bias=True),
            self.activation,
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.dense_layers(x)
        x = F.softmax(self.output(x),dim=1) #Applying softmax across rows
        return x


# In[9]:


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
 


# In[10]:


def cal_train(data,device,model,criterion,optimizer):
    train_loss=0
    corr_train=0
    tot_sample=0
    for idd,(x,y) in enumerate(tqdm(data)):
        x=x.to(device=device)
        yt=y.to(device=device)
        out = model(x)
        loss=criterion(out,yt)
        train_loss+=loss.item()
        _,pred=out.max(1)
        corr_train+=(pred==yt).sum()
        tot_sample+=pred.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return train_loss,tot_sample,corr_train


def train(args,wandb):
    torch.cuda.empty_cache()
    image_shape = (1,3,224,224) # All the images of dataset will get resized to this image shape
    test_data_path = 'inaturalist_12K/val/'
    train_data_path = 'inaturalist_12K/train/'
     
    wandb.run.name = 'ep-'+str(args.epochs)+'-lr-'+str(args.learning_rate)+'-bs-'+str(args.batch_size)+'-act-'+str(args.activation)+'-filter_sizes-'+str(args.size_filters) +'-ini_filt'+str(args.number_initial_filters)+'-n_d-'+str(args.neurons_in_dense_layer)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #CNN Model creation
    model = Model(args.number_initial_filters ,args.neurons_in_dense_layer,image_shape,args.dropout, args.activation, args.batch_norm, args.size_filters, args.filter_organization).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train Network
    for epoch in range(args.epochs):
        model.train()
        test_loader = create_data("test",test_data_path,args.data_aug, image_shape[2:], args.batch_size)
        train_loader, valid_loader = create_data("train",train_data_path,args.data_aug,image_shape[2:], args.batch_size)
        
        train_loss,total_samples,train_correct = cal_train(train_loader,device,model,criterion,optimizer)
        # Calculating training accuracy and training loss
        train_loss /= total_samples
        train_acc = round((train_correct / total_samples).item()  * 100, 4)
        
       
        
        # Calculating accuracy and loss for test and validataion data
        val_acc, val_loss = evaluate(device, valid_loader, model)
        test_acc, test_loss = evaluate(device, test_loader, model)

        
        #logging wandb data
        wandb.log(
          {'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
        )
        print('\nEpoch ', epoch, 'train_acc', train_acc, 'val_acc', val_acc, 'test_acc', test_acc, 'train_loss', train_loss, 'val_loss', val_loss, 'test_loss', test_loss) 


# In[11]:


# wandb.login(key = "67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e")
# In[12]:


def spawn_fn(args):
    with wandb.init(project="Assignment2_kaggle", entity="cs23m055") as run:
        # config = wandb.config
        train(args)


# In[ ]:


# wandb.agent(sweep_id, function=spawn_fn, count=5)




# sweep_id = wandb.sweep(sweep_config, project="Assigment2_kaggle", entity="cs23m055")
# print('sweep_id: ', sweep_id)
# args = {
#         "batch_norm": True,
#         "neurons_in_dense_layer": 1024,
#         "epochs" : 5,
#         "batch_size": 32,
#         "data_aug": False,
#         'size_filters':[7,5,5,3,3],
#         'filter_organization': 1,
#         'number_initial_filters': 128,
#         'activation': 'mish',
#         'learning_rate': 0.00001,
#         "dropout": 0.2
#     }
import argparse
parser = argparse.ArgumentParser(description="CNN")

parser.add_argument('-wp','--wandb_project', default="assignment2_kaggle", required=False,metavar="", type=str, help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we','--wandb_entity', default="cs23m055", required=False,metavar="", type=str, help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-bn','--batch_norm', default=True, required=False,metavar="", type=bool,choices= [True,False], help=' choices: [True,False]')
parser.add_argument('-ndl','--neurons_in_dense_layer', default=1024, required=False,metavar="", type=int, help='number of neurons in the dense layer')
parser.add_argument('-e','--epochs', default=5, required=False,metavar="", type=int, help='number of epochs')
parser.add_argument('-b','--batch_size', default=32, required=False,metavar="", type=int, help='Batch size used to train neural network')
parser.add_argument('-aug','--data_aug', default=False, required=False,metavar="", type=bool, help='Data augmentation [True/False]')
parser.add_argument('-fo','--filter_organization', default=1, required=False,metavar="", type=float, choices = [1, 2, 0.5],help='Filter organization')
parser.add_argument('-nif','--number_initial_filters', default=128, required=False,metavar="", type=int,help='number initial filters')
parser.add_argument('-a','--activation', default='mish', required=False,metavar="", type=str, choices = ['relu', 'mish', 'gelu', 'silu'],help='activation function')
parser.add_argument('-lr','--learning_rate', default=0.00001, required=False,metavar="", type=float, help='Learning rate used to optimize model parameter')
parser.add_argument('-d','--dropout', default=0.2,required=False,metavar="",type=float,help='Dropout to take care overfitting')
parser.add_argument('-sf','--size_filters',default=[7,5,5,3,3],nargs=5 ,required=False,metavar="",type=int,help=['Size of the filters for the 5 convolution layers'])

args = parser.parse_args()
wandb.login(key = "67fcf10073b0d1bfeee44a1e4bd6f3eb5b674f8e")
wandb.init(project=args.wandb_project,entity=args.wandb_entity)
train(args,wandb)
wandb.run.save("end")
wandb.run.finish()