#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Q1


# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
import random

class CNN(LightningModule):
    def __init__(self,in_channels,num_filters_conv, filter_sizes_conv, num_filters_dense ,activation='relu' ):
        super(CNN, self).__init__()
        
        # Select the activation function
        if activation == 'relu':
            self.acitvation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation  = nn.ELU()
        else:
            self.activation = nn.SiLU()
            
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters_conv[0], kernel_size=filter_sizes_conv[0],stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=num_filters_conv[0], out_channels=num_filters_conv[1], kernel_size=filter_sizes_conv[1],stride =1 , padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=num_filters_conv[1], out_channels=num_filters_conv[2], kernel_size=filter_sizes_conv[2],stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=num_filters_conv[2], out_channels=num_filters_conv[3], kernel_size=filter_sizes_conv[3],stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=num_filters_conv[3], out_channels=num_filters_conv[4], kernel_size=filter_sizes_conv[4],stride=1, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_filters_conv[-1]*3*3, out_features=10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x

# Assuming the use of PyTorch Lightning for training setup
class LitCNN(CNN):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
in_channels = 3  # For RGB images
num_filters_conv=[32, 64, 128, 256, 512]
filter_sizes_conv=[3, 3, 3, 3, 3]
num_filters_dense=[1024]
activation = random.choice(['relu','tanh','sigmoid','gelu', 'elu', 'silu'])
model = CNN(in_channels,num_filters_conv, filter_sizes_conv,num_filters_dense,activation)
print(model)
# Note: You'll need to define your data loaders and training loop.


# In[ ]:




