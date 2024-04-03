import torch
import torch.nn as nn
import torch.nn.functional as F


def get_output_shape(model, image_shape):
    return model(torch.zeros(*(image_shape))).data.shape


class ConvolutionBlocks(nn.Module):
    ''' Defines 5 convolution layers used in a CNN
    '''
    def __init__(self,in_channels,num_filters,filter_size,activation,neurons_dense):
        super().__init__()
        self.activation=activation
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=num_filters,kernel_size=filter_size,stride=(1, 1),padding=(1, 1),bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=filter_size,stride=(1, 1),padding=(1, 1),bias=False)
        self.conv3 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=filter_size,stride=(1, 1),padding=(1, 1),bias=False)
        self.conv4 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=filter_size,stride=(1, 1),padding=(1, 1),bias=False)
        self.conv5 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=filter_size,stride=(1, 1),padding=(1, 1),bias=False)
        self.pool  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
      x=self.pool(self.activation(self.conv1(x)))
      x=self.pool(self.activation(self.conv2(x)))
      x=self.pool(self.activation(self.conv3(x)))
      x=self.pool(self.activation(self.conv4(x)))
      x=self.pool(self.activation(self.conv5(x)))
      return x


class Model(nn.Module):
    ''' Defines a small CNN model consisting of 5 convolution layers, where each 
        convolution layer is followed by a ReLU activation and a max pooling layer.
    '''
    def __init__(self, in_channels, num_filters, filter_size, activation, neurons_dense, image_shape):
        super().__init__()
        self.activation=activation
        self.conv_blocks = ConvolutionBlocks(in_channels, num_filters, filter_size, activation, neurons_dense)
        sz = get_output_shape(self.conv_blocks, image_shape)  # automatically infer fc input size
        fc1_in_channels = sz[1] * sz[2] * sz[3]
        self.fc1   = nn.Linear(fc1_in_channels,neurons_dense,bias=True)  
        self.output= nn.Linear(neurons_dense, 10, bias=True)   
    
    def forward(self, x):
        x = self.conv_blocks(x) 
        x = self.activation(self.fc1(x.reshape(x.shape[0],-1)))
        x = F.softmax(self.output(x),dim=1) #Applying softmax across rows
        return x


if __name__ == '__main__':
    # Sample Hyperparameters
    activation = nn.ReLU()
    num_filters = 16
    filter_size = 5
    in_channels = 3
    neurons_dense = 32
    # Instantiate Model
    model = Model(in_channels,num_filters,filter_size,activation,neurons_dense, (1, 3, 100, 100))
    print('Model: ', model)
    sample_input = torch.randn(64,3,100,100)
    print(model(sample_input).shape)