import torch
import torchvision
torch.manual_seed(7)
from torch import  nn,optim


def load_img_shape(model_name):
  ''' Get default input shape for pre-defined models.'''
  if model_name == 'inceptionv3':
    return (1,3,299,299)
  else:
    return (1,3,224,224)


def load_model(model_name, full_training):
  '''Load pre-defined model with/ without pre-trained weights.'''
  if model_name == "resnet50":
    model = torchvision.models.resnet50(pretrained=not(full_training), progress=True)

    if full_training == False:
      for param in model.parameters():
        param.requires_grad = False

    model.fc= nn.Linear(2048, 10, bias=True)
    return model

  if model_name == "inceptionv3":
    model = torchvision.models.inception_v3(pretrained=not(full_training), progress=True)

    if full_training == False:
      for param in model.parameters():
        param.requires_grad = False

    model.AuxLogits.fc = nn.Linear(768, 10,bias=True)
    model.fc = nn.Linear(2048, 10, bias=True)
    return model

  if model_name == "densenet121":
    model = torchvision.models.densenet121(pretrained=not(full_training), progress=True)

    if full_training == False:
      for param in model.parameters():
        param.requires_grad = False

    model.classifier=nn.Linear(1024,10, bias=True)
    return model


def load_optimizer(model_param, opt_name, l_rate):
  '''Returns an optimizer to train model.'''
  if opt_name == "adam":
    optimizer = optim.Adam(model_param,lr=l_rate)
    return optimizer
  elif opt_name == "rmsprop":
    optimizer = optim.RMSprop(model_param,lr=l_rate)
    return optimizer
  elif opt_name == "nadam":
    optimizer = optim.NAdam(model_param,lr=l_rate)
    return optimizer
  elif opt_name == "sgd":
    optimizer = optim.SGD(model_param,lr=l_rate)
    return optimizer
