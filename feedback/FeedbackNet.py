import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F

class FeedbackNet(nn.Module):
    def __init__(self):
        super(FeedbackNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,256,256)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,256,256)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,256,256)
        self.relu1=nn.ReLU()
        #Shape= (256,12,256,256)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,128,128)
        

        
        self.fc=nn.Linear(in_features=128 * 128 * 12,out_features=1)
        self.sig=nn.Sigmoid()
        
        
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)

            
        #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,12*128*128)
            
            
        output=self.fc(output)
        output=self.sig(output)
            
        return output








