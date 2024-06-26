#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from tqdm import tqdm
from feedback.FeedbackNet import FeedbackNet


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Transforms
transformer=transforms.Compose([
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
])


#Dataloader

#Path for training and testing directory

parser = ArgumentParser()
parser.add_argument("--train_path", help="path to the train folder")
parser.add_argument("--test_path", help= "path to the test folder")

opt = parser.parse_args()

train_path = opt.train_path
test_path = opt.test_path


train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=64, shuffle=True
)


#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)



        
        
model=FeedbackNet().to(device)


#Optmizer and loss function
optimizer=Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
loss_function=nn.BCELoss()

num_epochs=10


#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

print(train_count,test_count)


print("r")
#Model training and saving best model

best_accuracy=0.0

for epoch in tqdm(range(num_epochs)):
    
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            labels = labels.reshape((64, 1))
            labels = labels.to(torch.float32)
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        #_,prediction=torch.max(outputs.data,1)
        prediction = outputs >0.5
        train_accuracy+=int(torch.sum(prediction==labels.data))
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            #print(labels.shape)
            labels=Variable(labels.cuda())
            labels = labels.reshape((64, 1))
            labels = labels.to(torch.float32)
            
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        
        
        #_,prediction=torch.max(outputs.data,1)
        prediction = outputs >0.5
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/test_count
    
    print("r")
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    print("r")
    
    #Save the best model
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint_vox_fomm_kernel5.model')
        best_accuracy=test_accuracy