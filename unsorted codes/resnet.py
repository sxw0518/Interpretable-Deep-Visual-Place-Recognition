# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 12:02:42 2018

@author: Xiangwei Shi
"""
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

# Image dataset direction
root_dir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/xiangweishi/balanced_pytorch_geo_classification'
train_dir = os.path.join(root_dir,'train')
valid_dir = os.path.join(root_dir,'valid')
test_dir = os.path.join(root_dir,'test')

# Image preprocessing
transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
        ])

# Dealing with balanced data (sampling images with different weights)
#(45600 training images of Pittsburgh, 45588 training images of Tokyo)

# Image dataset
train_dataset = datasets.ImageFolder(
        train_dir,
        transform = transform
        )
valid_dataset = datasets.ImageFolder(
        valid_dir,
        transform = transform
        )
test_dataset = datasets.ImageFolder(
        test_dir,
        transform = transform
        )
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 50,
        shuffle = True,
        num_workers = 2
        )
valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = 50,
        shuffle = True,
        num_workers = 2
        )
test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 50,
        shuffle = True,
        num_workers = 2
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# building model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512,2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=1e-5)

class initialization(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

print('Start')

loss_model = []
loss_epoch = initialization()
validation_error = []
training_error = []
test_error = []
best_validation_error = 1
for epoch in range(30):
	print('epoch:')
	print(epoch+1)
	model.train()
	for images,labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		loss_epoch.update(loss.item(),images.size(0))
	loss_model.append(loss_epoch.avg)
	print('training_loss:')
	print(loss_model[epoch])
	#Adjust the learning rate
	lr = 1e-4 * (0.1**(epoch//10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	print('learning_rate:')
	print(optimizer.param_groups[0]['lr'])
	#Record the training error
	print('Training error:')
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data,1)
			total = total + labels.size(0)
			correct = correct + (predicted == labels).sum().item()
	training_error.append(1-float(correct)/float(total))
	print(correct,total,training_error[epoch])
	#Validate the hyperparameter
	print('Validation error:')
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for images, labels in valid_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data,1)
			total = total + labels.size(0)
			correct = correct + (predicted == labels).sum().item()
	validation_error.append(1-float(correct)/float(total))
	print(correct,total,validation_error[epoch])
	if best_validation_error > validation_error[epoch]:
		best_validation_error = validation_error[epoch]
		best_model = model
		best_epoch = epoch + 1
		best_lr = optimizer.param_groups[0]['lr']
	if epoch%5 == 0:
		print('Test error:')
		correct = 0
		total = 0
		model.eval()
		with torch.no_grad():
			for images, labels in test_loader:
				images = images.to(device)
				labels = labels.to(device)
				outputs = model(images)
				_, predicted = torch.max(outputs.data,1)
				total = total + labels.size(0)
				correct = correct + (predicted == labels).sum().item()
		test_error.append(1-float(correct)/float(total))
		print(correct,total,test_error[epoch//5])

torch.save(best_model.state_dict(),'Balanced_epoch[%d]_learning_rate[%.4f]_resnet_2.pth'%(best_epoch,best_lr))

File = h5py.File('resnet_train.h5','w')
File['data'] = training_error
File.close()
File = h5py.File('resnet_valid.h5','w')
File['data'] = validation_error
File.close()
File = h5py.File('resnet_test.h5','w')
File['data'] = test_error
File.close()