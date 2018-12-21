import os
import h5py
import torch
import torch.cuda
import torch.optim
import torch.nn as nn
import argparse
from data import *
from networks import *

class calculation(object):
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self,val,n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum/self.count

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_path',type=str,help='Input Image Path')
	parser.add_argument('--model_path',type=str,help='Output Path')
	parser.add_argument('--epoch',type=int,default=30,help='Number of Total Epochs')
	parser.add_argument('--height',type=int,default=224,help='Height of Resized Image')
	parser.add_argument('--width',type=int,default=224,help='Width of Resized Image')
	parser.add_argument('--batch_size',type=int,default=50,help='Batch Size')
	parser.add_argument('--lr',type=float,default=1e-4,help='Initial Learning Rate')
	parser.add_argument('--weight_decay',type=float,default=1e-5,help='Weight Decay')
	parser.add_argument('--model',type=str,help='Model Name')
	parser.add_argument('--classes',type=int,default=2,help='Number of Classes')
	parser.add_argument('--pretrained',action='store_true',default=false,help='Use Pretrained Model')
	parser.add_argument('--test_frequency',type=int,default=1,help='Frequency of Test')
	parser.add_argument('--lr_frequency',type=int,default=10,help='Frequency of Changing Learing Rate')
	args = parser.parse_args()
	return args

def train_model(model,train_loader,optimizer,criterion,device,training_loss,training_acc):
	# switch to train model
	model.train()
	loss_epoch = calculation()
	for images,labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		ouputs = model(images)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		loss_epoch.update(loss.item(),images.size(0))
	training_loss.append(loss_epoch.avg)
	print('training loss:',loss_epoch.avg)
	# switch to evaluate model
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images,labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	training_acc.append(float(correct)/float(total))
	print('Training accuracy:',float(correct)/float(total))

def validate_model(model,valid_loader,optimizer,criterion,device,validation_acc):
	# switch to evaluate model
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images,labels in valid_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	validation_acc.append(float(correct)/float(total))
	print('Validation accuracy:',float(correct)/float(total))

def test_model(model,test_loader,optimizer,criterion,device,test_acc):
	# switch to evaluate model
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images,labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	test_acc.append(float(correct)/float(total))
	print('Test accuracy:',float(correct)/float(total))


def main():
	# catching arguments
	args = get_args()
	if args.model == None:
		print('Please enter the name of model that you want to use!')
		exit()
	# establish model dictionary
	model_names = sorted(name for name in networks.__dict__
		if name.islower() and not name.startswith("__")
		and callable(networks.__dict__[name]))
	# set pathes of images
	train_dir = os.path.join(args.image_path,'train')
	valid_dir = os.path.join(args.image_path,'valid')
	test_dir = os.path.join(args.image_path,'test')
	train_loader = data.dataloader.loader(train_dir,args.batch_size,args.height,args.width)
	valid_loader = data.dataloader.loader(valid_dir,args.batch_size,args.height,args.width)
	test_loader = data.dataloader.loader(test_dir,args.batch_size,args.height,args.width)
	# if cuda is available, use gpu, else use cpu
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# build model
	model = model_names[args.model](pretrained=args.pretrained,num_classes=args.classes).to(device)
	# set criterion and optimizer
	criterion = nn.CrossEntropyLoss()
	parameters = filter(lambda p:p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters,lr=args.lr,weight_decay=args.weight_decay)
	# initialization
	training_loss = []
	training_acc = []
	validation_acc = []
	test_acc = []
	best_validation_acc = 0
	best_model_acc = 0
	print('Start to train')
	for epoch in range(args.epoch):
		print('epoch:',epoch+1)
		# train for one epoch
		train_model(model,train_loader,optimizer,criterion,device,training_loss,training_acc)
		# validate for one epoch
		validate_model(model,valid_loader,optimizer,criterion,device,validation_acc)
		# test for every test_frequency epochs
		if (epoch + 1) % args.test_frequency == 0:
			test_model(model,test_loader,optimizer,criterion,device,test_acc)
		# adjust the learning rate for every lr_frequency epochs
		if (epoch + 1) % args.lr_frequency == 0:
			lr = args.lr * (0.1 ** (epoch + 1)//10)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print('learning rate:',lr)
		if best_validation_acc < validation_acc[epoch]:
			best_validation_acc = validation_acc[epoch]
			best_model = model
			best_epoch = epoch + 1
			if (epoch + 1) % args.test_frequency == 0:
				best_model_acc = test_acc[epoch]
			else:
				best_model.eval()
				best_correct = 0
				best_total = 0
				with torch.no_grad():
					for images,labels in test_loader:
						images = images.to(device)
						labels = labels.to(device)
						outputs = best_model(images)
						_, predicted = torch.max(outputs.data,1)
						best_total += labels.size(0)
						best_correct += (predicted == labels).sum().item()
				best_model_acc = float(correct)/float(total)
		print('test accuracy of current best model:',best_model_acc)
	torch.save(best_model.state_dict(),'epoch[%d]_valid_acc[%.4f]_test_acc[%.4f]_[%s].pth'%(best_epoch,best_validation_acc,best_model_acc,args.model))
	File = h5py.File('train.h5','w')
	File['data'] = training_acc
	File.close()
	File = h5py.File('valid.h5','w')
	File['data'] = validation_acc
	File.close()
	File = h5py.File('test.h5','w')
	File['data'] = test_acc
	File.close()

if __name__ == '__main__':
	main()