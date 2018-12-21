import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import torch.nn as nn
import torchvision.models as models
import cv2
import sys
import os
import numpy as np
import argparse

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask, im):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	image = np.float32(img)
	image[mask<=0.45] = 0
	cam = image
	cam = cam / np.max(cam)
	# cam = np.multiply(heatmap, np.float32(img))
	# cam = cam / np.max(cam)
	# cv2.imwrite("cam.jpg", np.uint8(255 * cam))
	cv2.imwrite(im + "_patch.jpg", np.uint8(cam*255))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_variables=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.ones(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/both.png',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()
	# if torch.cuda.is_available(): # GPU
	#     dtype = torch.cuda.FloatTensor
	#     ttype = torch.cuda.LongTensor
	# else: # CPU
	#     dtype = torch.FloatTensor
	#     ttype = torch.LongTensor
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # In order to make this code workable for the Resnet, which does not
    # contains features, classifier from torchvision.models
	# class net(nn.Module):
	# 	def __init__(self):
	# 		super(net,self).__init__()
	# 		self.features = nn.Sequential()
	# 		self.classifier = nn.Sequential()

	# 	def forward(self,x):
	# 		x = self.features(x)
	# 		x = x.view(x.size(0),-1)
	# 		x = self.classifier(x)
	# 		return x

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.


	# model_1 = models.vgg11()
	# mod = list(model_1.classifier.children())
	# mod.pop()
	# mod.append(torch.nn.Linear(4096, 2))
	# new_classifier = torch.nn.Sequential(*mod)
	# model_1.classifier = new_classifier
	# model_1.load_state_dict(torch.load('vgg.pth'))
	# my_model = model_1.type(dtype)

	# in Resnet18 
	# my_model = net()

	# model = models.resnet18()
	# model.fc = nn.Linear(512,2)
	# model.load_state_dict(torch.load('resnet.pth'))

	# mod = list(model.children())
	# layer = mod.pop()
	# classifier = nn.Sequential(layer)
	# my_model.classifier = classifier
	# features = nn.Sequential(*mod)
	# my_model.features = features


	# my_model = my_model.type(dtype)


	# grad_cam = GradCam(model = my_model, \
	# 				target_layer_names = ["7"], use_cuda=args.use_cuda)

	class Net(nn.Module):
		def __init__(self):
			super(Net,self).__init__()
			self.features = nn.Sequential(
			nn.Conv2d(3,20,5),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(20,64,7),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(64,96,5),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(96,128,7),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			)
			self.classifier = nn.Sequential(
				nn.Linear(128*9*9,4096),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(4096,100),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(100,2),
				)
		def forward(self,x):
			x = self.features(x)
			x = x.view(x.size(0),-1)
			x = self.classifier(x)
			return x

	simple_model = Net()
	simple_model.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced__epoch[27]_learning_rate[0.0000].pth'))
	simple_model = simple_model.to(device)
	simple_model.eval()
	simple_model_2 = Net()
	simple_model_2.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced_epoch[27]_learning_rate[0.0000]_simple_2.pth'))
	simple_model_2 = simple_model_2.to(device)
	simple_model_2.eval()
	class Simpler(nn.Module):
		def __init__(self):
			super(Simpler,self).__init__()
			self.features = nn.Sequential(
			nn.Conv2d(3,20,9),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(20,64,9),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.Conv2d(64,96,9),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			)
			self.classifier = nn.Sequential(
			nn.Linear(96*21*21,4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096,100),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(100,2),
			)
		def forward(self,x):
			x = self.features(x)
			x = x.view(x.size(0),-1)
			x = self.classifier(x)
			return x
	simpler_model = Simpler()
	simpler_model.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced__epoch[32]_learning_rate[0.000000]_simpler.pth'))
	simpler_model = simpler_model.to(device)
	simpler_model.eval()
	simpler_model_2 = Simpler()
	simpler_model_2.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced_epoch[22]_learning_rate[0.0000]_simpler_2.pth'))
	simpler_model_2 = simpler_model_2.to(device)
	simpler_model_2.eval()
	for image in os.listdir(args.image_path):
		image_dir = os.path.join(args.image_path,image)
		img = cv2.imread(image_dir, 1)
		img = np.float32(cv2.resize(img, (224, 224))) / 255
		input = preprocess_image(img)

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested index.
		target_index = None
		mask = grad_cam(input, target_index)
		show_cam_on_image(img, mask, image)