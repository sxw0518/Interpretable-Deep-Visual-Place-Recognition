# The Grad-CAM structure is on Pytorch framework.
# This code is modified on the basis of the work of https://github.com/jacobgil/pytorch-grad-cam

import torch
import torch.nn as nn
import cv2
import numpy as np

__all__ = [ 'preprocess_image','GradCam','generate_heatmap','generate_patch', ]

def preprocess_image(img,device):
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	# opencv imread in BGR order, this code adjust the order of image channel into RGB
	preprocessed_img = img.copy()[: , :, ::-1]
	# image normalization
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - mean[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / std[i]
	# coverting a numpy Height*Width*nChannels array to nChannels*Height*Width
	preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	# adding one fake dimension (nTensorSamples*nChannels*Height*Width)
	preprocessed_img.unsqueeze_(0)
	image = torch.tensor(preprocessed_img, requires_grad = True).to(device)
	return image

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
		one_hot = torch.tensor(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

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

def generate_heatmap(mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	return heatmap

def generate_patch(image, mask, threshold, patch_dir):
	image[mask<=threshold] = 0
	if np.max(image) > 0:
		cam = image / np.max(image)
	else:
		cam = image
	cv2.imwrite(patch_dir + "_patch.jpg", np.uint8(cam*255))