import torch
import torch.cuda
import torch.nn as nn
import numpy as np
import os
import cv2
import h5py
import argparse
from networks import *
from gradcam import *

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_folder_path',type=str,help='Input Image Folder Path')
	parser.add_argument('--patch_folder_path',type=str,help='Output Patch Folder Path')
	parser.add_argument('--heatmap_folder_path',type=str,help='Output Heatmap Folder Path')
	parser.add_argument('--height',type=int,default=224,help='Height of Resized Image')
	parser.add_argument('--width',type=int,default=224,help='Width of Resized Image')
	parser.add_argument('--model',type=str,help='Model Name')
	parser.add_argument('--compare',action='store_true',default=false,help='Compare Model')
	parser.add_argument('--classes',type=int,default=2,help='Number of Classes')
	parser.add_argument('--saved_model',type=str,help='Name of Saved Trained Model')
	parser.add_argument('--thershold',type=float,default=None,help='Threshold of Generating Visual Explanation')
	args = parser.parse_args()
	return args

class net(nn.Module):
	def __init__(self):
		super(net,self).__init__()
		self.features = nn.Sequential()
		self.classifier = nn.Sequential()

	def forward(self,x):
		x = self.features(x)
		x = x.view(x.size(0),-1)
		x = self.classifier(x)
		return x

def model_change_structure(model,model_name):
	""" The code of Grad-CAM needs 'features' and 'classifier' in the structure of models,
	in order to make it runnable for the models without 'features' and 'classifier', such as Resnet,
	the following class is coded for this purpose"""
	if 'vgg' in model_name or 'alexnet' in model_name or 'squeezenet' in model_name or 'densenet' in model_name:
		return model
	else:
		temp = net()
		mod = list(model.children())
		fc_layer = mod.pop()
		classifier = nn.Sequential(fc_layer)
		features = nn.Sequential(*mod)
		temp.features = features
		temp.classifier = classifier
		return temp


def main():
	args = get_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		use_gpu = True
	else:
		use_gpu = False
	model_names = sorted(name for name in networks.__dict__
		if name.islower() and not name.startswith("__")
		and callable(networks.__dict__[name]))
	model_name = {}
	grad_cam = {}
	folder_name = {}
	for i in range(len(args.model.split(','))):

	model = model_names[args.model](pretrained=False,num_classes=args.classes).to(device)
	model.load_state_dict(torch.load(args.saved_model,map_location=device))
	model = model_change_structure(model,args.model)
	model.eval()
	print('Loading models is done!')
	grad_cam = gradCam.GradCam(model=model,target_layer_names=[str(len(model.features))],use_cuda=use_gpu)
	print('Generating grad_cam is done!')
	target_index = -1 # If target_index is None, it means the highest scoring category.
	for fold in os.listdir(args.image_folder_path):
		target_index += 1
		folder = os.path.join(args.image_folder_path,fold)
		for image in os.listdir(folder):
			image_dir = os.path.join(folder,image)
			img = cv2.imread(image_dir,1)
			img = np.float32(cv2.resize(img,(args.width,args.height)))/255
			processed_img = gradCam.preprocess_image(img)
			mask = grad_cam(processed_img,target_index)
			heatmap_path = os.path.join(args.heatmap_folder_path,image)
			heatmap = gradCam.generate_heatmap(mask)
			heatmap_cam = heatmap + img
			heatmap_cam = heatmap_cam / np.max(heatmap_cam)
			cv2.imwrite(heatmap_path,np.uint8(heatmap_cam*255))
			if args.thershold:
				patch_path = os.path.join(args.patch_folder_path,image)
				gradCam.generate_patch(img,mask,args.thershold,patch_path)
	print('Heatmaps and patches are all saved!')


if __name__ == '__main__':
	main()
