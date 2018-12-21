# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
import lib.utils.data as torchdata
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import cv2
import pandas as pd
from torch.autograd import Variable
import torchvision.models as models
import csv
import h5py

def preprocess_image(img):
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
	image = Variable(preprocessed_img, requires_grad = True)
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
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
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

def generate_patch(image, mask, threshold):
	image[mask<=threshold] = 0
	if np.max(image) > 0:
		cam = image / np.max(image)
	else:
		cam = image
	return cam

def main(args):
	# torch.cuda.set_device(args.gpu_id)
	# coding book
	dataframe = pd.read_csv('data/object150_info.csv')
	code_book = dataframe['Name'].tolist()

	# Network Builders
	builder = ModelBuilder()
	net_encoder = builder.build_encoder(
		arch=args.arch_encoder,
		fc_dim=args.fc_dim,
		weights=args.weights_encoder)
	net_decoder = builder.build_decoder(
		arch=args.arch_decoder,
		fc_dim=args.fc_dim,
		num_class=args.num_class,
		weights=args.weights_decoder,
		use_softmax=True)
	crit = nn.NLLLoss(ignore_index=-1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
	segmentation_module = segmentation_module.to(device)
	print('segmentation_module is done')
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
	# # vgg11 =  models.vgg11()
	# # mod = list(vgg11.classifier.children())
	# # mod.pop()
	# # mod.append(torch.nn.Linear(4096,2))
	# # new_classifier = nn.Sequential(*mod)
	# # vgg11.classifier = new_classifier
	# # vgg11.load_state_dict(torch.load('/home/nfs/xiangweishi/vgg.pth'))
	# # vgg11 = vgg11.type(dtype)
	# vgg11_2 =  models.vgg11()
	# mod_2 = list(vgg11_2.classifier.children())
	# mod_2.pop()
	# mod_2.append(torch.nn.Linear(4096,2))
	# new_classifier_2 = nn.Sequential(*mod_2)
	# vgg11_2.classifier = new_classifier_2
	# vgg11_2.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced_epoch[18]_learning_rate[0.0000]_vgg_2.pth'))
	# vgg11_2 = vgg11_2.to(device)
	# vgg11_2.eval()
	# # # the second model
	# # resnet18 = models.resnet18()
	# # resnet18.fc = nn.Linear(512,2)
	# # resnet18.load_state_dict(torch.load('/home/nfs/xiangweishi/resnet.pth'))
	# # my_model = net()
	# # mod = list(resnet18.children())
	# # layer = mod.pop()
	# # classifier = nn.Sequential(layer)
	# # my_model.classifier = classifier
	# # features = nn.Sequential(*mod)
	# # my_model.features = features
	# # resnet18 = my_model.type(dtype)
	# resnet18_2 = models.resnet18()
	# resnet18_2.fc = nn.Linear(512,2)
	# resnet18_2.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced_epoch[13]_learning_rate[0.0000]_resnet_2.pth'))
	# my_model_2 = net()
	# mod_2 = list(resnet18_2.children())
	# layer_2 = mod_2.pop()
	# classifier_2 = nn.Sequential(layer_2)
	# my_model_2.classifier = classifier_2
	# features_2 = nn.Sequential(*mod_2)
	# my_model_2.features = features_2
	# resnet18_2 = my_model_2.to(device)
	# resnet18_2.eval()
	# # simple model
	# # class Net(nn.Module):
	# # 	def __init__(self):
	# # 		super(Net,self).__init__()
	# # 		self.features = nn.Sequential(
	# # 		nn.Conv2d(3,20,5),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		nn.Conv2d(20,64,7),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		nn.Conv2d(64,96,5),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		nn.Conv2d(96,128,7),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		)
	# # 		self.classifier = nn.Sequential(
	# # 			nn.Linear(128*9*9,4096),
	# # 			nn.ReLU(True),
	# # 			nn.Dropout(),
	# # 			nn.Linear(4096,100),
	# # 			nn.ReLU(True),
	# # 			nn.Dropout(),
	# # 			nn.Linear(100,2),
	# # 			)
	# # 	def forward(self,x):
	# # 		x = self.features(x)
	# # 		x = x.view(x.size(0),-1)
	# # 		x = self.classifier(x)
	# # 		return x
	# # # simple_model = Net()
	# # # simple_model.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced__epoch[27]_learning_rate[0.0000].pth'))
	# # # simple_model = simple_model.to(device)
	# # # simple_model.eval()
	# # simple_model_2 = Net()
	# # simple_model_2.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced_epoch[27]_learning_rate[0.0000]_simple_2.pth'))
	# # simple_model_2 = simple_model_2.to(device)
	# # simple_model_2.eval()
	# # # simpler model
	# # class Simpler(nn.Module):
	# # 	def __init__(self):
	# # 		super(Simpler,self).__init__()
	# # 		self.features = nn.Sequential(
	# # 		nn.Conv2d(3,20,9),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		nn.Conv2d(20,64,9),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		nn.Conv2d(64,96,9),
	# # 		nn.ReLU(True),
	# # 		nn.MaxPool2d(2,2),
	# # 		)
	# # 		self.classifier = nn.Sequential(
	# # 		nn.Linear(96*21*21,4096),
	# # 		nn.ReLU(True),
	# # 		nn.Dropout(),
	# # 		nn.Linear(4096,100),
	# # 		nn.ReLU(True),
	# # 		nn.Dropout(),
	# # 		nn.Linear(100,2),
	# # 		)
	# # 	def forward(self,x):
	# # 		x = self.features(x)
	# # 		x = x.view(x.size(0),-1)
	# # 		x = self.classifier(x)
	# # 		return x
	# # # simpler_model = Simpler()
	# # # simpler_model.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced__epoch[32]_learning_rate[0.000000]_simpler.pth'))
	# # # simpler_model = simpler_model.to(device)
	# # # simpler_model.eval()
	# # simpler_model_2 = Simpler()
	# # simpler_model_2.load_state_dict(torch.load('/home/nfs/xiangweishi/Balanced_epoch[22]_learning_rate[0.0000]_simpler_2.pth'))
	# # simpler_model_2 = simpler_model_2.to(device)
	# # simpler_model_2.eval()
	# print('visualization models done')
	# # model_name = {0:vgg11,1:resnet18,2:simple_model,3:simpler_model}
	# model_name = {0:vgg11_2,1:resnet18_2}
	# if torch.cuda.is_available():
	# 	use_gpu = True
	# else:
	# 	use_gpu = False
	# # grad_cam_vgg11 = GradCam(model = vgg11, target_layer_names = ["19"], use_cuda = use_gpu)
	# # grad_cam_resnet18 = GradCam(model = resnet18, target_layer_names = ["7"], use_cuda = use_gpu)
	# # grad_cam_simple_model = GradCam(model = simple_model, target_layer_names = ["10"], use_cuda = use_gpu)
	# # grad_cam_simpler_model = GradCam(model = simpler_model, target_layer_names = ["7"], use_cuda = use_gpu)
	# # grad_cam = {0:grad_cam_vgg11,1:grad_cam_resnet18,2:grad_cam_simple_model,3:grad_cam_simpler_model}
	# # grad_cam_simple_model_2 = GradCam(model = simple_model_2, target_layer_names = ["10"], use_cuda = use_gpu)
	# # grad_cam_simpler_model_2 = GradCam(model = simpler_model_2, target_layer_names = ["7"], use_cuda = use_gpu)
	# grad_cam_vgg11_2 = GradCam(model = vgg11_2, target_layer_names = ["19"], use_cuda = use_gpu)
	# grad_cam_resnet18_2 = GradCam(model = resnet18_2, target_layer_names = ["7"], use_cuda = use_gpu)
	# grad_cam = {0:grad_cam_vgg11_2,1:grad_cam_resnet18_2}
	# print('generating grad_cams is done')
	# # folder_name = {0:'vgg',1:'resnet',2:'simple',3:'simpler'}
	# folder_name = {0:'vgg_2',1:'resnet_2'}
	target_index = -1
	# threshold = {0:0.5,1:0.6}
	df = pd.DataFrame([])
	for fold in os.listdir(args.image_path):
		folder  = os.path.join(args.image_path,fold)
		target_index = target_index + 1
		if fold == 'Tokyo':
			for image in os.listdir(folder):
				image_dir = os.path.join(folder,image)
				list_test = [{'fpath_img': image_dir}]
				dataset_val = TestDataset(
				list_test, args, max_sample=args.num_val)
				loader_val = torchdata.DataLoader(
				dataset_val,
				batch_size=args.batch_size,
				shuffle=False,
				collate_fn=user_scattered_collate,
				num_workers=5,
				drop_last=True)
				segmentation_raw = test(segmentation_module, loader_val, args, device)
				img = cv2.imread(image_dir, 1)
				img = np.float32(cv2.resize(img, (224,224)))/255
				processed_img = preprocess_image(img)
				record = {}
				dict_list = []
				dict_list.append(image)
				record = {'Name':dict_list}
				# for t in range(len(threshold)):
				# 	for m in range(len(model_name)):
				segmentation_label = segmentation_raw.copy()
				segmentation_befor = np.zeros((segmentation_label.shape[0],segmentation_label.shape[1],3))
				# mask_1 = grad_cam[m](processed_img, target_index)
				segmentation_befor[:,:,0] = segmentation_label
				segmentation_befor[:,:,1] = segmentation_label
				segmentation_befor[:,:,2] = segmentation_label
				segmentation_label = cv2.resize(segmentation_befor,(224,224))[:,:,0]
				# segmentation_label[mask_1<=threshold[t]] = 150
				segmentation_label = segmentation_label.reshape(-1).astype(int)
				count = np.bincount(segmentation_label,minlength=151)
						# distribution_dir = os.path.join('/tudelft.net/staff-bulk/ewi/insy/VisionLab/xiangweishi/segmentation',folder_name[m])
						# distribution_dir = os.path.join(distribution_dir,image+'_'+str(t)+'.h5')
						# print(distribution_dir)
						# f = h5py.File(distribution_dir,'w')
						# f['data'] = count
						# f.close()
		# 				if np.argmax(count) == 150:
		# 					count[150] = 0
		# 					if np.argmax(count) == 0 and count[0]==0:
		# 						final_first_label = 'None'
		# 						final_second_label = ' '
		# 					else:
		# 						final_first_label = code_book[np.argmax(count)]
		# 						first = np.max(count)
		# 						count[np.argmax(count)] = 0
		# 						second = np.max(count)
		# 					if float(second)/float(first)>0.5:
		# 						final_second_label = code_book[np.argmax(count)]
		# 					else:
		# 						final_second_label = ' '
		# 				label = final_first_label+';'+final_second_label
				dict_list = []
				dict_list.append(count)
				# record[str(t)+folder_name[m]] = dict_list
				record['count'] = dict_list 
				# print(record)
				row = pd.DataFrame.from_dict(record)
				df = df.append(row,ignore_index=True)
				# print(df)
		df.to_csv('/tudelft.net/staff-bulk/ewi/insy/VisionLab/xiangweishi/all_segmentation_label_t.csv')
		print('done')

def test(segmentation_module, loader, args, device):

    segmentation_module.eval()
    for i, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']
        with torch.no_grad():
            pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img.to(device)
                del feed_dict['img_ori']
                del feed_dict['info']
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                pred = pred + pred_tmp.cpu() / len(args.imgSize)
            _, preds = torch.max(pred, dim=1)
            preds = as_numpy(preds.squeeze(0))
        print('[{}] iter {}' 
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i))
        return preds

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Path related arguments
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--model_path', required=True,
                        help='folder to model path')
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")

    # Model related arguments
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--result', default='.',
                        help='folder to output visualization results')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu_id for evaluation')

    args = parser.parse_args()
    print(args)

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.model_path,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.model_path,
                                        'decoder' + args.suffix)
    print(args.weights_encoder)
    print(args.weights_decoder)
    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_decoder), 'checkpoint does not exitst!'

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)