import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

__all__ = [ 'loader', ]

def transform(height,width):
	trans = transforms.Compose([
		transforms.Resize((width,height)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
		])
	return trans

def loader(dir,num,height,width):
	if num:
		data_loader = torch.utils.data.DataLoader(
			datasets.ImageFolder(dir, 
			transform = transform(height,width)),
			batch_size = num,
			shuffle = True,
			num_workers = 2)
	else:
		data_loader = torch.utils.data.DataLoader(
			datasets.ImageFolder(dir,
			transform = transform(height,width)),
			batch_size = num,
			shuffle = True,
			num_workers = 2)
	return data_loader