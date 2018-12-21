import torch
import torch.nn as nn

__all__ = [ 'simple', ]

# building model
class Net(nn.Module):
    def __init__(self,num_classes):
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
            nn.Linear(100,num_classes),
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def simple(pretrained=False, num_classes=None):
    model = Net(num_classes)
    if pretrained:
        model.load_state_dict(torch.load('simple.pth',map_location='cpu'))
    return model