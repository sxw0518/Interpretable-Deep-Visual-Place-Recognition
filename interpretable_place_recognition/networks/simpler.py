import torch
import torch.nn as nn

__all__ = [ 'simpler', ]

# building model
class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net,self).__init__()
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
            nn.Linear(100,num_classes),
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def simpler(pretrained=False, num_classes=None):
    model = Net(num_classes)
    if pretrained:
        model.load_state_dict(torch.load('simpler.pth',map_location='cpu'))
    return model