import torch
import torch.nn as nn
import  numpy as np

class Block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Block,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out
def make_layers(in_channels, layer_list,name="vgg"):
    layers = []
    if name=="vgg":
        for v in layer_list:
            layers += [Block(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)
class Layer(nn.Module):
    def __init__(self, in_channels, layer_list ,net_name):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, name=net_name)
    def forward(self, x):
        out = self.layer(x)
        return out


class VGG16(nn.Module):
    '''
    VGG model
    '''
    def __init__(self,num_classes):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = Layer(64, [64], "vgg")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128], "vgg")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256], "vgg")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512], "vgg")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512], "vgg")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        f5 = torch.flatten(f5,1)
        fc = self.classifier(f5)
        return fc
if __name__ == '__main__':
    x = torch.randn((8, 3, 224, 224))
    model = VGG16(100)
    y = model(x)
    print(y.shape)
