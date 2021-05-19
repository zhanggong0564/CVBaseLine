import torch.nn as nn
import torch



class BaseConv(nn.Module):
    def __init__(self,inplane,outplane,kernel_size,stride=1,padding=0,use_bn=True,use_maxpool=True):
        super(BaseConv, self).__init__()
        if use_bn:
            bias = False
        else:
            bias =True
        self.use_maxpool = use_maxpool
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplane,outplane,kernel_size,stride,padding,bias=bias)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
    def forward(self, x):
        x =self.conv1(x)
        if self.bn:
            x = self.bn(x)
        x =self.relu(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        return x


class AlexNet(nn.Module):
    def __init__(self,num_classes = 1000):
        super(AlexNet, self).__init__()
        self.conv1 = BaseConv(3,64,11,4,2)
        self.conv2 = BaseConv(64,192,5,padding=2)
        self.conv3 = BaseConv(192, 384, 3,padding=1,use_maxpool=False)
        self.conv4 = BaseConv(384, 256, 3,padding=1,use_maxpool=False)
        self.conv5 = BaseConv(256, 256, 3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self,x):
        x = self.conv1(x)
        x =self.conv2(x)
        x =self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x =self.classifier(x)
        return x
if __name__ == '__main__':
    x = torch.randn((8,3,224,224))
    model = AlexNet()
    y = model(x)
    print(y.shape)