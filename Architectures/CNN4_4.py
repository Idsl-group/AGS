import torch.nn as nn
import torch
from Utils.ch_sharp_utils_2d import *
from Utils.attribute import *
import captum.attr as ca
import matplotlib.pyplot as plt

class CNN4_4(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN4_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=(5,5),stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(5,5),stride=1)
        self.batchnorm32 = nn.BatchNorm2d(num_features=32)
        self.maxpool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1)
        self.batchnorm64 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.batch2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(64*2*2, num_classes))

    def forward(self, x):           # [250, 1, 28, 28]
        x = self.conv1(x)           # [250, 32, 24, 24]
        x = self.batchnorm32(x)
        x = self.relu(x)
        x = self.conv2(x)           # [250, 32, 20, 20]
        x = self.batchnorm32(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [250, 32, 10, 10]
        x = self.dropout(x)
        x = self.conv3(x)           # [250, 64, 8, 8]
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.conv4(x)              # [250, 64, 4, 4]
        x = self.batchnorm64(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [250, 64, 2, 2]
        x = self.dropout(x)
        out = self.classifier(x)
        return out


class AGS_CNN4_4(nn.Module):
    def __init__(self, net, n_steps=1, kernel_size=3, baselines=None, log_path=None):
        super(AGS_CNN4_4, self).__init__()
        self.model = net.cuda()
        if log_path:
            self.model.load_state_dict(torch.load(log_path))
        self.attributions = ca.IntegratedGradients(self.model)
        self.attrargs = {'baselines': baselines, 'n_steps': n_steps}
        self.kernel_size = kernel_size
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self, x, show=False, v=1):
        # Transfer everything to cpu, then cuda
        x = x.cpu();  self.attrargs['baselines'] = self.attrargs['baselines'].cpu();
        x = x.cuda(); self.attrargs['baselines'] = self.attrargs['baselines'].cuda();
        yt = self.model(x).max(1)[1]
        X = x.detach().clone()
        attr = attribute(self.attributions, x, yt, self.model, **self.attrargs).clip(0, 1)
        X = attr_batch_kernelSharp(X, self.kernel_size, h=attr, v=v)
        out = self.model(X)
        if show:
            plt.imshow(np.array((X - x)[0].detach().cpu()).reshape(28, 28), cmap='gray')
            plt.show()
            plt.imshow(np.array(X[0].detach().cpu()).reshape(28, 28), cmap='gray')
            plt.show()
        return out