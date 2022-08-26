import math
import sys

import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import autograd
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
from torch.nn.parameter import Parameter
import timm


# class AdaptiveConcatPool2d(nn.Module): # 사용 X
#     def __init__(self, sz=None):
#         super().__init__()
#         sz = sz or (1,1)
#         self.ap = nn.AdaptiveAvgPool2d(sz)
#         self.mp = nn.AdaptiveMaxPool2d(sz)

#     def forward(self, x):
#         return torch.cat([self.ap(x), self.mp(x)], 1)

# def l2_norm(input, axis=1):
#     norm = torch.norm(input,2, axis, True)
#     output = torch.div(input, norm)
#     return output

# class DenseNet169_change_avg(nn.Module):
#     def __init__(self):
#         super(DenseNet169_change_avg, self).__init__()
#         self.densenet169 = torchvision.models.densenet169(pretrained=True).features
#         self.avgpool = nn.AdaptiveAvgPool2d(1)  
#         self.relu = nn.ReLU()
#         self.mlp = nn.Linear(1664, 6)
#         self.sigmoid = nn.Sigmoid()   

#     def forward(self, x):
#         x = self.densenet169(x)      
#         x = self.relu(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.mlp(x)

#         return x

# class DenseNet121_change_avg(nn.Module): #default model
#     def __init__(self):
#         super(DenseNet121_change_avg, self).__init__()
#         self.densenet121 = torchvision.models.densenet121(pretrained=True).features
    
#         # input channel 바꾸기
#         #self.densenet121 = torchvision.models.densenet121(pretrained=True)

#         # new_classifier = nn.Sequential(*list(self.densenet121.classifier.children())[:-1])
#         # self.densenet121.classifier = new_classifier

#         # prev_w = self.densenet121.features.conv0.weight
#         # self.densenet121.features.conv0 = nn.Conv2d(48, 64, kernel_size=(7, 7), stride=(2, 2), padding=(4, 4), bias=False)   
#         # self.densenet121.features.conv0.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64, 45, 7, 7)), dim=1))
        
#         #self.densenet121.classifier = nn.Linear(4096, 6) # 1024,1000


#         self.avgpool = nn.AdaptiveAvgPool2d(1)  
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(1024, 6) 
#         #self.fc1 = nn.Linear(4096, 6)  
#         self.sigmoid = nn.Sigmoid()   

#     def forward(self, x):
#         x = self.densenet121(x)  #4096,1000
#         x = self.relu(x)
#         x = self.avgpool(x) #1000,1
#         x = x.view(x.size(0), -1) 
#         x = self.fc1(x)

        
#         return x


# CLS
class DenseNet121_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        
        return x

class se_resnext101_32x4d_256(nn.Module):
    def __init__(self):
        super(se_resnext101_32x4d_256, self).__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        #retrained_model = torch.load('/home/kka0602/RSNA2019_2DNet/RSNA2019_Intracranial-Hemorrhage-Detection/2DNet/src/data_test/data_test/se_resnext101_32x4d/test_bone_rsna/model_epoch_9_3.pth')

        # prev_w = self.model_ft.layer0.conv1.weight #(64, 3, 7, 7)
        # self.model_ft.layer0.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   
        # # # #self.model_ft.layer0.conv1.weight = nn.Parameter(torch.zeros(64, 1, 7, 7))
        # self.model_ft.layer0.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64, 9, 7, 7)), dim=1)) #(64, 4, 7, 7)
        # #print(self.model_ft.layer0.conv1.weight.shape)

        # pretrained_models2 = torch.load('/home/kka0602/RSNA2019_Intracranial-Hemorrhage-Detection_modified/2DNet/src/data_test/se_resnext101_32x4d/denoising_otitis_all_pretrained/model_epoch_19_3.pth')
        # self.model_ft.load_state_dict(pretrained_models2, strict=False) 
   
        # for params in self.model_ft.parameters():
        #     params.requires_grad = False 
        
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 3, bias=True))

    def forward(self, x):
        x = self.model_ft(x)
        return x
    

class efficientnet_b3(nn.Module):
    def __init__(self):
        super(efficientnet_b3, self).__init__()
        self.net = timm.create_model('efficientnet_b3', pretrained=False,  num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        num_ftrs = self.net.classifier.in_features
        self.fc = nn.Linear(num_ftrs , 128)
        self.fc2 =  nn.Linear(128 , 3) #class = 3
        self.dp = nn.Dropout(p=0.25)
    
    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc(x))
        x = self.dp(x)
        x= self.fc2(x)
        
        return x    

######################### ECA_module ##############################
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECABottleneck(nn.Module): #101 
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size) #efficient channel attention module
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out) #마지막 eca module

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    #ECA_block, [3, 4, 23, 3], num_classes=6, k_size=3
    def __init__(self, block, layers, num_classes=1, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # prev_w = self.model.layer0.conv1.weight 
        # self.model_ft.layer0.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64, 9, 7, 7)), dim=1))

        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)   
        self.sigmoid = nn.Sigmoid()  
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=3, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


### Denoiser 

class Unet(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, residual=False, down='conv', up='tconv', activation='selu'):
        super(Unet, self).__init__()

        self.residual = residual

        if down == 'maxpool':
            self.down1 = nn.MaxPool2d(kernel_size=2)
            self.down2 = nn.MaxPool2d(kernel_size=2)
            self.down3 = nn.MaxPool2d(kernel_size=2)
            self.down4 = nn.MaxPool2d(kernel_size=2)
        elif down == 'avgpool':
            self.down1 = nn.AvgPool2d(kernel_size=2)
            self.down2 = nn.AvgPool2d(kernel_size=2)
            self.down3 = nn.AvgPool2d(kernel_size=2)
            self.down4 = nn.AvgPool2d(kernel_size=2)
        elif down == 'conv':
            self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)
            self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, groups=64)
            self.down3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, groups=128)
            self.down4 = nn.Conv2d(256, 256, kernel_size=2, stride=2, groups=256)

            self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
            self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
            self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
            self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

            self.down1.bias.data = 0.01 * self.down1.bias.data + 0
            self.down2.bias.data = 0.01 * self.down2.bias.data + 0
            self.down3.bias.data = 0.01 * self.down3.bias.data + 0
            self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        if up == 'bilinear' or up == 'nearest':
            self.up1 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up2 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up3 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
            self.up4 = lambda x: nn.functional.interpolate(x, mode=up, scale_factor=2)
        elif up == 'tconv':
            self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, groups=256)
            self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, groups=128)
            self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, groups=64)
            self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, groups=32)

            self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
            self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
            self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
            self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

            self.up1.bias.data = 0.01 * self.up1.bias.data + 0
            self.up2.bias.data = 0.01 * self.up2.bias.data + 0
            self.up3.bias.data = 0.01 * self.up3.bias.data + 0
            self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock(n_channel_in, 32, residual, activation) 
        self.conv2 = ConvBlock(32, 64, residual, activation)
        self.conv3 = ConvBlock(64, 128, residual, activation)
        self.conv4 = ConvBlock(128, 256, residual, activation)

        self.conv5 = ConvBlock(256, 256, residual, activation)

        self.conv6 = ConvBlock(2 * 256, 128, residual, activation)
        self.conv7 = ConvBlock(2 * 128, 64, residual, activation)
        self.conv8 = ConvBlock(2 * 64, 32, residual, activation)
        self.conv9 = ConvBlock(2 * 32, n_channel_out, residual, activation)

        if self.residual:
            self.convres = ConvBlock(n_channel_in, n_channel_out, residual, activation)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        # print("shapes: c0:%sx:%s c4:%s " % (c0.shape,x.shape,c4.shape))
        x = torch.cat([x, c4], 1)  # x[:,0:128]*x[:,128:256],
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0))

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', residual=True, activation='leakyrelu', transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        x = self.conv1(x)

        if self.dropout:
            x = self.dropout1(x)

        if self.norm1:
            x = self.norm1(x)

        x = self.actfun1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]

        x = self.actfun2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x






