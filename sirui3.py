from __future__ import print_function
import argparse
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import vgg16
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import ast
from torch.nn import functional as F
from math import log10
import random
import torch.utils.data
import torch.backends.cudnn as cudn
import argparse
import pickle
from convlstm import *
import pytorch_msssim
'''encoder use vggnet not use unet and use convlstm but not use attention  '''
# class Encoder(torch.nn.Module):
#     def __init__(self, t_length=5, n_channel=3):
#         super(Encoder, self).__init__()
#
#         def Basic(intInput, intOutput):
#             return torch.nn.Sequential(
#                 torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(intOutput),
#                 torch.nn.ReLU(inplace=False),
#                 torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(intOutput),
#                 torch.nn.ReLU(inplace=False)
#             )
#
#         def Basic_(intInput, intOutput):
#             return torch.nn.Sequential(
#                 torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#                 torch.nn.BatchNorm2d(intOutput),
#                 torch.nn.ReLU(inplace=False),
#                 torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
#             )
#
#         self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
#         self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.moduleConv2 = Basic(64, 128)
#         self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.moduleConv3 = Basic(128, 256)
#         self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.moduleConv4 = Basic_(256, 512)
#         self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
#         self.moduleReLU = torch.nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         tensorConv1 = self.moduleConv1(x)
#         tensorPool1 = self.modulePool1(tensorConv1)
#
#         tensorConv2 = self.moduleConv2(tensorPool1)
#         tensorPool2 = self.modulePool2(tensorConv2)
#
#         tensorConv3 = self.moduleConv3(tensorPool2)
#         tensorPool3 = self.modulePool3(tensorConv3)
#
#         tensorConv4 = self.moduleConv4(tensorPool3)
#
#         return tensorConv4, tensorConv1, tensorConv2, tensorConv3

class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(512, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128, 64)

        self.moduleDeconv4 = Basic(64, 64)
        self.moduleUpsample = Upsample(64, 64)
        #self.moduleDeconv2 = Basic(256, 128)
        #self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(64, n_channel, 64)

    def forward(self, x):
        tensorConv = self.moduleConv(x) #1,512,32,32
        #print('tensorConv.shape===',tensorConv.shape)

        tensorUpsample4 = self.moduleUpsample4(tensorConv) #1,256,64,64
        #print('tensorUpsample4.shape===', tensorUpsample4.shape)

        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4) #1,128,64,64
       # print('tensorConv3.shape===', tensorDeconv3.shape)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3) # 1,64,128,128
       # print('tensorUpsample3.shape===', tensorUpsample3.shape)

        tensorDeconv4 = self.moduleDeconv4(tensorUpsample3)
        tensorUpsample4 = self.moduleUpsample(tensorDeconv4)
       # print('tensorUpsample4.shape===', tensorUpsample4.shape)

        output = self.moduleDeconv1(tensorUpsample4)

        return output

class mymodel(nn.Module):
    def __init__(self,input_size,input_dim,hidden_dim,kernel_size,num_layers,bias=False,n_channel =3,  t_length = 5):
        super(mymodel,self).__init__()
        self.have_cuda = True
        self.height,self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        #self.Encoder = Encoder(t_length,n_channel)
        self.Decoder = Decoder(t_length,n_channel)
        #self.ConvLSTM = ConvLSTM(input_size = (16,16),input_dim = 1024,hidden_dim = [512,256,128,64,32],kernel_size = (3,3),num_layers=5,#5层双层lstmbatch_first=True,bias = True)#用不上 用上了还得改参数

        self.Convlstm = ConvLSTM(input_size=(32, 32),
                                 input_dim=128,
                                 hidden_dim=128,
                                 kernel_size=(3, 3),
                                 num_layers=3,
                                 bias=True)

        #self.prior = vgg16(pretrained=False).features[:-1]  # [:-1]就是除了最后一行其他都选出来
        self.encoder = vgg16(pretrained=False)
        self.encoder.features[0] = nn.Conv2d(12, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 把第一层改成这个
        self.encoder.features[23] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1,ceil_mode=False)  # Size([8, 512, 32, 32])
        self.encoder = self.encoder.features[:-1]
        # self.feature = vgg16(pretrained=False)
        # self.feature.features[23] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1,
        #                                          ceil_mode=False)  # Size([8, 512, 32, 32])  等于是把每个像素点又过了一遍啥都不做
        # self.feature = self.feature.features[:-1]

    def forward(self,x,train = True):
      #  print('x===',x.shape)
        fea = self.encoder(x)
      #  print('fea===',fea.shape)
        if train:
            h = self.Convlstm(fea)
          #  print('h=====',h.shape)
            output = self.Decoder(h)
           # print('output===',output.shape)
            return output
        else:
            h = self.Convlstm(fea)
            out = self.Decoder(h)
            return out

def recons_loss(recon_x, x):
    # msssim 多尺度结构相似损失函数：基于多层（图片按照一定规则，由大到小缩放）的SSIM损失函数，相当于考虑了分辨率
   # print('reconshape =====',recon_x.shape)
  #  print('x shape===',x.shape)
    msssim = ((1-pytorch_msssim.ms_ssim(x,recon_x)))/2   #一种优化过的ssim算法
    #作者结合神经科学的研究，认为我们人类衡量两幅图的距离时，
    # 更偏重于两图的结构相似性，而不是逐像素计算两图的差异。因此作者提出了基于 structural similarity 的度量，声称其比 MSE 更能反映人类视觉系统对两幅图相似性的判断。
    f1 =  F.l1_loss(recon_x, x)  #l1损失：基于逐像素比较差异，然后取绝对值  l2损失：逐像素比较差异 取平方
    #L2损失函数会放大最大误差和最小误差之间的差距（比如2*2 和0.1*0.1），另外L2损失函数对异常点也比较敏感
    #论文证明 MS-SSIM+L1损失函数是最好的
    #作者这样组合的原因是，MS-SSIM容易导致亮度的改变和颜色的偏差，但它能保留高频信息（图像的边缘和细节），
    # 而L1损失函数能较好的保持亮度和颜色不变化。公式中α为0.84，是作者试验出来的，而G为高斯分布参数（MS-SSIM里面也要用到这个） Lmix = α*Lmsssim + (1-α)*G*L1  G是高斯分布函数
    return msssim+f1




