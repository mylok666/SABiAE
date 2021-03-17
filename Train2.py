import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import skimage.transform as trans
import math
from collections import OrderedDict  # 容器包
import copy
import time
from sklearn.metrics import roc_auc_score
from utils import DataLoader
import random
from pytorch_msssim import ms_ssim
import argparse
from sirui2 import mymodel,recons_loss
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="sirui")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--vali',type = float, default = 0.2,help = 'validation split')

args = parser.parse_args()

torch.manual_seed(2020)  # #为CPU设置种子用于生成随机数，以使得结果是确定的


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance  确保使用cudnn

train_folder = args.dataset_path + args.dataset_type + "/training/frames/"
#train_folder = args.dataset_path + args.dataset_type + "/videos/training_frames/"
test_folder = args.dataset_path + args.dataset_type + "/testing/frames/"

train_dataset = DataLoader(train_folder, transforms.Compose([transforms.ToTensor(), ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
# transforms.ToTensor() 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间

train_size = len(train_dataset)
print(train_size)
split = int(np.floor(args.vali * train_size))
# 在训练模型时使用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化
val_size = int(args.vali*len(train_dataset)) #验证用20%的训练集
train_size = int(len(train_dataset)-val_size) #训练集就是剩下80%
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers = args.num_workers, drop_last=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size,shuffle=True,
                              num_workers = args.num_workers, drop_last=True)

# Model setting
model = mymodel(input_size=(32,32),input_dim=128,hidden_dim=128,kernel_size=(3,3),num_layers=1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 设置学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # 学习率调整函数  这里使用余弦退火
model = nn.DataParallel(module = model)
model.cuda()  # 把模型放到显卡上

# Report the training process
log_dir = os.path.join('./exp/', args.dataset_type, args.exp_dir)

if not os.path.exists(log_dir):  # 若没有这个文件夹就创建
    os.makedirs(log_dir)

# Training
for epoch in range(args.epochs):
    labels_list = []
    model.train()
    print('This is the {} th training.'.format(epoch+1))
    start = time.time()
    """将数据喂入神经网络进行训练，通过enumerate输出我们想要的经过shuffle的bachsize大小的feature和label数据"""
    for j, (imgs) in enumerate(tqdm(train_loader)):  # 承接上面的dataloader函数，这个是一套固定用法
        #print('j==',j)
        imgs = Variable(imgs).cuda()  # Variable和tensor其实是一个东西，Variable实际上是一个容器，里面面装个tensor

        # 接下来就是跑模型的环节
        output = model.forward(x = imgs[:,0:12,:,:],train = True)
        optimizer.zero_grad()
        '''
        在每次backward后，grad值是会累加的，所以利用BP算法，每次迭代是需要将grad清零的。
        x.grad.data.zero_()
        '''
         # 最后一张是预测出来的 既12：（12，13，14）用于和生成出来的做损失函数   既计算重构误差
        loss = recons_loss(output,imgs[:,12:,:,:])
        loss.backward(retain_graph=True)  # 反向传递函数
        optimizer.step()  # 所有的optimizer都实现了step()方法，这个方法会更新所有的参数。它能按两种方式来使用

    model.eval()
    for i , (imgs) in enumerate(valid_loader):
        imgs = Variable(imgs).cuda()  # Variable和tensor其实是一个东西，Variable实际上是一个容器，里面面装个tensor
        # 接下来就是跑模型的环节
        output = model.forward(x=imgs[:, 0:12, :, :], train=True)
        # 最后一张是预测出来的 既12：（12，13，14）用于和生成出来的做损失函数   既计算重构误差
        valloss = recons_loss(output, imgs[:, 12:, :, :])


    scheduler.step()  # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)

    print(
        'Epoch [{}/{}], Step [{}/{}], Loss: {:.5f},valloss: {:.5f}'.format(
            epoch + 1, args.epochs, j + 1, train_size, loss.item(),valloss.item()))
    print('----------------------------------------')
    print(os.path.join(log_dir, 'model_{:.3f}_{:.5f}_{:.5f}.pth'.format(epoch+1,loss,valloss)))


    torch.save(model, os.path.join(log_dir, 'model_{:.3f}_{:.5f}_{:.5f}.pth'.format(epoch+1,loss,valloss)))

print('Training is finished')




