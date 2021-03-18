'''一个用来处理公共文件夹下的 拷贝自 Evaluate.py函数 '''
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
import math
from collections import OrderedDict
import copy
import time
from bags import *
#from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
#from model.utils import DataLoader
import argparse
from utils import DataLoader
from sirui import recons_loss
#from Train import log_dir
parser = argparse.ArgumentParser(description="NNU202")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='/home/y192202011/zhuomian/VADdata/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--exp_dir',type = str)
parser.add_argument('--frame_type',type = str,default='*.jpg')

args = parser.parse_args()

torch.manual_seed(2020)
logdir = os.path.join('./exp/', args.dataset_type, args.exp_dir)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

test_folder = args.dataset_path + args.dataset_type + "/testing/frames/"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
    transforms.ToTensor(),
]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)

test_size = len(test_dataset)
print('test_size = ',test_size)
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()


labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
if args.dataset_type == 'shanghai':  # 上海理工是一个一维的数据集，需要转成二维的
    labels = np.expand_dims(labels, 0)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
print(videos_list, '=videolist')
for video in videos_list:
    video_name = video.split('/')[-1]  ## frames\01--12
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, args.frame_type))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])  ##180 180 150 180 150 180 180 180 120 150 180 180 就是ped2testing12个文件夹各自有多少帧

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name][
                                                                        'length'] + label_length])  # label_length.shape = 1962 (2010-48)
    label_length += videos[video_name]['length']
    # print(label_length)   #180 360 510 690 840 1020 1200 1380 1500 1650 1830 2010 最后输出labellength = 2010
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]][
    'length']  # label_length = 180  videos_list[video_num].split('/')[-1] = frames/01  先加上第一个数据集的标签长度为什么？ 为了按序取数据集标签

model.eval()
from tqdm import tqdm
from torchvision.utils import save_image

for k, (imgs) in enumerate(tqdm(test_batch)):
    if k == label_length - 4 * (video_num + 1):  # 通过这个函数增加读取的标签数
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()


    outputs = model.forward(x=imgs[:,0:3*4],train = False) #取前面四张生成outputs
    print('output:',outputs.shape)
    print('input====',imgs[:,0:3*4].shape)
    outdir = os.path.join('./exp/',args.dataset_type,args.exp_dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # x_concat = torch.cat([imgs[:,12:].squeeze(0), outputs.squeeze(0)], dim=3)
    # print(x_concat.shape)
    a = (imgs[:,12:]+1)*127
    b = (outputs+1)*127
    x_concat = torch.cat([a, b], dim=3)
    save_image(x_concat, ("./exp/{}/{}/reconstructed-{}.jpg" .format(args.dataset_type,args.exp_dir,k+1)))
    mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)).item()
    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]))

anomaly_score_total_list = np.asarray(anomaly_score_total_list)
print('last line',labels_list.shape)
accuracy, eer = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0),type=args.dataset_type,dir = logdir)

print('The result of ', args.dataset_type)
print('AUC: ', accuracy * 100, '%', 'EER:', eer * 100, '%')
