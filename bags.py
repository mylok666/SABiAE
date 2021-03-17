'''评价指标'''
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import copy
import time
from sirui import recons_loss
from sklearn.metrics import roc_auc_score, roc_curve  # 画auc曲线用

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):
    return 10 * math.log10(1 / mse)  # mse代表图像和处理图像之间的均方误差 PSNR就是峰值信噪比

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def normalize_img(img):
    img_re = copy.copy(img)

    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re

def point_score(outputs, imgs):
    loss = recons_loss(outputs,imgs)
    error = loss((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score

def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr - min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list

def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 999999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far - item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer

def AUC(anomal_scores, labels,type,dir):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    fpr, tpr, _ = roc_curve(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores), pos_label=1)
    frr = 1 - tpr
    far = fpr
    eer = compute_eer(far, frr)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [1, 0], '--')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.title('{0} AUC :{1:.3f},EER :{2:.3f}'.format(type, frame_auc, eer))
    plt.savefig(os.path.join('{},{}_auc.png'.format(dir,type)))
    plt.close
    return frame_auc, eer

def score_sum(list1):
    list_result = []
    for i in range(len(list1)):
        list_result.append( list1[i] )

    return list_result
