# Loss functions
from torch import nn
from torchvision import models

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from torchvision.models import vgg16
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)



class CosineSimilarityLoss(nn.Module):
    def __init__(self,device):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1).to(device)

    def forward(self, features1, features2):
        # 计算两个特征向量之间的余弦相似度
        cosine_loss = 1 - self.cosine_sim(features1, features2)
        # 损失是1减去平均余弦相似度，以使损失最小化时相似度最大化
        return cosine_loss.mean()



def projectedDistributionLoss(x, y, num_projections=128):
    '''Projected Distribution Loss (https://arxiv.org/abs/2012.09289)
    x.shape = B,M,N,...
    '''
    def rand_projections(dim, device=device, num_projections=512):
        projections = torch.randn((dim,num_projections), device=device)
        projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=0, keepdim=True))    # columns are unit length normalized
        return projections
    x = x.reshape(x.shape[0], x.shape[1], -1)   # B,N,M
    y = y.reshape(y.shape[0], y.shape[1], -1)
    W = rand_projections(x.shape[-1], device=x.device, num_projections=num_projections)#x.shape[-1])

    e_x = torch.matmul(x,W) # multiplication via broad-casting
    e_y = torch.matmul(y,W)
    loss = 0
    for ii in range(e_x.shape[2]):
#        g = torch.sort(e_x[:,:,ii],dim=1)[0] - torch.sort(e_y[:,:,ii],dim=1)[0]; print(g.mean(), g.min(), g.max())
        loss = loss + F.l1_loss(torch.sort(e_x[:,:,ii],dim=1)[0] , torch.sort(e_y[:,:,ii],dim=1)[0])    # if this gives issues; try Huber loss later
    return loss


def full_channelwise_l1_distance(feature_map_1, feature_map_2):
    """
    Compute the L1 distance between each channel of feature_map_1 and each channel of feature_map_2.
    Both feature maps have the same shape (N, C, H, W).

    Args:
        feature_map_1 (torch.Tensor): Input tensor of shape (N, C, H, W)
        feature_map_2 (torch.Tensor): Input tensor of shape (N, C, H, W)

    Returns:
        torch.Tensor: Tensor of L1 distances of shape (N, C, C)
    """
    N, C, H, W = feature_map_1.shape

    # Expand feature_map_1 to (N, C, 1, H, W)
    expanded_map_1 = feature_map_1.unsqueeze(2)

    # Expand feature_map_2 to (N, 1, C, H, W)
    expanded_map_2 = feature_map_2.unsqueeze(1)

    # Calculate L1 distance, resulting shape will be (N, C, C, H, W)
    l1_distance = torch.abs(expanded_map_1 - expanded_map_2).mean(dim=[3, 4])  # Reduce over spatial dimensions

    return l1_distance


class ContrastLoss(nn.Module):
    def __init__(self, device):
        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss().to(device)

    def forward(self, i_g, j_g, i_c):
        positive_dis = self.l1(j_g, i_g)
        negtive1_dis = self.l1(j_g, i_c)
        negtive2_dis = self.l1(i_g, i_c)

        loss_contrastive = positive_dis / (negtive1_dis + negtive2_dis + 1e-7)


        return loss_contrastive
#
# class ContrastLoss(nn.Module):
#     def __init__(self, device):
#         super(ContrastLoss, self).__init__()
#         self.l1 = nn.L1Loss().to(device)
#         # self.cos = CosineSimilarityLoss(device)
#
#     def forward(self, i_g, j_g, i_c):
#         positive_dis = self.l1(j_g, i_g.detach())
#
#         negtive1_dis = self.l1(i_c, i_g)
#
#         negtive2_dis = self.l1(i_c, j_g)
#
#         loss_contrastive = positive_dis / (negtive1_dis + negtive2_dis + 1e-7)
#
#         return loss_contrastive


class ContrastLoss_nodiscriminator(nn.Module):
    def __init__(self, device):
        super(ContrastLoss_nodiscriminator, self).__init__()
        self.l1 = nn.L1Loss().to(device)
        self.cos = CosineSimilarityLoss(device)

    def forward(self, i_g, j_g, i_c):
        positive_dis = self.cos(j_g, i_g.detach())

        negtive1_dis = self.cos(i_g, i_c)

        negtive2_dis = self.cos(j_g, i_c)


        loss_contrastive = positive_dis / (negtive1_dis + negtive2_dis + 1e-7)

        return loss_contrastive
