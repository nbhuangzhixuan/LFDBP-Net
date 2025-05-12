import os

import numpy
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
import torchvision
from PIL import Image


def augment(img_Cloudy, img_GT):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        img_Cloudy = transforms.functional.rotate(img_Cloudy, rotate_degree)
        img_GT = transforms.functional.rotate(img_GT, rotate_degree)
        #img_SAR = transforms.functional.rotate(img_SAR, rotate_degree)

        return img_Cloudy, img_GT
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        img_Cloudy = vertical_flip(img_Cloudy)
        img_GT = vertical_flip(img_GT)
        #img_SAR = vertical_flip(img_SAR)

        return img_Cloudy, img_GT
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        img_Cloudy = horizontal_flip(img_Cloudy)
        img_GT = horizontal_flip(img_GT)
        #img_SAR = horizontal_flip(img_SAR)

        return img_Cloudy, img_GT
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return img_Cloudy, img_GT


class TrainDataset(Dataset):
    def __init__(self,  root_dir, user_transform=None):
        self.root_dir = root_dir
        self.user_transform = user_transform
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.cloudy_image_paths = []
        self.gt_image_paths = []




       # 加载 Cloudy 图像路径
        cloudy_season_path = os.path.join(root_dir, "winter", "train", 'Cloudy', "Hard")
        self.cloudy_image_paths = [os.path.join(cloudy_season_path, img_name) for img_name in os.listdir(cloudy_season_path)]

        # 加载 GT 图像路径
        gt_season_path = os.path.join(root_dir, "winter", "train", 'GT', "Hard")
        self.gt_image_paths = [os.path.join(gt_season_path, img_name) for img_name in os.listdir(gt_season_path)]


    def __len__(self):
        return len(self.cloudy_image_paths)

    def __getitem__(self, idx):

        img_Cloudy_path = self.cloudy_image_paths[idx]
        with rasterio.open(img_Cloudy_path) as img_c:
            img_Cloudy = img_c.read()
        all_indices = list(range(len(self.gt_image_paths)))

        # 随机选择一个GT索引
        random_idx = numpy.random.choice(all_indices)
        img_GT_path = self.gt_image_paths[random_idx]
        with rasterio.open(img_GT_path) as img_g:
            img_GT = img_g.read()

        if self.user_transform:
            # 应用变换（如果需要的话）
            img_Cloudy = self.transform(img_Cloudy)
            img_GT = self.transform(img_GT)
        else:
            img_Cloudy_tensor = self.transform(img_Cloudy).permute(1, 0, 2)
            img_GT_tensor = self.transform(img_GT).permute(1, 0, 2)


            # crop a patch
            i, j, h, w = transforms.RandomCrop.get_params(img_Cloudy_tensor, output_size=(256, 256))
            img_Cloudy_ = TF.crop(img_Cloudy_tensor, i, j, h, w)
            img_GT_ = TF.crop(img_GT_tensor, i, j, h, w)
            

            # data argumentation
            img_Cloudy_arg, img_GT_final = augment(img_Cloudy_, img_GT_)



        return img_Cloudy_arg, img_GT_final



class TestDataset(Dataset):
    def __init__(self,root_dir,level,season, user_transform=None):
        self.root_dir = root_dir
        self.user_transform = user_transform
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_paths = []
        self.level = level
        self.season = season





    
        season_path = os.path.join(root_dir, season, "test", 'Cloudy')#spring|summer|fall|winter
        level_path = os.path.join(season_path, level)#Easy|Normal|Hard
        for img_name in os.listdir(level_path):
            self.image_paths.append(os.path.join(level_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_Cloudy_path = self.image_paths[idx]
        img_Cloudy_path_split = img_Cloudy_path.split("_", 2)
        temp = img_Cloudy_path_split[2]
        img_GT_path = os.path.join(self.root_dir, self.season, "test", "GT", self.level, temp)
        with rasterio.open(img_Cloudy_path) as img_c:
            img_Cloudy = img_c.read()
        with rasterio.open(img_GT_path) as img_g:
            img_GT = img_g.read()


        if self.user_transform:
            # 应用变换（如果需要的话）
            img_Cloudy = self.transform(img_Cloudy)
            img_GT = self.transform(img_GT)
        else:
            img_Cloudy_tensor = self.transform(img_Cloudy).permute(1, 0, 2)
            img_GT_tensor = self.transform(img_GT).permute(1, 0, 2)





        return img_Cloudy_tensor, img_GT_tensor

