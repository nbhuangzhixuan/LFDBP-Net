import glob
import cv2
import random
import numpy as np
import pickle
import os

from torch.utils import data


class TrainDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        train_list_path = os.path.join(config.datasets_dir, 'train_list.txt')

        # 读取训练图像文件名列表
        with open(train_list_path, 'r') as file:
            self.train_list = [line.strip() for line in file]

        self.gt_list = [f for f in os.listdir(os.path.join(config.datasets_dir, 'ground_truth')) if
                        f in self.train_list]
        self.cloudy_list = [f for f in os.listdir(os.path.join(config.datasets_dir, 'cloudy_image')) if
                            f in self.train_list]

        # 随机打乱列表
        random.shuffle(self.gt_list)
        random.shuffle(self.cloudy_list)

    def __getitem__(self, index):
        gt_index = random.randint(0, len(self.gt_list) - 1)
        cloudy_index = random.randint(0, len(self.cloudy_list) - 1)

        gt_path = os.path.join(self.config.datasets_dir, 'ground_truth', self.gt_list[gt_index])
        cloudy_path = os.path.join(self.config.datasets_dir, 'cloudy_image', self.cloudy_list[cloudy_index])

        t = cv2.imread(gt_path, 1).astype(np.float32) / 255
        x = cv2.imread(cloudy_path, 1).astype(np.float32) / 255

        t = t.transpose(2, 0, 1)
        x = x.transpose(2, 0, 1)

        return x, t

    def __len__(self):
        return min(len(self.gt_list), len(self.cloudy_list))


class TestDataset(data.Dataset):
    def __init__(self, root):
        super().__init__()
        test_list_path = os.path.join(root, 'test_list.txt')

        # 读取测试图像文件名列表
        with open(test_list_path, 'r') as file:
            self.test_list = [line.strip() for line in file]

        self.test_dir = os.path.join(root, 'cloudy_image')
        self.gt_dir = os.path.join(root, 'ground_truth')

    def __getitem__(self, index):
        filename = self.test_list[index]

        cloudy_path = os.path.join(self.test_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        x = cv2.imread(cloudy_path, 1).astype(np.float32) / 255
        t = cv2.imread(gt_path, 1).astype(np.float32) / 255

        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t, filename

    def __len__(self):
        return len(self.test_list)