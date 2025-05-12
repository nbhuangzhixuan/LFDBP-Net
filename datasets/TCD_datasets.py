import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF


class TrainDatasets(Dataset):

    def __init__(self, root_dir, isTrain=True, x_name='cloudy', y_name='clean'):
        self.root_dir = root_dir
        self.isTrain = isTrain
        self.train_X = x_name
        self.train_Y = y_name
        self.X_dir_list = os.listdir(os.path.join(self.root_dir, self.train_X))  # get list of image paths in domain X
        self.Y_dir_list = os.listdir(os.path.join(self.root_dir, self.train_Y))  # get list of image paths in domain Y
        self.transforms = self.get_transforms(crop_size=256)

    def __len__(self):
        return len(self.X_dir_list)

    def __getitem__(self, index):
        X_img_name = self.X_dir_list[index % len(self.X_dir_list)]

        if self.isTrain:
            ind_Y = random.randint(0, len(self.Y_dir_list) - 1)
            Y_img_name = self.Y_dir_list[ind_Y]
        else:
            assert len(self.X_dir_list) == len(self.Y_dir_list)
            Y_img_name = self.Y_dir_list[index % len(self.Y_dir_list)]
            # Y_img_name = X_img_name
        name = Y_img_name.split('.jpg')[0].split('.png')[0]
        X_img = Image.open(os.path.join(self.root_dir, self.train_X, X_img_name))
        Y_img = Image.open(os.path.join(self.root_dir, self.train_Y, Y_img_name))
        X = self.transforms(X_img)
        Y = self.transforms(Y_img)

        return X, Y

    def get_transforms(self, crop_size=512):

        if self.isTrain:
            all_transforms = [transforms.RandomCrop(crop_size), transforms.ToTensor()]
        else:
            all_transforms = [transforms.ToTensor()]

        return transforms.Compose(all_transforms)


class TestDatasets(Dataset):

    def __init__(self, root_dir, isTrain=True, x_name='cloudy', y_name='clean'):
        self.root_dir = root_dir
        self.isTrain = isTrain
        self.train_X = x_name
        self.train_Y = y_name
        self.X_dir_list = os.listdir(os.path.join(self.root_dir, self.train_X))  # get list of image paths in domain X
        self.Y_dir_list = os.listdir(os.path.join(self.root_dir, self.train_Y))  # get list of image paths in domain Y
        self.transforms = self.get_transforms()

    def __len__(self):
        return len(self.X_dir_list)

    def __getitem__(self, index):
        X_img_name = self.X_dir_list[index % len(self.X_dir_list)]

        if self.isTrain:
            ind_Y = random.randint(0, len(self.Y_dir_list) - 1)
            Y_img_name = self.Y_dir_list[ind_Y]
        else:
            assert len(self.X_dir_list) == len(self.Y_dir_list)
            Y_img_name = self.Y_dir_list[index % len(self.Y_dir_list)]
            # Y_img_name = X_img_name
        name = Y_img_name.split('.jpg')[0].split('.png')[0]
        X_img = Image.open(os.path.join(self.root_dir, self.train_X, X_img_name))
        Y_img = Image.open(os.path.join(self.root_dir, self.train_Y, Y_img_name))
        X = self.transforms(X_img)
        Y = self.transforms(Y_img)

        return X, Y

    def get_transforms(self, crop_size=512):

        if self.isTrain:
            all_transforms = [transforms.RandomCrop(crop_size), transforms.ToTensor()]
        else:
            all_transforms = [transforms.ToTensor()]

        return transforms.Compose(all_transforms)
