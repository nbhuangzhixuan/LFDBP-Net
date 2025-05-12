import rasterio
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch

def save_tensor_image(tensor, filename):
    """
    将PyTorch张量保存为图像文件。

    参数:
    tensor: PyTorch张量，应为(C, H, W)格式且值范围在0到1之间。
    filename: 保存图像的文件名。
    """
    # 确保张量在CPU上，然后转换为NumPy数组
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 确保张量值在[0, 1]之间
    tensor = torch.clamp(tensor, 0, 1)

    # 将张量转换为NumPy数组
    image = tensor.detach().numpy().astype(np.float32)

    # 转换像素值范围到0-255并转换为8位整数
    image = (image * 255).astype(np.uint8)

    # 转换通道顺序(C, H, W) -> (H, W, C)
    image = np.transpose(image, (1, 2, 0))



    # 保存图像
    cv2.imwrite(filename, image)



def save_image_CSRD(tensor, filename):
    """
    将PyTorch张量保存为图像文件。

    参数:
    tensor: PyTorch张量，应为(C, H, W)格式且值范围在0到1之间。
    filename: 保存图像的文件名。
    """
    # 确保张量在CPU上，然后转换为NumPy数组
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 确保张量值在[0, 1]之间
    tensor = torch.clamp(tensor, 0, 1)

    # 将张量转换为NumPy数组
    image = tensor.numpy().astype(np.float32)

    # 转换像素值范围到0-255并转换为8位整数
    image = (image * 255).astype(np.uint8)

    # 转换通道顺序(C, H, W) -> (H, W, C)
    image = np.transpose(image, (1, 2, 0))

    # 如果是灰度图，去掉单通道维度
    if image.shape[2] == 1:
        image = image.squeeze(axis=2)
    elif image.shape[2] == 3:
        # 转换颜色空间 RGB -> BGR，因为cv2默认使用BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 保存图像
    cv2.imwrite(filename, image)


def show1(feature):
    feature_map_data_list = [feature[0, i].detach().cpu().numpy() for i in range(feature.shape[1])]
    # 可视化每个特征图的热力图
    plt.figure(figsize=(6, 6))
    for i, feature_map_data in enumerate(feature_map_data_list):
        if i == 30:
            break
        plt.subplot(6, 5, i + 1)
        plt.imshow(feature_map_data, cmap="gray")
        plt.title(f"Feature Map {i + 1}")
        plt.axis('off')
    plt.show()


def show2(feature):
    feature_map_data_list = [feature[0, i].detach().cpu().numpy() for i in range(feature.shape[1])]
    # 可视化每个特征图的热力图
    plt.figure(figsize=(12, 6))
    for i, feature_map_data in enumerate(feature_map_data_list):
        if i == 5:
            break
        plt.subplot(1, 5, i + 1)
        plt.imshow(feature_map_data, cmap="jet")
        plt.title(f"Feature Map {i + 1}")
        plt.axis('off')
    plt.show()
