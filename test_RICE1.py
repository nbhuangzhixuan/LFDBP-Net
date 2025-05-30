import os

import pyiqa
import time


from torch.utils.data import DataLoader
from datasets.RICE_datasets import TestDataset
from tqdm import tqdm
from models.Model_atp import DecloudingNetwork,Bottleneck
from utils.utils_test import to_psnr, to_ssim_skimage
import torch
import argparse

parser = argparse.ArgumentParser(description="Evaluate Declouding Network")

parser.add_argument('--dataset_root', type=str, required=True,
                    help='Root directory of the CSRD dataset')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
g_j_before = Bottleneck(width=96)
decloud_net = DecloudingNetwork(g_j_before)


g_j_before.to(device)
decloud_net.to(device)

brisque = pyiqa.create_metric('brisque', device=device)
niqe = pyiqa.create_metric('niqe', device=device)

decloudnet_weight_path = os.path.join("output_result/RICE1_weights","decloud_net.pth")
decloud_net.load_state_dict(torch.load(decloudnet_weight_path))

print('===> Loading test datasets')


test_dataset = TestDataset(args.dataset_root)
test_loader = DataLoader(dataset=test_dataset, num_workers=8,
                                    batch_size=1, shuffle=False)





with torch.no_grad():
    psnr_list = []
    ssim_list = []
    BRISQUE_list = []
    NIQE_list = []
    decloud_net.eval()
    for data_val in tqdm(test_loader):
        clean = data_val[1].to(device)
        hazy = data_val[0].to(device)

        _, _, _, _,img,_,_ = decloud_net(hazy)

        psnr_list.extend(to_psnr(img, clean))
        ssim_list.extend(to_ssim_skimage(img, clean))
        NIQE_list.append(niqe(img).item())
        BRISQUE_list.append(brisque(img).item())

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    avr_BRISQUE = sum(BRISQUE_list) / len(BRISQUE_list)
    avr_NIQE = sum(NIQE_list) / len(NIQE_list)

    print(
        f'PSNR: {avr_psnr:.2f}, SSIM: {avr_ssim:.3f}, BRISQUE: {avr_BRISQUE:.2f}, NIQE: {avr_NIQE:.3f}')
