import os
import re
import torch

def save_checkpoint(epoch, decloud_net,cloud_net,g_j_before, discriminator,discriminator_after,discriminator_i,discriminator_j_g,optimizer_T, optimizer_D,optimizer_D_after,optimizer_D_i,optimizer_D_j_g,scheduler_T, scheduler_D,scheduler_D_after,scheduler_D_i,scheduler_D_j_g, psnr, ssim,brique,musiq, model_save_dir):
    checkpoint = {
        'epoch': epoch,
        'decloud_net_state_dict': decloud_net.state_dict(),
        'cloud_net_state_dict': cloud_net.state_dict(),
        'g_j_before_state_dict': g_j_before.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'discriminator_after_state_dict': discriminator_after.state_dict(),
        'discriminator_i_state_dict': discriminator_i.state_dict(),
        'discriminator_j_g_state_dict': discriminator_j_g.state_dict(),
        'optimizer_T_state_dict': optimizer_T.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'optimizer_D_after_state_dict': optimizer_D_after.state_dict(),
        'optimizer_D_i_state_dict': optimizer_D_i.state_dict(),
        'optimizer_D_j_g_state_dict': optimizer_D_j_g.state_dict(),
        'scheduler_T_state_dict': scheduler_T.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict(),
        'scheduler_D_after_state_dict': scheduler_D_after.state_dict(),
        'scheduler_D_i_state_dict': scheduler_D_i.state_dict(),
        'scheduler_D_j_g_state_dict': scheduler_D_j_g.state_dict(),
        'psnr': psnr,
        'ssim': ssim
    }
    model_path = os.path.join(model_save_dir, f'checkpoint_epoch{epoch}_psnr{round(psnr, 2)}_ssim{round(ssim, 3)}_brisque{round(brique, 2)}_musiq{round(musiq, 3)}.pth')
    torch.save(checkpoint, model_path)

def load_checkpoint(model_path, decloud_net,cloud_net,g_j_before,discriminator,discriminator_after,discriminator_i,discriminator_j_g,optimizer_T, optimizer_D,optimizer_D_after,optimizer_D_i,optimizer_D_j_g,scheduler_T, scheduler_D,scheduler_D_after,scheduler_D_i,scheduler_D_j_g):
    checkpoint = torch.load(model_path)
    decloud_net.load_state_dict(checkpoint['decloud_net_state_dict'])
    cloud_net.load_state_dict(checkpoint['cloud_net_state_dict'])

    g_j_before.load_state_dict(checkpoint['g_j_before_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    discriminator_after.load_state_dict(checkpoint['discriminator_after_state_dict'])
    discriminator_i.load_state_dict(checkpoint['discriminator_i_state_dict'])
    discriminator_j_g.load_state_dict(checkpoint['discriminator_j_g_state_dict'])
    # optimizer_T.load_state_dict(checkpoint['optimizer_T_state_dict'])
    # optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    # optimizer_D_after.load_state_dict(checkpoint['optimizer_D_after_state_dict'])
    # optimizer_D_i.load_state_dict(checkpoint['optimizer_D_i_state_dict'])
    # optimizer_D_j_g.load_state_dict(checkpoint['optimizer_D_j_g_state_dict'])
    # scheduler_T.load_state_dict(checkpoint['scheduler_T_state_dict'])
    # scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
    # scheduler_D_after.load_state_dict(checkpoint['scheduler_D_after_state_dict'])
    # scheduler_D_i.load_state_dict(checkpoint['scheduler_D_i_state_dict'])
    # scheduler_D_j_g.load_state_dict(checkpoint['scheduler_D_j_g_state_dict'])



    epoch = checkpoint['epoch']
    psnr = checkpoint['psnr']
    ssim = checkpoint['ssim']
    return epoch, psnr, ssim


def load_checkpoint_test(model_path, decloud_net,cloud_net,g_j_before):
    checkpoint = torch.load(model_path)
    decloud_net.load_state_dict(checkpoint['decloud_net_state_dict'])
    cloud_net.load_state_dict(checkpoint['cloud_net_state_dict'])

    g_j_before.load_state_dict(checkpoint['g_j_before_state_dict'])


def find_latest_checkpoint(dir_path, prefix):
    checkpoint_files = [f for f in os.listdir(dir_path) if f.startswith(prefix)]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.findall(r'epoch(\d+)', x)[0]))
    return os.path.join(dir_path, latest_checkpoint)
