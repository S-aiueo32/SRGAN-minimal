from math import log10
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter

from tqdm import tqdm

from dataset import DatasetFromFolder, DatasetFromFolderEval
from loss import GeneratorLoss, DiscriminatorLoss
from model import Generator, Discriminator
from pytorch_ssim import ssim

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../data/VOC2012')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--patch_size', type=int, default=96)
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8])
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--pretrain_epochs', type=int, default=0)
parser.add_argument('--loss_type', type=str, default='vgg22', choices=['vgg22', 'vgg54'])
parser.add_argument('--adv_coefficient', type=int, default=1e-3)
parser.add_argument('--cuda', action='store_true', default=False)
opt = parser.parse_args()

torch.manual_seed(123)
if opt.cuda:
    torch.cuda.manual_seed(123)

device = torch.device(
    'cuda:0' if opt.cuda and torch.cuda.is_available() else 'cpu')

print('===> Loading train dataset')
train_set = DatasetFromFolder(image_dir=Path(opt.data_root) / 'train',
                              patch_size=opt.patch_size,
                              upscale_factor=opt.upscale_factor,
                              data_augmentation=True)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

print('===> Loading validation dataset')
val_set = DatasetFromFolderEval(image_dir=Path(opt.data_root) / 'val',
                                upscale_factor=opt.upscale_factor,)
val_loader = DataLoader(val_set, shuffle=False)

print('===> Building model')
netG = Generator(opt.upscale_factor).to(device)
print('# Generator parameters:', sum(p.numel() for p in netG.parameters()))
netD = Discriminator(opt.patch_size).to(device)
print('# Discriminator parameters:', sum(p.numel() for p in netD.parameters()))

print('===> Defining criterions')
mse_loss = nn.MSELoss().to(device)
criterionG = GeneratorLoss(opt.loss_type, opt.adv_coefficient).to(device)
criterionD = DiscriminatorLoss().to(device)

print('===> Defining optimizers')
optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=1e-4)

writer = SummaryWriter()
log_dir = Path(writer.log_dir)
sample_dir = log_dir / 'sample'
sample_dir.mkdir(exist_ok=True)
weight_dir = log_dir / 'weights'
weight_dir.mkdir(exist_ok=True)

global_step = 0
for epoch in range(1, opt.num_epochs + 1):
    netG.train(), netD.train()
    for iteration, (input_img, real_img) in enumerate(train_loader, 1):
        input_img = input_img.to(device)
        real_img = real_img.to(device)
        fake_img = netG(input_img)

        # Update D
        optimizerD.zero_grad()
        d_out_real = netD(real_img)
        d_out_fake = netD(fake_img)
        d_loss = criterionD(d_out_real, d_out_fake)
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        # Update G
        optimizerG.zero_grad()
        g_loss = criterionG(d_out_fake, real_img, fake_img)
        g_loss.backward()
        optimizerG.step()

        writer.add_scalar('train/batch_g_loss', g_loss, global_step)
        writer.add_scalar('train/batch_d_loss', d_loss, global_step)
        writer.add_scalar('train/batch_d_out_real', d_out_real.mean(), global_step)
        writer.add_scalar('train/batch_d_out_fake', d_out_fake.mean(), global_step)

        writer.add_histogram('train/fake_img_hist', fake_img, global_step)
        writer.add_histogram('train/d_out_real_hist', d_out_real, global_step)
        writer.add_histogram('train/d_out_fake_hist', d_out_fake, global_step)

        print('[Epoch{}({}/{})]'.format(epoch, iteration, len(train_loader)), 
              'G_Loss: {:.6f}, D_Loss: {:.6f},'.format(g_loss, d_loss),
              'D(x): {:.6f}, D(G(z)): {:.6f}'.format(d_out_real.mean(), d_out_fake.mean()))

        global_step += 1

    grid_sr = make_grid(fake_img[:9], nrow=3)
    grid_hr = make_grid(real_img[:9], nrow=3)
    writer.add_image('train/sr_img', grid_sr, epoch)
    writer.add_image('train/hr_img', grid_hr, epoch)
    save_image(grid_sr, sample_dir / 'train_result_epoch{:05}_sr.png'.format(epoch), nrow=1)
    save_image(grid_hr, sample_dir / 'train_result_epoch{:05}_hr.png'.format(epoch), nrow=1)

    netG.eval()
    with torch.no_grad():
        val_psnr = val_ssim = 0
        img_pool = []
        for iteration, (input_img, real_img, filename) in enumerate(tqdm(val_loader), 1):
            input_img = input_img.to(device)
            real_img = real_img.to(device)
            fake_img = netG(input_img)

            val_psnr += 10 * log10(1 / mse_loss(fake_img, real_img).item())
            val_ssim += ssim(fake_img, real_img).item()

            if iteration <= 3:
                img_pool.append([fake_img, filename])

            if iteration == len(val_loader):
                val_psnr /= len(val_loader)
                val_ssim /= len(val_loader)

        for i, (img, filename) in enumerate(img_pool):
            print('[Sample] {} ===>  eval_result_epoch{:05}_{:02}.png'.format(filename, epoch, i))
            writer.add_images('val/sr_img_{}'.format(i), img, epoch)
            save_image(img, sample_dir / 'eval_result_epoch{:05}_{:02}.png'.format(epoch, i), nrow=1)
            
        writer.add_scalar('val/psnr', val_psnr, epoch)
        writer.add_scalar('val/ssim', val_ssim, epoch)
        print('===> [Validation score][Epoch {}] PSNR:{:.6f}, SSIM:{:.6f}'.format(epoch, val_psnr, val_ssim))

    if global_step % 1e+5 == 0:
        for param_group in optimizerG.param_groups:
                param_group['lr'] /= 10.0
        for param_group in optimizerD.param_groups:
                param_group['lr'] /= 10.0

    torch.save(netG.state_dict(), str(weight_dir / 'weight_epoch{:05}.pth'.format(epoch)))