import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import pytorch_ssim
from dataset import DatasetFromFolder, DatasetFromFolderEval
from loss import GeneratorLoss, DiscriminatorLoss
from model import Generator, Discriminator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--patch_size', type=int, default=96)
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8])
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--adv_coefficient', type=int, default=1e-3)
parser.add_argument('--cuda', action='store_true', default=False)
opt = parser.parse_args()

device = torch.device(
    'cuda:0' if opt.cuda and torch.cuda.is_available() else 'cpu')

print('===> Loading Train Dataset')
train_set = DatasetFromFolder(image_dir='./data/General-100/train',
                              patch_size=opt.patch_size,
                              upscale_factor=opt.upscale_factor,
                              data_augmentation=True)
train_loader = DataLoader(train_set, shuffle=True)

print('===> Loading Validation Dataset')
val_set = DatasetFromFolderEval(image_dir='./data/General-100/val',
                                upscale_factor=opt.upscale_factor,)
val_loader = DataLoader(val_set, shuffle=False)

print('===> Building Model')
netG = Generator(opt.upscale_factor).to(device)
print('# Generator parameters:', sum(p.numel() for p in netG.parameters()))
netD = Discriminator().to(device)
print('# Discriminator parameters:', sum(p.numel() for p in netD.parameters()))

criterionG = GeneratorLoss().to(device)
criterionD = DiscriminatorLoss().to(device)

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

for epoch in range(1, opt.num_epochs + 1):
    netG.train(), netD.train()
    for iteration, (lr_img, hr_img) in enumerate(train_loader, 1):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        # Generate image
        sr_img = netG(lr_img)

        # Update D
        optimizerD.zero_grad()
        real_out = netD(hr_img).mean()
        fake_out = netD(sr_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward()
        optimizerD.step()

        # Update G
        criterionG.zero_grad()
        g_loss = criterionG(sr_img, hr_img, fake_out)
        g_loss.backward()
        optimizerG.step()

        print('[Epoch{}({}/{})] G_Loss: {:.6f}, D_Loss: {:.6f}'.format(epoch, iteration, len(train_loader), g_loss, d_loss))

    save_image(sr_img, './tmp/{}.png'.format(epoch))

    netG.eval()
    with torch.no_grad():
        for (lr_img, hr_img, filename) in val_loader:
            #print(lr_img.shape, hr_img.shape, filename)
            pass
