import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter

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

torch.manual_seed(123)
if opt.cuda:
    torch.cuda.manual_seed(123)

device = torch.device(
    'cuda:0' if opt.cuda and torch.cuda.is_available() else 'cpu')

print('===> Loading train dataset')
train_set = DatasetFromFolder(image_dir='../data/VOC2012/train/',
                              patch_size=opt.patch_size,
                              upscale_factor=opt.upscale_factor,
                              data_augmentation=True)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

print('===> Loading validation dataset')
val_set = DatasetFromFolderEval(image_dir='../data/VOC2012/val/',
                                upscale_factor=opt.upscale_factor,)
val_loader = DataLoader(val_set, shuffle=False)

print('===> Building model')
netG = Generator(opt.upscale_factor).to(device)
print('# Generator parameters:', sum(p.numel() for p in netG.parameters()))
netD = Discriminator().to(device)
print('# Discriminator parameters:', sum(p.numel() for p in netD.parameters()))

print('===> Definig criterions')
criterionG = GeneratorLoss().to(device)
criterionD = DiscriminatorLoss().to(device)

print('===> Definig optimizers')
optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=1e-4)

writer = SummaryWriter()

global_step = 0
for epoch in range(1, opt.num_epochs + 1):
    netG.train(), netD.train()
    for iteration, (input_img, real_img) in enumerate(train_loader, 1):
        input_img = input_img.to(device)
        real_img = real_img.to(device)

        # Generate image
        fake_img = netG(input_img)

        # Update D
        optimizerD.zero_grad()
        d_out_real = netD(real_img)
        d_out_fake = netD(fake_img)
        d_loss = criterionD(d_out_real, d_out_fake)
        if epoch >= 2:
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
              ' D(x): {}, D(G(z)): {}'.format(d_out_real.mean(), d_out_fake.mean()))

        global_step += 1

    grid_img = make_grid(fake_img[:9], nrow=3, normalize=True, range=(-1, 1))
    writer.add_image('train/batch_image', grid_img, global_step)
    save_image(grid_img, './tmp/train_result_epoch{:05}.png'.format(epoch), nrow=1)

    netG.eval()
    with torch.no_grad():
        for (lr_img, hr_img, filename) in val_loader:
            #print(lr_img.shape, hr_img.shape, filename)
            pass
