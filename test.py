from pathlib import Path
from math import log10

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from PIL import Image

from dataset import DatasetFromFolder, DatasetFromFolderEval
from model import Generator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/Set5')
parser.add_argument('--weight_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8])
parser.add_argument('--cuda', action='store_true', default=False)
opt = parser.parse_args()

torch.manual_seed(123)
if opt.cuda:
    torch.cuda.manual_seed(123)

device = torch.device(
    'cuda:0' if opt.cuda and torch.cuda.is_available() else 'cpu')

print('===> Loading validation dataset')
test_set = DatasetFromFolderEval(image_dir=Path(opt.data_dir),
                                upscale_factor=opt.upscale_factor)
test_loader = DataLoader(test_set, shuffle=False)

print('===> Building model')
model = Generator(opt.upscale_factor).to(device)
criterion = nn.MSELoss().to(device)

model.load_state_dict(torch.load(opt.weight_path, map_location=device))

model.eval()
total_loss, total_psnr = 0, 0
total_loss_b, total_psnr_b = 0, 0
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch[0], batch[1]
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        total_loss += loss.data
        total_psnr += 10 * log10(1 / loss.data)

        inputs = F.to_pil_image(inputs.squeeze(0).cpu())
        inputs = inputs.resize((inputs.size[0] *  opt.upscale_factor, inputs.size[1] *  opt.upscale_factor), Image.BICUBIC)
        inputs = F.to_tensor(inputs)
        loss = criterion(inputs, targets)
        total_loss_b += loss.data
        total_psnr_b += 10 * log10(1 / loss.data)

        save_image(prediction, Path(opt.save_dir) / '{}_sr.png'.format(batch[2][0]), nrow=1)
        save_image(inputs, Path(opt.save_dir) / '{}_lr.png'.format(batch[2][0]), nrow=1)
        save_image(targets, Path(opt.save_dir) / '{}_hr.png'.format(batch[2][0]), nrow=1)

print("===> [Bicubic] Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(total_loss_b / len(test_loader), total_psnr_b / len(test_loader)))
print("===> [SRCNN] Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(total_loss / len(test_loader), total_psnr / len(test_loader)))

