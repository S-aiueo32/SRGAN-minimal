import torch
from torch import nn
from torchvision.models.vgg import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self, loss_type='vgg22', adv_coefficient=1e-3):
        super(GeneratorLoss, self).__init__()
        self.content_loss = VGGLoss(loss_type)
        self.mse_loss = nn.MSELoss()
        self.adv_coefficient = adv_coefficient
        self.bce_loss = nn.BCELoss()

    def forward(self, d_out_fake, real_img, fake_img):
        mse_loss = self.mse_loss(real_img, fake_img)
        content_loss = self.content_loss(real_img, fake_img)
        adv_loss = torch.mean(-torch.log(d_out_fake + 1e-3))
        return mse_loss + 2e-6 * content_loss + self.adv_coefficient * adv_loss


class VGGLoss(nn.Module):
    def __init__(self, loss_type):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        if loss_type == 'vgg22':
            vgg_net = nn.Sequential(*list(vgg.features[:9]))
        elif loss_type == 'vgg54':
            vgg_net = nn.Sequential(*list(vgg.features[:36]))
        
        for param in vgg_net.parameters():
            param.requires_grad = False

        self.vgg_net = vgg_net.eval()
        self.mse_loss = nn.MSELoss()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406], requires_grad=False))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225], requires_grad=False))

    def forward(self, real_img, fake_img):
        real_img = real_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        fake_img = fake_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        feature_real = self.vgg_net(real_img)
        feature_fake = self.vgg_net(fake_img)
        return self.mse_loss(feature_real, feature_fake)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, d_out_real, d_out_fake):
        loss_real = self.bce_loss(d_out_real, torch.ones_like(d_out_real))
        loss_fake = self.bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        return loss_real + loss_fake
