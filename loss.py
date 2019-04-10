from torch import nn
from torchvision.models.vgg import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self, loss_type='vgg22', adv_coefficient=1e-3):
        super(GeneratorLoss, self).__init__()
        self.adv_coefficient = adv_coefficient
        if loss_type in ['vgg22', 'vgg54']:
            self.content_loss = VGGLoss(loss_type)
        elif loss_type == 'mse':
            self.content_loss = nn.MSELoss()

    def forward(self, fake_img, target_img, fake_label):
        content_loss = self.content_loss(fake_img, target_img)
        adversarial_loss = -fake_label.log().mean()
        return content_loss + self.adv_coefficient * adversarial_loss


class VGGLoss(nn.Module):
    def __init__(self, vgg_type='vgg22'):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        if vgg_type == 'vgg22':
            self.net = nn.Sequential(*list(vgg.features[:7])).eval()
        elif vgg_type == 'vgg54':
            self.net = nn.Sequential(*list(vgg.features[:30])).eval()

        for param in self.net.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.mse_loss(self.net(input), self.net(target))


class DiscriminatorLoss(nn.Module):
    def __call__(self, real_out, fake_out):
        return (1 - real_out) + fake_out
