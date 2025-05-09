'''
https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/tree/master

Conv2dtranspose
https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
https://velog.io/@hayaseleu/Transposed-Convolutional-Layer%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80
output shape of conv2dtranspose k=kernel, s=stride, p=padding
1d input length l -> l + (s-1)*(l-1) + 2*(k-p-1)
then Conv by k*k kernel, stride=1
output shape: l + (s-1)*(l-1) + 2*(k-p-1) - k + 1

torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
def output_shape(l,k,s,p):
    return l + (s-1)*(l-1) + 2*(k-p-1) - k + 1
DCGAN-F (args.output_channel x 32 x 32)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import one_hot
import numpy as np

# G(z)s
class generator(nn.Module):
    # initializers
    def __init__(self, args, d=128):
        super(generator, self).__init__()

        if args.dataset == 'celebA':
            self.deconv1_1 = nn.ConvTranspose2d(args.latent_dim, d * 8, 4, 1, 0)  # 1x1 → 4x4
            self.deconv1_1_bn = nn.BatchNorm2d(d * 8)
            self.deconv1_2 = nn.ConvTranspose2d(args.num_classes, d * 8, 4, 1, 0)  # 1x1 → 4x4
            self.deconv1_2_bn = nn.BatchNorm2d(d * 8)
            # 4x4 → 8x8
            self.deconv2 = nn.ConvTranspose2d(d * 16, d * 8, 4, 2, 1)
            self.deconv2_bn = nn.BatchNorm2d(d * 8)
            # 8x8 → 16x16
            self.deconv3 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
            self.deconv3_bn = nn.BatchNorm2d(d * 4)
            # 16x16 → 32x32
            self.deconv4 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
            self.deconv4_bn = nn.BatchNorm2d(d * 2)
            # 32x32 → 64x64 (Final output size for CelebA)
            self.deconv5 = nn.ConvTranspose2d(d * 2, args.output_channel, 4, 2, 1)
        
        else:
            self.deconv1_1 = nn.ConvTranspose2d(args.latent_dim, d*2, 4, 1, 0) # in ch, out ch, kernel size, stride, padding
            self.deconv1_1_bn = nn.BatchNorm2d(d*2)
            self.deconv1_2 = nn.ConvTranspose2d(args.num_classes, d*2, 4, 1, 0)
            self.deconv1_2_bn = nn.BatchNorm2d(d*2)
            self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
            self.deconv2_bn = nn.BatchNorm2d(d*2)
            self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
            self.deconv3_bn = nn.BatchNorm2d(d)
            self.deconv4 = nn.ConvTranspose2d(d, args.output_channel, 4, 2, 1)
            # self.deconv5 = nn.ConvTranspose2d(d*2, args.output_channel, 4, 2, 1)
            
        self.args = args

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        if len(input.shape) == 2:
            input = input.unsqueeze(2).unsqueeze(3)
        if len(label.shape) == 2:
            label = label.unsqueeze(2).unsqueeze(3) # Reshape label: [batch, label_dim] → [batch, label_dim, 1, 1]
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        
        if self.args.dataset == 'celebA':
            x = F.relu(self.deconv4_bn(self.deconv4(x)))  # → [batch, d*2, 32, 32]
            x = torch.tanh(self.deconv5(x)) 
        else:
            x = torch.tanh(self.deconv4(x))
        return x
    
    def sample_cond_image(self, args, sample_num=0,  condition_index=None):
        with torch.no_grad():
            z_ = torch.randn((sample_num, self.args.latent_dim)).view(-1, self.args.latent_dim, 1, 1).to(args.device)

            # Choose a specific condition based on the provided index
            if condition_index is None:
                y_ = torch.randint(self.args.num_classes, (sample_num,)).to(args.device)
            else:
                y_ = torch.tensor([condition_index] * sample_num).to(args.device)

            onehot = torch.zeros(self.args.num_classes, self.args.num_classes)
            onehot = onehot.scatter_(1, torch.LongTensor(np.arange(self.args.num_classes)).view(self.args.num_classes,1), 1).view(self.args.num_classes, self.args.num_classes, 1, 1) # 10 x 10 eye matrix
            onehot = onehot.to(self.args.device)

            y_label_ = onehot[y_]
            y_label_ = y_label_.to(self.args.device)
            gen_imgs = self.forward(z_, y_label_)

            # c = torch.randint(10, (args.batch_size, )).to(args.device) # MAX_NUM, (SIZE, )
            one_c = one_hot(y_, self.args.num_classes).to(args.device)

            return gen_imgs, one_c

    def sample_image(self, args, sample_num=0):
        with torch.no_grad():
        
            z_ = torch.randn((sample_num, self.args.latent_dim)).view(-1, self.args.latent_dim, 1, 1).to(args.device)
            y_ = (torch.rand(sample_num, 1) * self.args.num_classes).type(torch.LongTensor).squeeze()

            onehot = torch.zeros(self.args.num_classes, self.args.num_classes)
            onehot = onehot.scatter_(1, torch.LongTensor(range(self.args.num_classes)).view(self.args.num_classes,1), 1).view(self.args.num_classes, self.args.num_classes, 1, 1) # 10 x 10 eye matrix

            y_label_ = onehot[y_]
            y_label_ = y_label_.to(args.device)
            gen_imgs = self.forward(z_, y_label_)
            
            # c = torch.randint(10, (args.batch_size, )).to(args.device) # MAX_NUM, (SIZE, )
            one_c = one_hot(y_, args.num_classes).to(args.device)

            return gen_imgs, one_c

    def sample_image_4visualization(self, sample_num):
        
        with torch.no_grad():
            z_ = torch.randn((sample_num, self.args.latent_dim)).view(-1, self.args.latent_dim, 1, 1).to(self.args.device)
            labels = torch.arange(0,self.args.num_classes).to(self.args.device) # context for us just cycles throught the mnist labels
            labels = labels.repeat(int(sample_num/labels.shape[0]))

            onehot = torch.zeros(self.args.num_classes, self.args.num_classes)
            onehot = onehot.scatter_(1, torch.LongTensor(range(self.args.num_classes)).view(self.args.num_classes,1), 1).view(self.args.num_classes, self.args.num_classes, 1, 1) # 10 x 10 eye matrix
            onehot = onehot.to(self.args.device)
            
            y_label_ = onehot[labels]
            y_label_ = y_label_.to(self.args.device)
            gen_imgs = self.forward(z_, y_label_)

            return gen_imgs


class discriminator(nn.Module):
    # initializers
    def __init__(self, args, d=128):
        super(discriminator, self).__init__()
        self.args = args
        # self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1) # 1 64
        self.conv1_1 = nn.Conv2d(args.output_channel, int(d/2), 4, 2, 1) # 1 64
        self.conv1_2 = nn.Conv2d(args.num_classes, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) # 128 256
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1) # 256 512
        self.conv3_bn = nn.BatchNorm2d(d*4)
        if args.dataset == 'celebA':
            self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1) # 512 1
            self.conv4_bn = nn.BatchNorm2d(d*8)
            self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)    
        else:
            self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0) 

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label): # input 1 x 32 x 32
        x = F.leaky_relu(self.conv1_1(input), 0.2) # 32 * 32 * 32
        y = F.leaky_relu(self.conv1_2(label), 0.2) # 32 * 32 * 32
        x = torch.cat([x, y], 1) # 64 * 32 * 32
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2) # 128 * 16 * 16
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2) # 256 * 8 * 8
        if self.args.dataset == 'celebA':
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2) # 512 * 4 * 4 
            x = torch.sigmoid(self.conv5(x)) #
        else:
            x = torch.sigmoid(self.conv4(x)) #

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()