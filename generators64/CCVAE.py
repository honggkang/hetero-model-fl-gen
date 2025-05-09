'''
https://github.com/debtanu177/CVAE_MNIST/blob/master/train_cvae.py
https://github.com/sksq96/pytorch-vae/blob/master/vae.py
https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/nn/benchmarks/cifar/convnets.py
https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
CVAE (args.output_channel x 32 x 32)
'''

import torch 
import numpy as np
import torch.nn as nn
import torch.nn. functional as F
from utils.util import one_hot
import numpy as np

class CCVAE(nn.Module):
    def __init__(self, args):
        super(CCVAE,self).__init__()
        # self.latent_size = latent_size (16)
        # self.num_classes = args.num_classes
        self.args = args

        # For encode
        # conv_in_dim = args.output_channel+self.args.num_classes if args.dataset == 'celebA' else args.output_channel+1
        conv_in_dim = args.output_channel+1
        self.conv1 = nn.Conv2d(conv_in_dim, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        if self.args.img_size == 64:
            # encoder
            self.conv_last = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1)
            self.bn_last = nn.BatchNorm2d(2048)
            self.fc_dim = 2048

            # decoder
            self.deconv_first = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
            self.bn_first = nn.BatchNorm2d(1024)
        else:
            self.fc_dim = 1024
        
        self.mu = nn.Linear(self.fc_dim, args.latent_size)
        self.logvar = nn.Linear(self.fc_dim, args.latent_size)

        # For decoder
        self.linear = nn.Linear(args.latent_size + args.num_classes, self.fc_dim)
        self.conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv10 = nn.ConvTranspose2d(64, args.output_channel, kernel_size=4, stride=2, padding=1)

    def encoder(self,x,y):
        if self.args.dataset == 'celebA':
            y = y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])  # Reshape y to match image dims
        else:
            y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
            y = torch.ones([x.shape[0], 1, x.shape[2], x.shape[3]]).to(self.args.device)*y # let's make only one channel for label
        t = torch.cat((x,y),dim=1) # 32 x 32
        # implement BN
        t = F.relu(self.bn1(self.conv1(t))) # 16
        t = F.relu(self.bn2(self.conv2(t))) # 8
        t = F.relu(self.bn3(self.conv3(t))) # 4
        t = F.relu(self.bn4(self.conv4(t))) # 2
        t = F.relu(self.bn5(self.conv5(t))) # 1
        if self.args.img_size == 64:
            t = F.relu(self.bn_last(self.conv_last(t))) # 1
        t = t.reshape((x.shape[0], -1))
        
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.args.device)
        return eps*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], self.fc_dim, 1, 1)) # linear

    def decoder(self, z):
        t = F.relu(self.linear(z))
        t = self.unFlatten(t) # 1024 *1 *1
        if self.args.img_size == 64:
            t = F.relu(self.bn_first(self.deconv_first(t)))
        t = F.relu(self.bn6(self.conv6(t))) # 512 *2 *2
        t = F.relu(self.bn7(self.conv7(t))) # 256 *4 *4
        t = F.relu(self.bn8(self.conv8(t))) # 128 *8 *8
        t = F.relu(self.bn9(self.conv9(t))) # 64 *16 *16
        t = torch.tanh(self.conv10(t)) # args.output_channel *32 *32
        return t

    def forward(self, x, y):
        mu, logvar = self.encoder(x,y)
        z = self.reparameterize(mu, logvar)
        # Class conditioning
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar
    
    def sample_image(self, args, sample_num=0):
        with torch.no_grad():
            z = torch.randn(sample_num, args.latent_size).to(self.args.device)

            if args.dataset == 'celebA':
                y = torch.randint(0, 2, (sample_num, self.args.num_classes)).float().to(self.args.device)
                z = torch.cat((z, y), dim=1)  # Concatenate latent vector with labels
                pred = self.decoder(z)
                
                return pred, y
            
            else:
                y = (torch.rand(sample_num, 1) * self.args.num_classes).type(torch.LongTensor).squeeze()
                label = np.zeros((y.shape[0], self.args.num_classes))
                label[np.arange(z.shape[0]), y] = 1
                label = torch.tensor(label)
                pred = self.decoder(torch.cat((z, label.float().to(self.args.device)), dim=1))
                one_c = one_hot(y, args.num_classes).to(self.args.device)

                return pred, one_c

    def sample_cond_image(self, args, sample_num=0,  condition_index=None):
        with torch.no_grad():
            if args.dataset == 'celebA':
                return self.sample_image(sample_num, condition_index)
            
            else:
                z = torch.randn(sample_num, args.latent_size).to(self.args.device)

                # Choose a specific condition based on the provided index
                if condition_index is None:
                    y = (torch.rand(sample_num, 1) * self.args.num_classes).type(torch.LongTensor).squeeze()
                else:
                    y = torch.tensor([condition_index] * sample_num)

                label = np.zeros((y.shape[0], self.args.num_classes))
                label[np.arange(z.shape[0]), y] = 1
                label = torch.tensor(label)
                pred = self.decoder(torch.cat((z, label.float().to(self.args.device)), dim=1))
                one_c = one_hot(y, args.num_classes).to(self.args.device)

                return pred, one_c


    def sample_image_4visualization(self, sample_num):
        sample_num = self.args.num_classes if sample_num < self.args.num_classes else sample_num
        z = torch.randn(sample_num, self.args.latent_size).to(self.args.device)

        if self.args.dataset == 'celebA':
            y = torch.randint(0, 2, (sample_num, self.args.num_classes)).float().to(self.args.device)
            with torch.no_grad():
                z = torch.cat((z, y), dim=1)
                pred = self.decoder(z)
        else:
            y = torch.arange(0, self.args.num_classes)
            y = y.repeat(int(sample_num/y.shape[0])) 

            with torch.no_grad():
                label = np.zeros((y.shape[0], self.args.num_classes))

                label[np.arange(z.shape[0]), y] = 1
                label = torch.tensor(label)

                pred = self.decoder(torch.cat((z, label.float().to(self.args.device)), dim=1))
        return pred