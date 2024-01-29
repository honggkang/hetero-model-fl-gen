'''
https://github.com/debtanu177/CVAE_MNIST/blob/master/train_cvae.py
https://github.com/sksq96/pytorch-vae/blob/master/vae.py
https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/nn/benchmarks/cifar/convnets.py
https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
CVAE-F (args.output_channel x 16 x 16)
'''

import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import traceback
from utils.util import one_hot


batch_size = 100
learning_rate = 1e-3
max_epoch = 100
device = torch.device("cuda")
num_workers = 5
load_epoch = -1
generate = True


class CCVAE(nn.Module):
    def __init__(self, args):
        super(CCVAE,self).__init__()
        # self.latent_size = latent_size (16)
        # self.num_classes = args.num_classes
        self.args = args

        # For encode
        self.conv1 = nn.Conv2d(args.output_channel+1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.mu = nn.Linear(int(512/256*args.img_size*args.img_size), args.latent_size)
        self.logvar = nn.Linear(int(512/256*args.img_size*args.img_size), args.latent_size)

        # For decoder
        self.linear = nn.Linear(args.latent_size + args.num_classes, int(512/256*args.img_size*args.img_size))
        self.conv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.ConvTranspose2d(64, args.output_channel, kernel_size=4, stride=2, padding=1)

    def encoder(self,x,y):
        y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        y = torch.ones([x.shape[0], 1, x.shape[2], x.shape[3]]).to(self.args.device)*y # let's make only one channel for label
        t = torch.cat((x,y),dim=1)
        # implement BN
        t = F.relu(self.bn1(self.conv1(t))) # 8*8
        t = F.relu(self.bn2(self.conv2(t))) # 4*4
        t = F.relu(self.bn3(self.conv3(t))) # 2*2
        t = F.relu(self.bn4(self.conv4(t))) # 1*1
        t = t.reshape((x.shape[0], -1))

        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.args.device)
        return eps*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 512, 1, 1)) # linear

    def decoder(self, z):
        t = F.relu(self.linear(z))
        t = self.unFlatten(t) # 512 *1 *1
        t = F.relu(self.bn5(self.conv5(t))) # 256 *2 *2
        t = F.relu(self.bn6(self.conv6(t))) # 128 *4 *4
        t = F.relu(self.bn7(self.conv7(t))) # 64 *8 *8
        t = F.relu(self.conv8(t)) # 3 *16 *16
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
            y = (torch.rand(sample_num, 1) * 10).type(torch.LongTensor).squeeze()

            label = np.zeros((y.shape[0], 10))
            label[np.arange(z.shape[0]), y] = 1
            label = torch.tensor(label)
            pred = self.decoder(torch.cat((z, label.float().to(self.args.device)), dim=1))
            one_c = one_hot(y, args.num_classes).to(self.args.device)

        return pred, one_c

    def sample_image_4visualization(self, sample_num):
        z = torch.randn(sample_num, self.args.latent_size).to(self.args.device)
        y = torch.arange(0,10)
        y = y.repeat(int(sample_num/y.shape[0]))

        with torch.no_grad():
            label = np.zeros((y.shape[0], 10))
            label[np.arange(z.shape[0]), y] = 1
            label = torch.tensor(label).float().to(self.args.device)

            pred = self.decoder(torch.cat((z,label), dim=1))
        return pred
        
def plot(epoch, pred, y,name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16,16))
    for i in range(6):
        ax = fig.add_subplot(3,2,i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
        ax.title.set_text(str(y[i]))
    plt.savefig("./images/{}epoch_{}.jpg".format(name, epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld


def train(epoch, model, train_loader, optim):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i,(x,y) in enumerate(train_loader):
        try:
            label = np.zeros((x.shape[0], 10))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)

            optim.zero_grad()   
            pred, mu, logvar = model(x.to(device),label.to(device))
            
            recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]
            if i == 0:
                print("Gradients")
                for name,param in model.named_parameters():
                    if "bias" in name:
                        print(name,param.grad[0],end=" ")
                    else:
                        print(name,param.grad[0,0],end=" ")
                    print()
        except Exception as e:
            traceback.print_exe()
            torch.cuda.empty_cache()
            continue
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss

def test(epoch, model, test_loader):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                label = np.zeros((x.shape[0], 10))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x.to(device),label.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    # print("gr:", x[0,0,:5,:5])
                    # print("pred:", pred[0,0,:5,:5])
                    plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
            except Exception as e:
                traceback.print_exe()
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        


def generate_image(epoch,z, y, model):
    with torch.no_grad():
        label = np.zeros((y.shape[0], 10))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)

        pred = model.decoder(torch.cat((z.to(device),label.float().to(device)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy(),name='Eval_')
        print("data Plotted")


def load_data():
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)



if __name__ == "__main__":
    train_loader, test_loader = load_data()
    print("dataloader created")
    model = CCVAE().to(device)
    print("model created")
    
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("model {} loaded".format(load_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch+1, max_epoch):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader)
            if generate:
                z = torch.randn(6, 32).to(device)
                y = torch.tensor([1,2,3,4,5,6]) - 1
                generate_image(i,z, y, model)
            
        print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss:{}".format(i, max_epoch,train_total, train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kld, test_loss))

        save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))


    # i, (example_data, exaple_target) = next(enumerate(test_loader))
    # print(example_data[0,0].shape)
    # plt.figure(figsize=(5,5), dpi=100)
    # plt.imsave("example.jpg", example_data[0,0], cmap='gray',  dpi=1000)