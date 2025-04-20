from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

class DatasetSplit_CelebA(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, attr = self.dataset[self.idxs[item]]
        return image, attr[20] # return gender attribute


##########################################
#                  DCGAN                 #
##########################################

class LocalUpdate_DCGAN(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True) # , drop_last=True
        
    def train(self, gnet, dnet, iter, optg=None, optd=None):
        gnet.train()
        dnet.train()
        
        # train and update
        G_optimizer = torch.optim.Adam(gnet.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        D_optimizer = torch.optim.Adam(dnet.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        if optg:
            G_optimizer.load_state_dict(optg)
        if optd:
            D_optimizer.load_state_dict(optd)

        # if iter == int(0.5*self.args.gen_wu_epochs):
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 10
        #     print("learning rate change!")
        # elif iter == int(0.75*self.args.gen_wu_epochs):
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 10
        #     print("learning rate change!")

        g_epoch_loss = []
        d_epoch_loss = []

        # adversarial_loss = torch.nn.MSELoss()
        BCE_loss = nn.BCELoss().to(self.args.device)

        # label preprocess
        onehot = torch.zeros(10, 10)
        img_size = self.args.img_shape[1]
        batch_size = self.args.local_bs

        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1) # 10 x 10 eye matrix
        fill = torch.zeros([10, 10, img_size, img_size])
        for i in range(10):
            fill[i, i, :, :] = 1


        for iter in range(self.args.gen_local_ep):
            D_losses = []
            D_real_losses = []
            D_fake_losses = []
            G_losses = []

            y_real_ = torch.ones(batch_size).to(self.args.device)
            y_fake_ = torch.zeros(batch_size).to(self.args.device)
                
            for batch_idx, (x_, y_) in enumerate(self.ldr_train):
                ''' ---------------------------------
                Train Discriminator
                maximize log(D(x)) + log(1 - D(G(z)))
                --------------------------------- '''
                dnet.zero_grad()
                mini_batch = x_.size()[0]

                if mini_batch != batch_size:
                    y_real_ = torch.ones(mini_batch).to(self.args.device)
                    y_fake_ = torch.zeros(mini_batch).to(self.args.device)

                y_fill_ = fill[y_]
                x_, y_fill_ = x_.to(self.args.device), y_fill_.to(self.args.device)

                D_result = dnet(x_, y_fill_).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_)

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
                y_label_ = onehot[y_]
                y_fill_ = fill[y_]
                z_, y_label_, y_fill_ = z_.to(self.args.device), y_label_.to(self.args.device), y_fill_.to(self.args.device)

                G_result = gnet(z_, y_label_)
                D_result = dnet(G_result, y_fill_).squeeze()

                D_fake_loss = BCE_loss(D_result, y_fake_)
                D_fake_score = D_result.data.mean()

                D_train_loss = D_real_loss + D_fake_loss
                D_real_losses.append(D_real_loss)
                D_fake_losses.append(D_fake_loss)

                D_train_loss.backward()
                D_optimizer.step()

                D_losses.append(D_train_loss.data)

                ''' -------------------
                Train Generator
                maximize log(D(G(z)))
                ------------------- '''
                gnet.zero_grad()

                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
                y_label_ = onehot[y_]
                y_fill_ = fill[y_]
                z_, y_label_, y_fill_ = z_.to(self.args.device), y_label_.to(self.args.device), y_fill_.to(self.args.device)

                G_result = gnet(z_, y_label_)
                D_result = dnet(G_result, y_fill_).squeeze()

                G_train_loss = BCE_loss(D_result, y_real_)

                G_train_loss.backward()
                G_optimizer.step()

                G_losses.append(G_train_loss.data)
        
            g_epoch_loss.append(sum(G_losses)/len(G_losses))
            d_epoch_loss.append(sum(D_losses)/len(D_losses))
            # print('Real loss {:4f}, Fake loss{:4f}'.format(sum(D_real_losses)/len(D_real_losses), sum(D_fake_losses)/len(D_fake_losses)))

        try:
            return gnet.state_dict(), dnet.state_dict(), sum(g_epoch_loss) / len(g_epoch_loss), sum(d_epoch_loss) / len(d_epoch_loss), G_optimizer.state_dict(), D_optimizer.state_dict()
        except:
            return gnet.state_dict(), dnet.state_dict(), -1, -1


def loss_function_cvae(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    # recon_loss = F.binary_cross_entropy(pred.view(-1, 1024), x.view(-1, 1024), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld


class LocalUpdate_CVAE(object): # CVAE raw
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        if args.dataset == 'celebA':
            self.ldr_train = DataLoader(DatasetSplit_CelebA(dataset, idxs), batch_size=args.local_bs, shuffle=True) # , drop_last=True
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, opt=None):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.001)
        if opt:
            optimizer.load_state_dict(opt)
        epoch_loss = []
        
        for iter in range(self.args.gen_local_ep):
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device) # images.shape: torch.Size([batch_size, 1, 28, 28])
                if self.args.dataset == 'celebA':
                    label = labels.unsqueeze(-1).float()
                else:
                    label = np.zeros((images.shape[0], 10))
                    label[np.arange(images.shape[0]), labels] = 1
                    label = torch.tensor(label)
            
                # labels = one_hot(labels, 10).to(self.args.device)
                # recon_batch, mu, logvar = net(images, labels)
                optimizer.zero_grad()
                net.zero_grad()
                pred, mu, logvar = net(images, label.to(self.args.device))
                
                recon_loss, kld = loss_function_cvae(images, pred, mu, logvar)
                loss = recon_loss + kld
                loss.backward()
                optimizer.step()
                
                train_loss += loss.detach().cpu().numpy()
            epoch_loss.append(train_loss/len(self.ldr_train.dataset))
            optimizer.zero_grad()
                    
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), optimizer.state_dict()
    
            
##########################################
#                  DDPM                  #
##########################################

class LocalUpdate_DDPM(object): # DDPM
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = tqdm(DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True))
        self.lr = 1e-4

    def train(self, net, lr_decay_rate, opt=None):
        net.train()
        # train and update
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        if opt:
            optim.load_state_dict(opt)
        epoch_loss = []

        for iter in range(self.args.gen_local_ep):
            optim.param_groups[0]['lr'] = self.lr*lr_decay_rate
            # (1-(self.args.local_ep*(round-1) + iter)/(self.args.local_ep*(self.args.epochs+self.args.wu_epochs)))
            loss_ema = None
            batch_loss = []
            train_loss = 0
            for images, labels in self.ldr_train:
                optim.zero_grad()
                images = images.to(self.args.device) # images.shape: torch.Size([batch_size, 1, 28, 28])
                labels = labels.to(self.args.device)
                # images = images.view(-1, self.args.output_channel, self.args.img_size, self.args.img_size)
                # images = images.view(-1, 1, self.args.feature_size, self.args.feature_size) # self.args.local_bs
                # save_image(images.view(self.args.local_bs, 1, 14, 14),
                #             'imgFedCVAE/' + 'sample_' + '.png')
                loss = net(images, labels)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                optim.step()
                batch_loss.append(loss_ema)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), optim.state_dict()