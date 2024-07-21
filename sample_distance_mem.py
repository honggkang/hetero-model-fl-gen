'''
skeleton code for evaluating mean NND ratio
'''
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
import lpips

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist') # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument("--output_channel", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--model_type", type=str, default='ddpm') 
### DCGAN
parser.add_argument('--latent_dim', type=int, default=100)
### VAE parameters
parser.add_argument('--latent_size', type=int, default=16) # local epochs for training generator
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128)
parser.add_argument('--n_T', type=int, default=100)
parser.add_argument('--guide_w', type=float, default=0.0) # 0, 0.5, 2
### CUDA
parser.add_argument('--device_id', type=str, default='2')
parser.add_argument("--gen_dir", type=str, default='checkpoint/FedDDPMmnist_ddpm_GeFL_w0.00.pt') 
parser.add_argument("--rs",type=int,default=0)

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

#########
bs=2
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
z = Variable(FloatTensor(np.random.normal(0, 1, (bs, args.latent_dim)))).view(-1, args.latent_dim, 1, 1) # generator
x = torch.zeros(bs, args.output_channel, args.img_size, args.img_size, device='cuda') # discriminator , VAE, DDPM
c = Variable(LongTensor(np.random.randint(0, args.num_classes, bs)))

loss_fn_alex = lpips.LPIPS(net='alex').to(args.device)

######### DCGAN #########
if args.model_type == 'dcgan':
    from generators32.DCGAN import *
    
    if args.dataset == 'mnist':
        fedgen = generator(args, d=128).to(args.device)
        add = discriminator(args, d=128).to(args.device)
    elif args.dataset == 'cifar10':
        fedgen = generator(args, d=256).to(args.device)
        add = discriminator(args, d=64).to(args.device)
        
    # fedgen.load_state_dict(torch.load('checkpoint/FedDCGANmixnet0.pt')) # cifar10
    # fedgen.load_state_dict(torch.load('checkpoint/FedDCGANmixnet1.pt'))
    fedgen.load_state_dict(torch.load('checkpoint/FedDCGANmixnet2.pt'))
    # fedgen.load_state_dict(torch.load('checkpoint/FedDCGANupdateGEN0.pt'))
    # fedgen.load_state_dict(torch.load('checkpoint/FedDCGANupdateGEN1.pt'))
    # fedgen.load_state_dict(torch.load('checkpoint/FedDCGANupdateGEN2.pt'))
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1) # 10 x 10 eye matrix
    y_ = (torch.rand(bs, 1) * 10).type(torch.LongTensor).squeeze()
    y_label_ = onehot[y_]
    y_label_ = Variable(y_label_.cuda())
    fill = torch.zeros([10, 10, args.img_size, args.img_size])
    for i in range(10):
        fill[i, i, :, :] = 1
    y_fill_ = fill[y_]
    y_fill_ = Variable(y_fill_.cuda())

    tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                            transforms.Resize(args.img_size)
                            ]) # mnist is already normalised 0 to 1

    # summary(fedgen, z, y_label_) # torchsummaryX
    # summary(add, x, y_fill_)

# ######### CVAE #########
elif args.model_type == 'cvae':
    from generators32.CCVAE import *
    fedgen = CCVAE(args).to(args.device) # [transforms.ToTensor(),]
    # fedgen.load_state_dict(torch.load('checkpoint/FedCVAEcifar10_lr_fix_0.pt')) # cifar10
    # fedgen.load_state_dict(torch.load('checkpoint/FedCVAEcifar10_lr_fix_1.pt'))
    fedgen.load_state_dict(torch.load('checkpoint/FedCVAEcifar10_lr_fix_2.pt'))
    # fedgen.load_state_dict(torch.load('checkpoint/FedCVAEfreezeGEN0.pt')) # mnist
    # fedgen.load_state_dict(torch.load('checkpoint/FedCVAEfreezeGEN1.pt'))
    # fedgen.load_state_dict(torch.load('checkpoint/FedCVAEfreezeGEN2.pt'))
    y = (torch.rand(bs, 1) * 10).type(torch.LongTensor).squeeze()
    label = np.zeros((bs, 10))
    label[np.arange(bs), y] = 1
    label = torch.tensor(label).to(args.device)
    tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(args.img_size)
                            ]) # mnist is already normalised 0 to 1

    if args.dataset == 'cifar10':
        tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                            transforms.Resize(args.img_size)
                            ]) # mnist is already normalised 0 to 1
            
    # # summary(fedgen, x, label) # torchsummaryX

######## DDPM #########
elif args.model_type == 'ddpm':
    from DDPM.ddpm32 import *
    fedgen = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
                    betas=(1e-4, 0.02), drop_prob=0.1).to(args.device) # [transforms.ToTensor(),]
    fedgen.load_state_dict(torch.load(args.gen_dir)) # evaluate over args.guide_w = 0, 2
    # fedgen.load_state_dict(torch.load('checkpoint/FedDDPM01001.pt')) # evaluate over args.guide_w = 0, 2
    # fedgen.load_state_dict(torch.load('checkpoint/FedDDPM01002.pt')) # evaluate over args.guide_w = 0, 2
    tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(args.img_size)
                            ]) # mnist is already normalised 0 to 1
    if args.dataset == 'cifar10':
        tf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                            transforms.Resize(args.img_size)
                            ]) # mnist is already normalised 0 to 1
    # summary(fedgen, x, c) # torchsummaryX


""" synthetic data subset """
fedgen.eval()
with torch.no_grad():
    img_batch, _ = fedgen.sample_image(args, sample_num=600) # outputs imgs of size (sample_num, 1*32*32)
img_batch = img_batch.view(-1, args.output_channel, args.img_size, args.img_size) # (sample_num, 1, 32, 32)
synset = img_batch

save_image(img_batch, 'imgs/imgFedGEN/SynOrig_Ex1' + '.png', nrow=10, normalize=True)

if args.dataset == 'mnist':
    trainset = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True)
    valset = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=False, transform=tf, download=True)
elif args.dataset == 'cifar10':
    trainset = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=True, download=True, transform=tf)
    valset = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=False, download=True, transform=tf)


def NN_distance(sample, dataset):
    with torch.no_grad():
        min_dist = np.inf
        if torch.is_tensor(dataset):
            sample_copied = sample.unsqueeze(0).repeat(len(dataset), 1, 1, 1) # .to(args.device)
            dataset = dataset.to(args.device)
            # distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(sample_copied, dataset), 2), (1, 2, 3)))
            distance = loss_fn_alex(sample_copied, dataset)
            min_dist = torch.min(distance)
        else:
            for data_idx in range(len(dataset)):
                neighbor, _ = dataset[data_idx]
                neighbor = neighbor.to(args.device)
                # distance = torch.norm(sample-neighbor)
                distance = loss_fn_alex(sample, neighbor)
                # distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(sample, neighbor), 2), (0, 1, 2)))
                if min_dist > distance:
                    min_dist = distance
    return min_dist
    
def NN_mem(sample, valset, synset):
    val_dist = NN_distance(sample, valset)
    syn_dist = NN_distance(sample, synset)
    rho = val_dist/syn_dist
    return rho 
    
    
""" Make subset of train/val dataset """
import ipdb
train_indices = torch.randperm(len(trainset))[:1000]
trainset = Subset(trainset, train_indices)

val_indices = torch.randperm(len(valset))[:600]
valset = Subset(valset, val_indices)

""" Nearest Neighbors """
rho_sum = 0
rho_list = []
print("Start computing memorization")
for train_idx in range(len(trainset)):
    tr_data, _ = trainset[train_idx]
    tr_data = tr_data.to(args.device)
    # memorization
    rho =  NN_mem(tr_data, valset, synset)
    if (train_idx+1) % 100 == 0:
        print('{:d} completed'.format(train_idx+1))
    rho = rho.detach().cpu()
    rho_list.append(rho)
    rho_sum += rho

""" results visualization """    
rho_list = np.array(rho_list)
if args.model_type == 'ddpm':
    model_type = args.model_type + str(args.guide_w)
else:
    model_type = args.model_type
np.savetxt("./memorization_results/"+model_type+'_'+args.dataset+args.rs+".csv", rho_list, delimiter=',')
prob_of_mem = len([rho for rho in rho_list if rho>=1])/len(trainset)
mean = np.average(rho_list)
std = np.std(rho_list)
print("mean: ", mean, " stdev: ", std, " prob of memorization: ", prob_of_mem)

plt.figure()
plt.hist(rho_list, 50, density=True, label=model_type, alpha=0.6)
plt.title('mean: '+str(mean)+' stdev: '+ str(std)+' prob of mem: '+str(prob_of_mem))
plt.xlabel('memorization score')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.savefig('./memorization_results/'+model_type+'_'+args.dataset+'_memorization.png')

