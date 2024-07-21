'''
skeleton code for evaluating IS-score
'''
import os
import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from pytorch_gan_metrics import get_inception_score, get_fid

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')  # stl10, cifar10, svhn, mnist, fmnist
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument("--output_channel", type=int, default=1, help="number of image channels")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--bs", type=int, default=1000, help="size of the batches")

### GAN
parser.add_argument('--latent_dim', type=int, default=100)
### VAE parameters
parser.add_argument("--latent_size", type=int, default=50, help="dimensionality of the latent space") # VAE
### DDPM parameters
parser.add_argument('--n_feat', type=int, default=128)
parser.add_argument('--n_T', type=int, default=100)
parser.add_argument('--guide_w', type=float, default=0)  # 0, 0.5, 2
### CUDA
parser.add_argument('--device_id', type=str, default='0')
### Parameters for FID
parser.add_argument('--gen', type=str, default='ddpm')  # gan(GAN), vae(VAE), ddpm(DDPM)
parser.add_argument('--gen_dir', type=str, default='checkpoint/ddpm_warmup_0.0__epoch200_rs2.pt')
parser.add_argument('--img_dir', type=str, default='imgs/')

args = parser.parse_args()
args.device = 'cuda:' + args.device_id
args.img_shape = (args.output_channel, args.img_size, args.img_size)

# cifar10 dcgan checkpoint : checkpoint/FedDCGANcifar10_gan_frz_fix_0.pt
# cifar10 cvae checkpoint: checkpoint/FedCVAEcifar10_cvae_frz_fix_50_0.pt
# cifar10 ddpm w=0 checkpoint: checkpoint/path_for_fid/FedDDPM_warm_0_cifar10_ddpm_frz_fix_0.pt
# cifar10 ddpm w=2 checkpoint: checkpoint/path_for_fid/FedDDPM_warm_2.0_cifar10_ddpm_frz_fix_0.pt

if args.dataset == 'mnist':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.img_size),
        # transforms.Normalize([0.5], [0.5])
    ])  # mnist is already normalised 0 to 1

    train_data = datasets.MNIST(root='/home/chaseo/GeFL/.data/mnist', train=True, transform=tf, download=True)
    test_data = datasets.MNIST(root='/home/chaseo/GeFL/.data/mnist', train=False, transform=tf, download=True)
elif args.dataset == 'cifar10':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_data = datasets.CIFAR10('/home/chaseo/GeFL/.data/cifar', train=True, download=True, transform=tf)
    test_data = datasets.CIFAR10('/home/chaseo/GeFL/.data/cifar', train=False, download=True, transform=tf)

# Check if the provided generator value is in the list of valid generators
valid_generators = ['gan', 'GAN', 'vae', 'VAE', 'ddpm', 'DDPM', 'dcgan', 'DCGAN', 'cvae', 'CVAE']
if args.gen.lower() not in valid_generators:
    print("Warning: The provided --generator value is not one of 'gan', 'vae', or 'ddpm'.")
    print("Please provide a valid generator value.")
if not os.path.exists(args.gen_dir):
    print("Please provide a valid generator model path.")


######## GAN #########
if args.gen == "gan" or args.gen == "GAN":
    from mlp_generators.GAN import *

    fedgen = Generator(args).to(args.device)  # [transforms.ToTensor(),transforms.Normalize([0.5], [0.5])]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    add = Discriminator(args).to(args.device)
    fedgen.eval()
    
    # Initialize lists to store Inception Scores
    num_images = args.bs
    all_IS = []
    all_IS_std = []

    # Loop through different condition_idx values
    for condition_idx in range(10):
        generated_images, _ = fedgen.sample_cond_image(args, sample_num=num_images, condition_index=condition_idx)
        IS, IS_std = get_inception_score(generated_images)

        # Print and store Inception Scores for each condition
        print(f"Class {condition_idx}: IS = {IS}, IS_std = {IS_std}")
        all_IS.append(IS)
        all_IS_std.append(IS_std)

    # Calculate and print the average Inception Scores
    avg_IS = np.mean(all_IS)
    avg_IS_std = np.mean(all_IS_std)
    print(f"\nAverage Inception Scores: IS = {avg_IS}, IS_std = {avg_IS_std}")         


######## DCGAN ########
elif args.gen == "dcgan" or args.gen == "DCGAN":
    from generators32.DCGAN import *

    fedgen = generator(args, d=128).to(args.device)
    fedgen.load_state_dict(torch.load(args.gen_dir))
    add = discriminator(args, d=128).to(args.device)
    fedgen.eval()

    # Initialize lists to store Inception Scores
    num_images = args.bs
    all_IS = []
    all_IS_std = []
    
    with torch.no_grad():
        # Loop through different condition_idx values
        for condition_idx in range(10):
            generated_images, _ = fedgen.sample_cond_image(args, sample_num=num_images, condition_index=condition_idx)
            if args.dataset == 'mnist' or args.dataset == 'fmnist':
                generated_images = generated_images.expand(generated_images.shape[0], 3, generated_images.shape[2], generated_images.shape[3])
            IS, IS_std = get_inception_score(generated_images)
        
            # Print and store Inception Scores for each condition
            print(f"Class {condition_idx}: IS = {IS}, IS_std = {IS_std}")
            all_IS.append(IS)
            all_IS_std.append(IS_std)
        
        # Calculate and print the average Inception Scores
        avg_IS = np.mean(all_IS)
        avg_IS_std = np.mean(all_IS_std)
        print(f"\nAverage Inception Scores: IS = {avg_IS}, IS_std = {avg_IS_std}")                      
        


######### VAE #########
elif args.gen == "vae" or args.gen == "VAE":
    from mlp_generators.VAE import *

    fedgen = CVAE(args).to(args.device)  # [transforms.ToTensor(),]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    # summary(fedgen, x, c) # torchsummaryX
    fedgen.eval()

    # Initialize lists to store Inception Scores
    num_images = args.bs
    all_IS = []
    all_IS_std = []

    # Loop through different condition_idx values
    for condition_idx in range(10):
        generated_images, _ = fedgen.sample_cond_image(args, sample_num=num_images, condition_index=condition_idx)
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            generated_images = generated_images.expand(generated_images.shape[0], 3, generated_images.shape[2], generated_images.shape[3])
        IS, IS_std = get_inception_score(generated_images)

        # Print and store Inception Scores for each condition
        print(f"Class {condition_idx}: IS = {IS}, IS_std = {IS_std}")
        all_IS.append(IS)
        all_IS_std.append(IS_std)

    # Calculate and print the average Inception Scores
    avg_IS = np.mean(all_IS)
    avg_IS_std = np.mean(all_IS_std)
    print(f"\nAverage Inception Scores: IS = {avg_IS}, IS_std = {avg_IS_std}")      


######### CVAE #########
elif args.gen == "cvae" or args.gen == "CVAE":
    from generators32.CCVAE import *
    args.latent_size = 16 # CVAE

    fedgen = CCVAE(args).to(args.device)  # [transforms.ToTensor(),]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    # summary(fedgen, x, c) # torchsummaryX
    fedgen.eval()

    # Initialize lists to store Inception Scores
    all_IS = []
    all_IS_std = []
    num_images = args.bs
    
    # Loop through different condition_idx values
    for condition_idx in range(10):
        generated_images, _ = fedgen.sample_cond_image(args, sample_num=num_images, condition_index=condition_idx)
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            generated_images = generated_images.expand(generated_images.shape[0], 3, generated_images.shape[2], generated_images.shape[3])
        IS, IS_std = get_inception_score(generated_images)

        # Print and store Inception Scores for each condition
        print(f"Class {condition_idx}: IS = {IS}, IS_std = {IS_std}")
        all_IS.append(IS)
        all_IS_std.append(IS_std)

    # Calculate and print the average Inception Scores
    avg_IS = np.mean(all_IS)
    avg_IS_std = np.mean(all_IS_std)
    print(f"\nAverage Inception Scores: IS = {avg_IS}, IS_std = {avg_IS_std}")      

######### DDPM #########
elif args.gen == "ddpm" or args.gen == "DDPM":
    # from DDPM.ddpm28 import *
    from DDPM.ddpm32 import *

    fedgen = DDPM(args,
                  nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
                  betas=(1e-4, 0.02), drop_prob=0.1).to(args.device)  # [transforms.ToTensor(),]
    fedgen.load_state_dict(torch.load(args.gen_dir))
    fedgen.eval()

    # Initialize lists to store Inception Scores
    num_images = args.bs
    all_IS = []
    all_IS_std = []

    with torch.no_grad():
        # Loop through different condition_idx values
        for condition_idx in range(10):
            generated_images, _ = fedgen.sample_cond_image(args, sample_num=num_images, condition_index=condition_idx)
            if args.dataset == 'mnist' or args.dataset == 'fmnist':
                generated_images = generated_images.expand(generated_images.shape[0], 3, generated_images.shape[2], generated_images.shape[3])
            IS, IS_std = get_inception_score(generated_images)

            # Print and store Inception Scores for each condition
            print(f"Class {condition_idx}: IS = {IS}, IS_std = {IS_std}")
            all_IS.append(IS)
            all_IS_std.append(IS_std)

        # Calculate and print the average Inception Scores
        avg_IS = np.mean(all_IS)
        avg_IS_std = np.mean(all_IS_std)
        print(f"\nAverage Inception Scores: IS = {avg_IS}, IS_std = {avg_IS_std}")      