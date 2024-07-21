import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    ### clients
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--frac', type=float, default=1)
    parser.add_argument('--partial_data', type=float, default=0.5)
    
    ### model & feature size
    parser.add_argument('--models', type=str, default='cnn3-layer2') # cnn (MNIST), cnnbn (FMNIST), mlp
    parser.add_argument('--output_channel', type=int, default=3) # local epochs for training generator
    parser.add_argument('--img_size', type=int, default=32) # local epochs for training generator
    parser.add_argument('--orig_img_size', type=int, default=32) # local epochs for training generator
    parser.add_argument('--feat_channel', type=int, default=10)

    ### dataset
    parser.add_argument('--dataset', type=str, default='cifar10') # stl10, cifar10, svhn, fmnist, mnist, emnist
    parser.add_argument('--noniid', action='store_true') # default: false
    parser.add_argument('--dir_param', type=float, default=0.3)
    parser.add_argument('--num_classes', type=int, default=10)

    ### optimizer
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--local_bs', type=int, default=128)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    ### reproducibility
    parser.add_argument('--rs', type=int, default=0)
    parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
    parser.add_argument('--device_id', type=str, default='3')

    ### GeFL-F
    parser.add_argument('--wu_epochs', type=int, default=50) # 20 warm-up epochs for FE
    parser.add_argument('--freeze_FE', type=bool, default=True)

    ### GeFL / GeFL-F
    parser.add_argument('--gen_wu_epochs', type=int, default=200) # warm-up epochs for generator

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--local_ep_gen', type=int, default=1) # local epochs for training main nets by generated samples
    parser.add_argument('--gen_local_ep', type=int, default=5) # local epochs for training generator

    parser.add_argument('--aid_by_gen', type=bool, default=False) # False True
    parser.add_argument('--freeze_gen', type=bool, default=False) # GAN: False
    parser.add_argument('--avg_FE', type=bool, default=False) # True: LG-FedAvg
    parser.add_argument('--only_gen', type=bool, default=False)
    parser.add_argument('--train_gen_only',type=bool, default=False)
    
    ### logging
    parser.add_argument('--sample_test', type=int, default=10) # local epochs for training generator
    parser.add_argument('--save_imgs', type=bool, default=False) # local epochs for training generator
    parser.add_argument('--wandb', type=bool, default=False) # True False
    parser.add_argument('--wandb_proj_name', type=str, default='CIFAR10-CNNF-new')
    parser.add_argument('--name', type=str, default='L2-vaef') # L-A: bad character
    parser.add_argument('--gen_model', type=str, default='vaef') # vae, gan, ddpm
    
    ### VAE parameters
    parser.add_argument('--latent_size', type=int, default=50) # local epochs for training generator
    # 16 for mnist/fmnist, 50 for cifar10
    ### GAN parameters
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--gan_lr', type=float, default=0.0002) # GAN lr
    parser.add_argument('--latent_dim', type=int, default=100)
    ### DDPM parameters
    parser.add_argument('--n_feat', type=int, default=128) # 128 ok, 256 better (but slower)
    parser.add_argument('--n_T', type=int, default=400) # 400, 500
    parser.add_argument('--guide_w', type=float, default=0.0) # 0, 0.5, 2

    ### Target nets
    parser.add_argument('--lr', type=float, default=1e-1)

    ### FedProx parameters
    parser.add_argument('--fedprox', type=bool, default=False) # local epochs for training generator
    parser.add_argument('--mu', type=float, default=1e-2)

    ### AvgKD ####
    parser.add_argument('--avgKD', type=bool, default=False)

    ### resume after warming up ###
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--saved_ckpt', type=str, default='checkpoint/FedDDPM_cifar10_cnn_ddpmw0.00.pt')
    parser.add_argument('--saved_ckpt_g', type=str, default='checkpoint/FedDDPM_cifar10_cnn_ddpmw0.00.pt') # for gan gen
    parser.add_argument('--saved_ckpt_d', type=str, default='checkpoint/FedDDPM_cifar10_cnn_ddpmw0.00.pt') # for gan dis
    args = parser.parse_args()
    args.device = 'cuda:' + args.device_id
    
    if args.dataset == 'fmnist' or 'mnist':
        args.epochs = 50
        if args.freeze_gen:
            args.gen_wu_epochs = 100   
        else:
            args.gen_wu_epochs = 50 
    if args.dataset == 'cifar10':
        args.epochs = 100
        if args.freeze_gen:
            args.gen_wu_epochs = 200
        else:
            args.gen_wu_epochs = 100
    return args