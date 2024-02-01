import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    ### clients
    parser.add_argument('--num_users', type=int, default=10)
    parser.add_argument('--frac', type=float, default=1)
    parser.add_argument('--partial_data', type=float, default=0.1)
    
    ### model & feature size
    parser.add_argument('--models', type=str, default='cnnbn') # cnn (MNIST), cnnbn (FMNIST), mlp
    parser.add_argument('--output_channel', type=int, default=1) # local epochs for training generator
    parser.add_argument('--img_size', type=int, default=32) # local epochs for training generator
    parser.add_argument('--orig_img_size', type=int, default=32) # local epochs for training generator

    ### dataset
    parser.add_argument('--dataset', type=str, default='fmnist') # stl10, cifar10, svhn, fmnist, mnist, emnist
    parser.add_argument('--noniid', action='store_true') # default: false
    parser.add_argument('--dir_param', type=float, default=0.3)
    parser.add_argument('--num_classes', type=int, default=10)

    ### optimizer
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--local_bs', type=int, default=64)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    ### reproducibility
    parser.add_argument('--rs', type=int, default=0)
    parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
    parser.add_argument('--device_id', type=str, default='0')

    ### GeFL-F
    parser.add_argument('--wu_epochs', type=int, default=20) # 20 warm-up epochs for FE
    parser.add_argument('--freeze_FE', type=bool, default=True)

    ### GeFL / GeFL-F
    parser.add_argument('--gen_wu_epochs', type=int, default=100) # warm-up epochs for generator

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--local_ep_gen', type=int, default=1) # local epochs for training main nets by generated samples
    parser.add_argument('--gen_local_ep', type=int, default=5) # local epochs for training generator

    parser.add_argument('--aid_by_gen', type=int, default=1) # False True
    parser.add_argument('--freeze_gen', type=int, default=1) # GAN: False
    parser.add_argument('--avg_FE', type=int, default=1) # True: LG-FedAvg
    parser.add_argument('--only_gen', type=int, default=0)

    ### logging
    parser.add_argument('--sample_test', type=int, default=10) # local epochs for training generator
    parser.add_argument('--save_imgs', type=bool, default=False) # local epochs for training generator
    parser.add_argument('--wandb', type=int, default=0) # True False
    parser.add_argument('--wandb_proj_name', type=str, default='gefl')
    parser.add_argument('--name', type=str, default='gefl') # L-A: bad character

    parser.add_argument('--gen_model', type=str, default='vae') # vae, gan, ddpm
    ### VAE parameters
    parser.add_argument('--latent_size', type=int, default=16) # local epochs for training generator
    ### GAN parameters
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--gan_lr', type=float, default=0.0002) # GAN lr
    parser.add_argument('--latent_dim', type=int, default=100)
    ### DDPM parameters
    parser.add_argument('--n_feat', type=int, default=128) # 128 ok, 256 better (but slower)
    parser.add_argument('--n_T', type=int, default=200) # 400, 500
    parser.add_argument('--guide_w', type=float, default=0.0) # 0, 0.5, 2

    ### Target nets
    parser.add_argument('--lr', type=float, default=1e-1)

    ### FedProx parameters
    parser.add_argument('--fedprox', type=bool, default=False) # local epochs for training generator
    parser.add_argument('--mu', type=float, default=1e-2)

    ### AvgKD ####
    parser.add_argument('--avgKD', type=bool, default=False)

    args = parser.parse_args()
    args.device = 'cuda:' + args.device_id
    
    if args.dataset == 'fmnist' or 'mnist':
        args.gen_wu_epochs = 100
        args.epochs = 50
        if not args.freeze_gen:
            args.gen_wu_epochs = args.gen_wu_epochs - args.epochs

    if 'ddpm' in args.gen_model:
        args.name = args.name + 'w' + str(args.guide_w)
                
    return args