# utils/setup_utils.py
import os
from datetime import datetime
import wandb
from utils.getData import getDataset, dict_iid
from utils.getModels import getModel


def setup_experiment(args):

    if args.gen_model == 'ddpm':
        args.save_dir = 'imgs/FedDDPM/'
    elif args.gen_model == 'vae':
        args.save_dir = 'imgs/FedCVAE/'
    elif args.gen_model == 'gan':
        args.save_dir = 'imgs/FedDCGAN/'
    elif args.gen_model == 'vaef':
        args.save_dir = 'imgs/FedCVAEF/'
    elif args.gen_model == 'ganf':
        args.save_dir = 'imgs/FedDCGANF/'
    elif args.gen_model == 'ddpmf':
        args.save_dir = 'imgs/FedDDPMF/'
        
    if 'f' in args.gen_model:
        args.output_channel = 3
        args.img_size = 16
        args.n_T = 500
    args.img_shape = (args.output_channel, args.img_size, args.img_size)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset_train, dataset_test = getDataset(args)
    dict_users = dict_iid(dataset_train, int(1/args.partial_data*args.num_users)) # , args.rs
    local_models, common_net = getModel(args)
    try:
        w_comm = common_net.state_dict()
    except:
        w_comm = 'null'
    ws_glob = [local_models[_].state_dict() for _ in range(args.num_models)]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'./output/gefl/{timestamp}{args.name}{args.rs}'
    if not os.path.exists(filename):
        os.makedirs(filename)
        
    if args.wandb:
        run = wandb.init(dir=filename, project=args.wandb_proj_name, name=f'{args.name}{args.rs}', reinit=True, settings=wandb.Settings(code_dir="."))
        wandb.config.update(args)
    else:
        run = None

    if not args.aid_by_gen:
        args.gen_wu_epochs = 0
        args.local_ep_gen = 0
        args.gen_local_ep = 0

    return dataset_train, dataset_test, dict_users, local_models, common_net, w_comm, ws_glob, run