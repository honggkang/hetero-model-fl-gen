from args import parse_args
import random
import copy

import numpy as np
import torch
import wandb

from utils.user_sampling import user_select
from utils.setup import setup_experiment

from utils.localUpdateTarget import LocalUpdate, LocalUpdate_header, LocalUpdate_onlyGen
from utils.localUpdateGenF import LocalUpdate_CVAEF
from utils.avg import LGFedAvg, model_wise_FedAvg, FedAvg, LGFedAvg_frozen_FE
from utils.util import save_generated_images, evaluate_models
from generators16.CCVAE import *


def main():

    dataset_train, dataset_test, dict_users, local_models, common_net, w_comm, ws_glob, run = setup_experiment(args)
    print(args)
    
    loss_train = []
    best_perf = [0 for _ in range(args.num_models)]
    ''' -----------------------------------------
    Warming up Feature extractor of Target models
    ----------------------------------------- '''
    for iter in range(1, args.wu_epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        loss_locals = []
        
        idxs_users = user_select(args)
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model = local_models[dev_spec_idx]
            model.load_state_dict(ws_glob[dev_spec_idx])
                        
            local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
            weight, loss, _ = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=args.lr)

            ws_local[dev_spec_idx].append(copy.deepcopy(weight))
            loss_locals.append(loss)
        
        if args.avg_FE: # LG-FedAvg
            ws_glob, w_comm = LGFedAvg(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else: # FedAvG
            ws_glob = model_wise_FedAvg(args, ws_glob, ws_local)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Warm-up Target Net Round {:3d}, Avg loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        if iter == 1 or iter % args.sample_test == 0 or iter == args.wu_epochs:
            best_perf = evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf)

    common_net.load_state_dict(w_comm)
    common_net.eval()

    gen_glob = CCVAE(args).to(args.device)
    opt = torch.optim.Adam(gen_glob.parameters(), lr=1e-3, weight_decay=0.001).state_dict()
    opts = [copy.deepcopy(opt) for _ in range(args.num_users)]

    ''' -------------------------------
    Federated Training generative model
    ------------------------------- '''
    for iter in range(1, args.gen_wu_epochs+1):
        gen_w_local, loss_locals = [], []
        
        idxs_users = user_select(args)
        for idx in idxs_users:
            local = LocalUpdate_CVAEF(args, common_net, dataset=dataset_train, idxs=dict_users[idx])
            gen_weight, loss, opts[idx] = local.train(net=copy.deepcopy(gen_glob), opt=opts[idx])
            gen_w_local.append(copy.deepcopy(gen_weight))
            loss_locals.append(loss)
        
        gen_w_glob = FedAvg(gen_w_local)
        gen_glob.load_state_dict(gen_w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)

        if args.save_imgs and (iter % args.sample_test == 0 or iter == args.gen_wu_epochs):
            save_generated_images(args.save_dir, gen_glob, args, iter)
        print('Warm-up Gen Round {:3d}, CVAE Average loss {:.3f}'.format(iter, loss_avg))

    ''' ----------------------------------------
    Train main networks by local sample
    and generated samples, then update generator
    ---------------------------------------- '''
    for iter in range(1,args.epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        gen_w_local, loss_locals, gen_loss_locals, gloss_locals = [], [], [], []

        idxs_users = user_select(args)
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model = local_models[dev_spec_idx]
            model.load_state_dict(ws_glob[dev_spec_idx])

            if args.only_gen: # necessarily aid_by_gen=True
                local = LocalUpdate_onlyGen(args, dataset=dataset_train, idxs=dict_users[idx])
                weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), feature_start=True, gennet=copy.deepcopy(gen_glob), learning_rate=args.lr)
            else:
                local = LocalUpdate_header(args, dataset=dataset_train, idxs=dict_users[idx])
                if args.aid_by_gen:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), feature_extractor=common_net, gennet=copy.deepcopy(gen_glob), learning_rate=args.lr)
                else:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), feature_extractor=common_net, learning_rate=args.lr) # weights of models

            ws_local[dev_spec_idx].append(weight)
            loss_locals.append(loss)
            gen_loss_locals.append(gen_loss)

        ws_glob, w_comm = LGFedAvg_frozen_FE(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        print('Round {:3d}, Average loss {:.3f}, Average loss by Gen {:.3f}'.format(iter, loss_avg, gen_loss_avg))

        loss_train.append(loss_avg)
        if iter==1 or iter % args.sample_test == 0 or iter == args.epochs:
            best_perf = evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf)

    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))

    if args.wandb:
        run.finish()

    return sum(best_perf)/len(best_perf)


if __name__ == "__main__":
    args = parse_args()
    args.gen_model = 'vaef'
    
    results = []
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs)
        np.random.seed(args.rs)
        random.seed(args.rs)
        results.append(main())
        args.rs = args.rs+1
        print(results)