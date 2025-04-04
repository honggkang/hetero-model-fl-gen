from args import parse_args
import random
import copy

import numpy as np
import torch
import wandb

from utils.user_sampling import user_select
from utils.setup import setup_experiment
from utils.localUpdateTarget import LocalUpdate, LocalUpdate_onlyGen
from utils.localUpdateGen import LocalUpdate_DDPM
from utils.avg import LGFedAvg, model_wise_FedAvg, FedAvg
from utils.getGenTrainData import generator_traindata
from DDPM.ddpm32 import * # DDPM.ddpm28
from utils.util import save_generated_images, evaluate_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    dataset_train, dataset_test, dict_users, local_models, common_net, w_comm, ws_glob, run = setup_experiment(args)    
    print(args)

    loss_train = []
    lr = 1e-1 # MLP
    gen_glob = DDPM(args, nn_model=ContextUnet(in_channels=args.output_channel, n_feat=args.n_feat, n_classes=args.num_classes),
                    betas=(1e-4, 0.02), drop_prob=0.1).to(args.device)
    print(f"Number of parameters in gen_glob: {count_parameters(gen_glob)}")
    opt = torch.optim.Adam(gen_glob.parameters(), lr=1e-4).state_dict()
    opts = [copy.deepcopy(opt) for _ in range(args.num_users)]    

    ''' -------------------------------
    Federated Training generative model
    ------------------------------- '''
    for iter in range(1, args.gen_wu_epochs+1):
        gen_w_local, gloss_locals = [], []
        
        idxs_users = user_select(args)
        for idx in idxs_users:
            local = LocalUpdate_DDPM(args, dataset=train_data, idxs=dict_users[idx])
            lr_rate = 1-iter/args.gen_wu_epochs
            g_weight, gloss, opts[idx] = local.train(net=copy.deepcopy(gen_glob), lr_decay_rate=lr_rate, opt=opts[idx])
            gen_w_local.append(copy.deepcopy(g_weight))
            gloss_locals.append(gloss)
        
        gen_w_glob = FedAvg(gen_w_local)
        gen_glob.load_state_dict(gen_w_glob)
        gloss_avg = sum(gloss_locals) / len(gloss_locals)
        if args.save_imgs and (iter % args.sample_test == 0 or iter == args.gen_wu_epochs):
            save_generated_images(args.save_dir, gen_glob, args, iter)
        print('Warm-up GEN Round {:3d}, G Avg loss {:.3f}'.format(iter, gloss_avg))

    best_perf = [0 for _ in range(args.num_models)]
    ''' -----------------------------------------------------
    Train main networks by local sample and generated samples
    ----------------------------------------------------- '''
    for iter in range(1,args.epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        gen_w_local, loss_locals, gen_loss_locals, gloss_locals = [], [], [], []
        
        idxs_users = user_select(args)        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model = local_models[dev_spec_idx]
            model.load_state_dict(ws_glob[dev_spec_idx])

            if args.only_gen:
                local = LocalUpdate_onlyGen(args, dataset=dataset_train, idxs=dict_users[idx])
                weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=lr)
            else:
                local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
                if args.aid_by_gen:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=lr)
                else:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=lr)
                    
            ws_local[dev_spec_idx].append(weight)
            loss_locals.append(loss)
            gen_loss_locals.append(gen_loss)        

        if args.avg_FE:
            ws_glob, w_comm = LGFedAvg(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else:
            ws_glob = model_wise_FedAvg(args, ws_glob, ws_local)
        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        
        print('Round {:3d}, Average loss {:.3f}, Average loss by Gen {:.3f}'.format(iter, loss_avg, gen_loss_avg))

        loss_train.append(loss_avg)
        if iter == 1 or iter % args.sample_test == 0 or iter == args.epochs:
            best_perf = evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf)            

    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))
    torch.save(gen_w_glob, 'checkpoint/FedDDPM' + str(args.name) + str(args.rs) + '.pt')

    if args.wandb:
        run.finish()

    return sum(best_perf)/len(best_perf)

if __name__ == "__main__":
    args = parse_args()
    args.gen_model = 'ddpm'
    train_data = generator_traindata(args)
    
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