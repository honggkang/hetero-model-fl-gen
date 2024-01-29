from args import parse_args
import random
import copy

import numpy as np
import torch
import wandb

from utils.user_sampling import user_select
from utils.setup import setup_experiment
from utils.localUpdateTarget import LocalUpdate, LocalUpdate_onlyGen
from utils.localUpdateGen import LocalUpdate_DCGAN
from utils.avg import LGFedAvg, model_wise_FedAvg, FedAvg
from utils.getGenTrainData import generator_traindata
from generators32.DCGAN import *
from utils.util import save_generated_images, evaluate_models


def main():
    
    dataset_train, dataset_test, dict_users, local_models, common_net, w_comm, ws_glob, run = setup_experiment(args)
    print(args)            

    loss_train = []
    gen_glob = generator(args, d=128).to(args.device)
    dis_glob = discriminator(args, d=128).to(args.device)
    gen_glob.weight_init(mean=0.0, std=0.02)
    dis_glob.weight_init(mean=0.0, std=0.02)
    optg = torch.optim.Adam(gen_glob.parameters(), lr=args.gan_lr, betas=(args.b1, args.b2)).state_dict()
    optd = torch.optim.Adam(dis_glob.parameters(), lr=args.gan_lr, betas=(args.b1, args.b2)).state_dict()    
    optgs = [copy.deepcopy(optg) for _ in range(args.num_users)]
    optds = [copy.deepcopy(optd) for _ in range(args.num_users)]

    ''' ---------------------------
    Federated Training generative model
    --------------------------- '''    
    for iter in range(1, args.gen_wu_epochs+1):
        gen_w_local, dis_w_local, gloss_locals, dloss_locals = [], [], [], []
        
        idxs_users = user_select(args)
        for idx in idxs_users:
            local = LocalUpdate_DCGAN(args, dataset=train_data, idxs=dict_users[idx])
            g_weight, d_weight, gloss, dloss, optgs[idx], optds[idx] = local.train(gnet=copy.deepcopy(gen_glob), dnet=copy.deepcopy(dis_glob), iter=iter, optg=optgs[idx], optd=optds[idx])
            gen_w_local.append(copy.deepcopy(g_weight))
            dis_w_local.append(copy.deepcopy(d_weight))           
            gloss_locals.append(gloss)
            dloss_locals.append(dloss)
        
        gen_w_glob = FedAvg(gen_w_local)
        dis_w_glob = FedAvg(dis_w_local)        
        gen_glob.load_state_dict(gen_w_glob)
        dis_glob.load_state_dict(dis_w_glob)
        gloss_avg = sum(gloss_locals) / len(gloss_locals)
        dloss_avg = sum(dloss_locals) / len(dloss_locals)
        if args.save_imgs and (iter % args.sample_test == 0 or iter == args.gen_wu_epochs):
            save_generated_images(args.save_dir, gen_glob, args, iter)
        print('Warm-up Gen Round {:3d}, G Avg loss {:.3f}, D Avg loss {:.3f}'.format(iter, gloss_avg, dloss_avg))

    # torch.save(gen_w_glob, 'models/save/Fed' + '_' + str(args.models) + '_DCGAN_G_sameSize.pt')
    
    best_perf = [0 for _ in range(args.num_models)]

    ''' ----------------------------------------
    Train main networks by local sample and generated samples, then update generator
    ---------------------------------------- '''
    for iter in range(1, args.epochs+1):
        ws_local = [[] for _ in range(args.num_models)]
        gen_w_local, dis_w_local = [], []
        loss_locals, gen_loss_locals = [], []
        gloss_locals, dloss_locals = [], []
        
        idxs_users = user_select(args)        
        for idx in idxs_users:
            dev_spec_idx = min(idx//(args.num_users//args.num_models), args.num_models-1)
            model = local_models[dev_spec_idx]
            model.load_state_dict(ws_glob[dev_spec_idx])
            
            if args.only_gen:
                local = LocalUpdate_onlyGen(args, dataset=dataset_train, idxs=dict_users[idx])
                weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=args.lr)
            else:
                local = LocalUpdate(args, dataset=dataset_train, idxs=dict_users[idx])
                if args.aid_by_gen: # synthetic data updates header & real data updates whole target network
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), gennet=copy.deepcopy(gen_glob), learning_rate=args.lr)
                else:
                    weight, loss, gen_loss = local.train(net=copy.deepcopy(model).to(args.device), learning_rate=args.lr)

            ws_local[dev_spec_idx].append(weight)
            loss_locals.append(loss)
            gen_loss_locals.append(gen_loss)

            if args.aid_by_gen and not args.freeze_gen:
                local_gen = LocalUpdate_DCGAN(args, dataset=train_data, idxs=dict_users[idx])
                g_weight, d_weight, gloss, dloss, optgs[idx], optds[idx] = local_gen.train(gnet=copy.deepcopy(gen_glob), dnet=copy.deepcopy(dis_glob), iter=args.gen_wu_epochs+iter, optg=optgs[idx], optd=optds[idx])
                gen_w_local.append(copy.deepcopy(g_weight))
                dis_w_local.append(copy.deepcopy(d_weight))
                gloss_locals.append(gloss)
                dloss_locals.append(dloss)
                
        if  args.aid_by_gen and not args.freeze_gen:
            gloss_avg = sum(gloss_locals) / len(gloss_locals)
            dloss_avg = sum(dloss_locals) / len(dloss_locals)
            gen_w_glob = FedAvg(gen_w_local)
            dis_w_glob = FedAvg(dis_w_local)
            gen_glob.load_state_dict(gen_w_glob)
            dis_glob.load_state_dict(dis_w_glob)

            if args.save_imgs and (iter % args.sample_test == 0 or iter == args.epochs):
                save_generated_images(args.save_dir, gen_glob, args, args.gen_wu_epochs+iter)
            print('Gen Round {:3d}, G Avg loss {:.3f}, D Avg loss {:.3f}'.format(args.gen_wu_epochs+iter, gloss_avg, dloss_avg))
        else:
            gloss_avg = -1
            dloss_avg = -1

        if args.avg_FE: # LG-FedAVG
            ws_glob, w_comm = LGFedAvg(args, ws_glob, ws_local, w_comm) # main net, feature extractor weight update
        else: # FedAVG
            ws_glob = model_wise_FedAvg(args, ws_glob, ws_local)
        loss_avg = sum(loss_locals) / len(loss_locals)
        try:
            gen_loss_avg = sum(gen_loss_locals) / len(gen_loss_locals)
        except:
            gen_loss_avg = -1
        print('Round {:3d}, Avg loss {:.3f}, Avg loss by Gen samples {:.3f}, G Avg loss {:.3f}, D Avg loss {:.3f}'.format(iter, loss_avg, gen_loss_avg, gloss_avg, dloss_avg))

        loss_train.append(loss_avg)
        if iter == 1 or iter % args.sample_test == 0 or iter == args.epochs:
            best_perf = evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf)
                                    
    print(best_perf, 'AVG'+str(args.rs), sum(best_perf)/len(best_perf))
    torch.save(gen_w_glob, 'checkpoint/FedDCGAN' + str(args.name) + str(args.rs) + '.pt')

    if args.wandb:
        run.finish()

    return sum(best_perf)/len(best_perf)

if __name__ == "__main__":
    args = parse_args()
    args.gen_model = 'gan'
    train_data = generator_traindata(args)
        
    results = []
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        np.random.seed(args.rs)
        random.seed(args.rs)
        results.append(main())
        args.rs = args.rs+1
        print(results)