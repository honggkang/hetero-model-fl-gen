# Heterogeneous-Model Federated Learning
Federated Learning (FL) has emerged as a promising paradigm in distributed learning, aiming to train a single global model while preserving the privacy of individual users. However, the increasing size of recent models introduces challenges of model heterogeneity, encompassing diverse computing capabilities and network bandwidth conditions across clients.
In this paper, we propose Generative Model-Aided Federated Learning GeFL, which incorporates a conditional generative model trained in a federated manner to aggregate global knowledge under model heterogeneity. Through a series of experiments on various image classification tasks, we demonstrate the discernible performance improvements of GeFL compared to baselines, as well as its limitations in terms of privacy and scalability. To tackle concerns addressed in GeFL, we introduce a novel framework, GeFL-F, feature-generative model-aided FL, by decoupling each target network into a common feature extractor and heterogeneous header. We empirically demonstrate the consistent performance gains of GeFL-F, while proving better privacy preservation and robustness to a large number of clients.


## LG-FedAvg
python GeFL_CVAE.py --aid_by_gen 0

## GeFL (CVAE)
python GeFL_CVAE.py --aid_by_gen 1 --num_users 10 --device_id 0 --gen_model vae

## GeFL-F (CVAE-F)
python GeFL_CVAE-F.py --aid_by_gen 1 --num_users 100 --device_id 0 --name 100cvaef --gen_model vaef

## GeFL-F (DDPM-F)
python GeFL_DDPM-F.py --aid_by_gen 1 --guide_w 0 --num_users 100 --device_id 0 --name 100ddpmf  --gen_model ddpmf