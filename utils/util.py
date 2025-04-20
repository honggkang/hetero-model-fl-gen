
# from math import ceil as up
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import logging
import copy
from torchvision.utils import save_image
import wandb


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    test_loss = 0
    correct = 0
    total_samples = 0


    data_loader = DataLoader(datatest, batch_size=args.bs)
    with torch.no_grad():
      for idx, (data, target) in enumerate(data_loader):
          if 'cuda' in args.device:
              data, target = data.to(args.device), target.to(args.device)
          logits, log_probs = net_g(data)

          if args.dataset == 'celebA':  # Multi-label classification
              test_loss += F.binary_cross_entropy_with_logits(logits, target, reduction='sum').item()
        # Convert logits to binary predictions (threshold = 0.5)
              y_pred = (torch.sigmoid(logits) > 0.5).float()
              correct += (y_pred == target).sum().item()
              total_samples += target.numel()  # Total number of labels
          else:
              test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
              y_pred = log_probs.data.max(1, keepdim=True)[1]
              correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
              total_samples += len(target)
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    net_g.to('cpu')
    return accuracy, test_loss


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        logger.handlers.clear()
        
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.terminator = ""
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.terminator = ""
        logger.addHandler(console_handler)
    logger.info(filepath)
    
    # with open(filepath, "r") as f:
        # logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def save_generated_images(dir, gen_model, args, iter):
    gen_model.eval()
    sample_num = 40
    samples = gen_model.sample_image_4visualization(sample_num)
    save_image(samples.view(sample_num, args.output_channel, args.img_size, args.img_size),
                dir + str(args.name)+ str(args.rs) +'_' + str(iter) + '.png', nrow=10)  # normalize=True
    gen_model.train()


def evaluate_models(local_models, ws_glob, dataset_test, args, iter, best_perf):
    acc_test_tot = []

    for i in range(args.num_models):
        model_e = local_models[i]
        model_e.load_state_dict(ws_glob[i])
        model_e.eval()
        acc_test, loss_test = test_img(model_e, dataset_test, args)

        if acc_test > best_perf[i]:
            best_perf[i] = float(acc_test)

        acc_test_tot.append(acc_test)
        print("Testing accuracy " + str(i) + ": {:.2f}".format(acc_test))

        if args.wandb:
            wandb.log({
                "Communication round": iter,
                "Local model " + str(i) + " test accuracy": acc_test
            })

    if args.wandb:
        wandb.log({
            "Communication round": iter,
            "Mean test accuracy": sum(acc_test_tot) / len(acc_test_tot)
        })

    return best_perf
