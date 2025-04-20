from torchvision import datasets, transforms

def generator_traindata(args):
    if args.dataset == 'celebA':
        if args.gen_model == 'ddpm' or args.gen_model == 'vae':
            tf = transforms.Compose([transforms.CenterCrop(178), transforms.Resize(args.img_size), transforms.ToTensor(),])
        elif args.gen_model == 'gan':
            tf = transforms.Compose([transforms.CenterCrop(178), transforms.Resize(args.img_size), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    else:
        if args.gen_model == 'ddpm' or args.gen_model == 'vae':
            tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),])
        elif args.gen_model == 'gan':
            tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
        

    if args.dataset == 'mnist':
        train_data = datasets.MNIST(root='.data/mnist', train=True, transform=tf, download=True)
    elif args.dataset == 'fmnist':
        train_data = datasets.FashionMNIST('.data/fmnist', train=True, transform=tf, download=True)
    elif args.dataset == 'cifar10':
        train_data = datasets.CIFAR10('.data/cifar10', train=True, transform=tf, download=True)
    elif args.dataset == 'celebA':
        train_data = datasets.CelebA('.data/celeba', split='train', target_type='attr', download=True, transform=tf)                
        # Image batch shape: torch.Size([BS, 3, 64, 64])
        # Attribute batch shape: torch.Size([BS, 40])
        # First image attributes: tensor([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #         1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])

    return train_data