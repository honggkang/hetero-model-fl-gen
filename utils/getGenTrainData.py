from torchvision import datasets, transforms

def generator_traindata(args):
    if args.gen_model == 'ddpm' or args.gen_model == 'vae':
        tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),])
    elif args.gen_model == 'gan':
        tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    if args.dataset == 'mnist':
        train_data = datasets.MNIST(root='/home/hong/NeFL/.data/mnist', train=True, transform=tf, download=True)
    elif args.dataset == 'fmnist':
        train_data = datasets.FashionMNIST('/home/hong/NeFL/.data/fmnist', train=True, transform=tf, download=True)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        train_data = datasets.CIFAR10('/home/hong/NeFL/.data/cifar', train=True, download=True, transform=transform_train)
    return train_data