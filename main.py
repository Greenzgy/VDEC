import warnings
import argparse
import torch
import os
# from trainer import train
from dataloader import *
from test import modeltest



def main():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='har')
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--plr', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--loss_path', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.dataset == 'mnist':
        args.M = 10
        args.plr = 0.0001
        args.lr = 0.00001
        args.batch = 256
        args.n_epochs = 300
        args.pretrain_epochs = 50
        args.pretrain_path = "save/pretrain_mnist.pkl"
        args.input_path = "dataset"
        args.loss_path = "save/mnist_loss.txt"
        args.model_path = "save/mnist.pkl"

    elif args.dataset == 'fashionmnist':
        args.M = 10
        args.plr = 0.0001
        args.lr = 0.00001
        args.batch = 256
        args.n_epochs = 300
        args.pretrain_epochs = 50
        args.pretrain_path = "save/pretrain_fashionmnist.pkl"
        args.input_path = "dataset"
        args.loss_path = "save/fashionmnist_loss.txt"
        args.model_path = "save/fashionmnist.pkl"

    elif args.dataset == 'usps':
        args.M = 10
        args.plr = 0.001
        args.lr = 0.0001
        args.batch = 100
        args.n_epochs = 300
        args.pretrain_epochs = 100
        args.pretrain_path = "save/pretrain_usps.pkl"
        args.input_path = "dataset"
        args.loss_path = "save/usps_loss.txt"
        args.model_path = "save/usps.pkl"

    elif args.dataset == 'stl-10':
        args.M = 10
        args.plr = 0.0001
        args.lr = 0.00001
        args.batch = 128
        args.input_dim = 2048
        args.n_epochs = 300
        args.pretrain_epochs = 100
        args.pretrain_path = "save/pretrain_stl-10.pkl"
        args.input_path = "dataset"
        args.loss_path = "save/stl-10_loss.txt"
        args.model_path = "save/stl-10.pkl"

    elif args.dataset == 'reuters10k':
        args.M = 4
        args.plr = 0.0001
        args.lr = 0.00001
        args.batch = 100
        args.input_dim = 2000
        args.n_epochs = 300
        args.pretrain_epochs = 50
        args.pretrain_path = "save/pretrain_reuters10k.pkl"
        args.input_path = "dataset"
        args.loss_path = "save/reuters10k_loss.txt"
        args.model_path = "save/reuters10k.pkl"

    elif args.dataset == 'har':
        args.M = 6
        args.plr = 0.0001
        args.lr = 0.00001
        args.batch = 100
        args.input_dim = 561
        args.n_epochs = 300
        args.pretrain_epochs = 50
        args.pretrain_path = "save/pretrain_har.pkl"
        args.input_path = "dataset"
        args.loss_path = "save/har_loss.txt"
        args.model_path = "save/har.pkl"

    modeltest(device, args.dataset, args.input_path, args.input_dim, args.model_path, args.M, args.batch)


if __name__ == "__main__":
    main()