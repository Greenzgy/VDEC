import h5py
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import torchvision
from torch.utils.data import TensorDataset, DataLoader


def get_dataset(name, datasets_path):
    if name == 'mnist':
        transform = torchvision.transforms.Resize(32)
        train = torchvision.datasets.MNIST(root=datasets_path, train=True, transform=None, download=True)
        test = torchvision.datasets.MNIST(root=datasets_path, train=False, transform=None, download=True)
        x_train, y_train = train.train_data, train.train_labels
        x_test, y_test = test.test_data, test.test_labels
        x = torch.cat((x_train, x_test), 0)
        y = torch.cat((y_train, y_test), 0)
        x = transform(x)
        x = np.divide(x, 255)
        x = x.unsqueeze(1).to(torch.float32)
        y = y.cpu().numpy()

        return x, y

    elif name == 'fashionmnist':
        transform = torchvision.transforms.Resize(32)
        train = torchvision.datasets.FashionMNIST(root=datasets_path, train=True, transform=None, download=True)
        test = torchvision.datasets.FashionMNIST(root=datasets_path, train=False, transform=None, download=True)
        x_train, y_train = train.train_data, train.train_labels
        x_test, y_test = test.test_data, test.test_labels
        x = torch.cat((x_train, x_test), 0)
        y = torch.cat((y_train, y_test), 0)
        x = transform(x)
        x = np.divide(x, 255).unsqueeze(1)
        y = y.cpu().numpy()

        return x, y

    elif name == 'usps':
        with h5py.File(datasets_path + '/usps.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
            x = np.concatenate((X_tr, X_te), 0)
            y = np.concatenate((y_tr, y_te), 0)
            x = x.reshape(x.shape[0], 16, 16)
            x = torch.tensor(x)
            x = x.unsqueeze(1).to(torch.float32)

        return x, y

    elif name == 'stl-10':
        train = torchvision.datasets.STL10(root=datasets_path, split='train', transform=torchvision.transforms.ToTensor(), download=True)
        test = torchvision.datasets.STL10(root=datasets_path, split='test', transform=torchvision.transforms.ToTensor(), download=True)
        x_train, y_train = train.data, train.labels
        x_test, y_test = test.data, test.labels
        X = np.concatenate((x_train, x_test), 0)
        Y = np.concatenate((y_train, y_test), 0)

        image_train = X.astype('float32') / 255

        image_train[:, 0, :, :] = (image_train[:, 0, :, :] - 0.485) / 0.229
        image_train[:, 1, :, :] = (image_train[:, 1, :, :] - 0.456) / 0.224
        image_train[:, 2, :, :] = (image_train[:, 2, :, :] - 0.406) / 0.225

        res50_model = torchvision.models.resnet50(pretrained=True)
        res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
        res50_conv.eval()
        data = torch.from_numpy(image_train)
        dataloader = DataLoader(TensorDataset(data), batch_size=200, shuffle=False)
        res50_conv = res50_conv.cuda()
        total_output = []
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0].cuda()
            output = res50_conv(inputs)
            total_output.append(output.data)
        total_output = torch.cat(total_output, dim=0)
        feature_train = torch.sum(torch.sum(total_output, dim=-1), dim=-1) / 9

        return feature_train, Y

    elif name == 'reuters10k':
        data = scio.loadmat('dataset/reuters10k/reuters10k.mat')
        X = data['X']
        Y = data['Y'].squeeze()
        X = torch.tensor(X)
        X = X.to(torch.float32)

        return X, Y

    elif name == 'har':
        data = scio.loadmat('dataset/har/HAR.mat')
        X = data['X']
        Y = data['Y'] - 1
        X = X[:10200]
        Y = Y[:10200].squeeze()
        X = torch.tensor(X)
        X = X.to(torch.float32)

        return X, Y