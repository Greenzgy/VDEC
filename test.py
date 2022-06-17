import numpy as np
import torch
from linear_assignment_ import linear_assignment
from model.VDEC import *
from sklearn import metrics
from dataloader import *
from torch.utils.data import DataLoader

def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w


def modeltest(device, dataset, input_path, input_dim, model_path, class_num, batch):
    print('-----test-----')
    data, label = get_dataset(dataset, input_path)
    test_loader = DataLoader(dataset=data, batch_size=batch, shuffle=False, num_workers=class_num)

    if dataset == 'mnist' or 'fashionmnist':
        model = VDECCNN().to(device)

    elif dataset == 'usps':
        model = VDECUSPS().to(device)

    elif dataset == 'stl-10' or 'reuters10k' or 'har':
        model = VDECMlp(class_num, input_dim).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

    with torch.no_grad():
        q = []
        for x in test_loader:
            x = x.to(device)
            _, _, _, _, Q, _ = model(x)
            q.append(Q)
        q = torch.cat(q, 0)
        c = np.argmax(q.detach().cpu().numpy(), axis=1)
        nmi = metrics.normalized_mutual_info_score(label, c)
        print('Acc={:.4f}%, NMI={:.4f}'.format(cluster_acc(c, label)[0] * 100, nmi))