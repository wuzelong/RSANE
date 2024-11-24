import warnings
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.utils import to_dense_adj
import utils
from torch_geometric.datasets import AttributedGraphDataset, Amazon, Twitch, WebKB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

warnings.simplefilter(action='ignore', category=FutureWarning)
def ten_fold_cross_validation(x, y, seed):
    MiF1 = []
    MaF1 = []
    LR = LogisticRegression(penalty='l2', solver='liblinear', random_state=seed)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LR.fit(x_train, y_train)
        LR_pred = LR.predict(x_test)
        MiF1.append(f1_score(y_test, LR_pred, average='micro', zero_division=1))
        MaF1.append(f1_score(y_test, LR_pred, average='macro', zero_division=1))
    avgMi = sum(MiF1) / len(MiF1)
    avgMa = sum(MaF1) / len(MaF1)
    print(round(avgMi, 4), round(avgMa, 4))


def node_classification_radio(x, y, seed):
    MiF1 = [0] * 10
    MaF1 = [0] * 10
    for times in range(10):
        seed_everything(seed + times)
        idx = []

        kfold = KFold(n_splits=10, shuffle=True, random_state=seed + times)
        for train_index, test_index in kfold.split(x):
            idx.append(test_index)

        for i in range(9):
            test_index = []
            train_index = []
            for j in range(i + 1):
                train_index.append(idx[j])
            for j in range(i + 1, 10):
                test_index.append(idx[j])
            test_index = np.concatenate(test_index).ravel()
            train_index = np.concatenate(train_index).ravel()

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            LR = LogisticRegression(penalty='l2', solver='liblinear', random_state=seed + times)
            LR.fit(x_train, y_train)
            LR_pred = LR.predict(x_test)
            MiF1[i] += f1_score(y_test, LR_pred, average='micro', zero_division=1)
            MaF1[i] += f1_score(y_test, LR_pred, average='macro', zero_division=1)
    for i in range(9):
        print(round(MiF1[i] / 10, 4), round(MaF1[i] / 10, 4))


if __name__ == '__main__':
    args = utils.parse_args()
    seed_everything(args.seed)
    if args.data == 'Cora' or args.data == 'CiteSeer':
        data = AttributedGraphDataset(root='data/', name=args.data)
    elif args.data == 'Amazon':
        data = Amazon(root='data/', name='Photo')
    elif args.data == 'Twitch':
        data = Twitch(root='data/', name='DE')
    elif args.data == 'WebKB':
        data = WebKB(root='data/', name='Cornell')
    data = data[0]

    A = to_dense_adj(data.edge_index)[0]
    rows, cols = A.size()
    diag_indices = torch.arange(min(rows, cols))
    A[diag_indices, diag_indices] = 1
    X = data.x
    Y = data.y
    data.edge_index = torch.nonzero(A, as_tuple=False).t()

    encode, decode_A, decode_X, scores = utils.train_RSANE(A, X, args)

    if torch.is_tensor(encode):
        x = encode.cpu().detach().numpy()
    else:
        x = encode
    print('ten-fold:')
    ten_fold_cross_validation(x, Y, args.seed)
    print('train radio from 0.1 to 0.9:')
    node_classification_radio(x, Y, args.seed)
