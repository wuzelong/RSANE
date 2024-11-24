import numpy as np
import torch
from sklearn import metrics
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models.RSANE import RSANE


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', default='Cora', help="the name of dataset")
    parser.add_argument('--epochs', default=100, type=int, help="the number of training epochs")
    parser.add_argument('--lr', default=0.001, type=float, help="the learning rate of optimizer")
    parser.add_argument('--seed', default=20231214, type=int, help="random seed")
    parser.add_argument('--radio', default='005', help='hide radio for link prediction')

    parser.add_argument('--adj_sizes', default=[2708, 270], nargs='+', type=int, help='RSANE')
    parser.add_argument('--att_sizes', default=[1433, 143], nargs='+', type=int, help='RSANE')
    parser.add_argument('--hidden_sizes', default=[128], nargs='+', type=int, help='RSANE')
    parser.add_argument('--beta', default=1, type=float, help='RSANE')
    parser.add_argument('--eta1', default=0, type=float, help='RSANE')
    parser.add_argument('--eta2', default=0, type=float, help='RSANE')
    parser.add_argument('--gam1', default=2, type=float, help='RSANE')
    parser.add_argument('--gam2', default=2, type=float, help='RSANE')
    parser.add_argument('--alpha', default=0.5, type=float, help='RSANE')
    parser.add_argument('--mu', default=0.5, type=float, help='RSANE')
    parser.add_argument('--ksi', default=0.9, type=float, help='RSANE')

    args = parser.parse_args()

    return args


def laplacian_norm(A):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    D = torch.diag(A.sum(dim=1)).to(device)
    try:
        D_inv = torch.inverse(D)
    except torch.linalg.LinAlgError:
        D_inv = torch.pinverse(D)
    D = torch.sqrt(D_inv)
    return D @ A @ D


def cosine_similarity(X):
    norm = X / X.norm(dim=1, keepdim=True)
    norm[torch.isnan(norm)] = 0
    cos_sim = norm @ norm.t()
    return cos_sim


def Q_similarity(U, V, alpha, beta, mu):
    n = len(U)
    I = torch.eye(n)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    I = I.to(device)
    Q = (1 - alpha) * U + beta * beta * alpha * V
    tmp = I - mu * V
    try:
        tmp = torch.inverse(tmp) - I
    except torch.linalg.LinAlgError:
        tmp = torch.pinverse(tmp) - I
    Q = Q * tmp
    return Q


def train_RSANE(A, X, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RSANE(args.adj_sizes, args.att_sizes, args.hidden_sizes, args.eta1, args.eta2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
    A = A.to(device)
    X = X.to(device)

    L = laplacian_norm(A)
    R = cosine_similarity(X)
    QL = Q_similarity(R, L, args.alpha, args.beta, args.mu)
    SL = torch.exp(QL)

    C1 = args.gam1 + SL
    C1[A == 0] = 1

    C2 = torch.ones_like(X)
    C2[X != 0] = args.gam2
    idx_X = torch.nonzero(X)
    QLX = QL @ X
    C2[idx_X[:, 0], idx_X[:, 1]] += QLX[idx_X[:, 0], idx_X[:, 1]]

    C3 = args.ksi * L + (1 - args.ksi) * QL

    model.train()
    model_opt = model
    loss_opt = -1
    tag = True
    lamb = torch.log(torch.pow(torch.tensor([1 / len(A)] * len(A)), -1)).to(device)

    for epoch in range(args.epochs):
        opt.zero_grad()
        Loss, o = model(A, X, C1, C2, C3, lamb)
        lamb = torch.log(torch.pow(o / Loss.item(), -1))
        Loss.backward()
        opt.step()
        if tag or loss_opt > Loss:
            loss_opt = Loss
            model_opt = model
            tag = False

    model_opt.eval()
    model_opt = model_opt.to(device)
    encode, decode_A, decode_X = model_opt.savector(A, X)
    return encode, decode_A, decode_X, lamb.view(-1).detach().cpu().numpy()


def get_rankings_2D(scores, pos, neg):
    pos_rankings = scores[pos[:, 0], pos[:, 1]]
    pos_labels = np.ones_like(pos_rankings)

    neg_rankings = scores[neg[:, 0], neg[:, 1]]
    neg_labels = np.zeros_like(neg_rankings)

    rankings = np.concatenate([pos_rankings, neg_rankings])
    labels = np.concatenate([pos_labels, neg_labels])

    sorted_indices = np.argsort(rankings)[::-1]
    rankings = rankings[sorted_indices]
    labels = labels[sorted_indices]
    return rankings, labels


def get_AUC(rankings, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, rankings)
    auc = metrics.auc(fpr, tpr)
    return round(auc, 4)


def get_Recall(labels, total, k):
    cnt = 0
    for i in range(k):
        if labels[i] == 1:
            cnt += 1
    recall = cnt / total
    return round(recall, 4)


def get_rankings_1D(scores, pos, neg):
    pos_rankings = scores[pos]
    pos_labels = np.ones_like(pos_rankings)

    neg_rankings = scores[neg]
    neg_labels = np.zeros_like(neg_rankings)

    rankings = np.concatenate([pos_rankings, neg_rankings])
    labels = np.concatenate([pos_labels, neg_labels])

    sorted_indices = np.argsort(rankings)[::-1]
    rankings = rankings[sorted_indices]
    labels = labels[sorted_indices]
    return rankings, labels
