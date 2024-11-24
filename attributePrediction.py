import warnings
import torch
from torch_geometric import seed_everything
from torch_geometric.datasets import AttributedGraphDataset, Amazon, Twitch, WebKB
from torch_geometric.utils import to_dense_adj
import utils

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    num_features = data.num_features
    data = data[0]

    X = data.x
    Y = data.y
    A = to_dense_adj(data.edge_index)[0]
    pos = torch.load('data/posX/' + args.data + '_X_pos_' + args.radio + '.pth')
    neg = torch.load('data/negX/' + args.data + '_X_neg_' + args.radio + '.pth')
    X[pos[:, 0], pos[:, 1]] = 0
    rows, cols = A.size()
    diag_indices = torch.arange(min(rows, cols))
    A[diag_indices, diag_indices] = 1
    data.edge_index = torch.nonzero(A, as_tuple=False).t()

    encode, decode_A, decode_X, scores = utils.train_RSANE(A, X, args)

    decode = decode_X
    if torch.is_tensor(decode):
        decode = decode.detach().cpu().numpy()
    rankings, labels = utils.get_rankings_2D(decode, pos, neg)
    auc = utils.get_AUC(rankings, labels)
    recall = utils.get_Recall(labels, len(pos), len(pos))
    print('AUC:', auc, 'Recall:', recall)
