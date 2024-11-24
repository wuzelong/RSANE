import warnings
import torch
from torch_geometric import seed_everything
from torch_geometric.utils import to_dense_adj
import pickle
import utils

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    args = utils.parse_args()
    seed_everything(args.seed)

    with open('data/outliers/' + args.data + '.pkl', 'rb') as f:
        data = pickle.load(f)
    X = data.x
    Y = data.y
    A = to_dense_adj(data.edge_index)[0]
    rows, cols = A.size()
    diag_indices = torch.arange(min(rows, cols))
    A[diag_indices, diag_indices] = 1
    data.edge_index = torch.nonzero(A, as_tuple=False).t()

    encode, decode_A, decode_X, scores = utils.train_RSANE(A, X, args)
    if torch.is_tensor(encode):
        encode = encode.detach().cpu().numpy()

    scores *= -1
    outliers_idx = torch.load('data/outliers/' + args.data + '_outliers_idx' + '.pth')
    normal_idx = torch.load('data/outliers/' + args.data + '_normal_idx' + '.pth')
    rankings, labels = utils.get_rankings_1D(scores, outliers_idx, normal_idx)
    auc = utils.get_AUC(rankings, labels)
    print('AUC:', auc)
    recall = utils.get_Recall(labels, len(outliers_idx), int(len(rankings) * 0.25))
    print('Recall@25%:', recall)
