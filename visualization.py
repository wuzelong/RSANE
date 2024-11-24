import warnings
from sklearn.utils.validation import check_scalar
import torch
from torch_geometric import seed_everything
from torch_geometric.datasets import AttributedGraphDataset, Amazon, Twitch, WebKB
from torch_geometric.utils import to_dense_adj
import utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from matplotlib.colors import ListedColormap
from utils import parse_args

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    args = parse_args()
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

    A = to_dense_adj(data.edge_index)[0]
    rows, cols = A.size()
    diag_indices = torch.arange(min(rows, cols))
    A[diag_indices, diag_indices] = 1
    X = data.x
    Y = data.y
    data.edge_index = torch.nonzero(A, as_tuple=False).t()

    encode, decode_A, decode_X, scores = utils.train_RSANE(A, X, args)

    if torch.is_tensor(encode):
        encode = encode.detach().cpu().numpy()
    tsne = TSNE(n_components=2, metric='cosine', perplexity=30, random_state=args.seed)
    X_tsne = tsne.fit_transform(encode)

    sc = round(silhouette_score(X_tsne, Y), 4)
    chs = round(calinski_harabasz_score(X_tsne, Y), 4)
    print('SC:',sc,'CHS:',chs)

    # t-sne
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive']
    # cmap = ListedColormap(colors)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap=cmap, s=3)
    # plt.axis('off')
    # plt.savefig('RSANE.pdf')
    # plt.show()
