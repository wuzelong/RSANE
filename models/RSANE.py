import torch
import torch.nn as nn
import torch.nn.functional as F


class RSANE(nn.Module):
    def __init__(self, adj_sizes, att_sizes, hidden_sizes, eta1, eta2):
        super(RSANE, self).__init__()
        self.eta1 = eta1
        self.eta2 = eta2
        self.K = len(hidden_sizes)

        self.pre_A_encode = nn.Linear(adj_sizes[0], adj_sizes[1])
        self.pre_X_encode = nn.Linear(att_sizes[0], att_sizes[1])

        self.con_A_encode = nn.Linear(adj_sizes[1], hidden_sizes[0])
        self.con_X_encode = nn.Linear(att_sizes[1], hidden_sizes[0])

        self.encode_layers = nn.ModuleList()
        for i in range(self.K - 1):
            self.encode_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.decode_layers = nn.ModuleList()
        for i in range(self.K - 1, 0, -1):
            self.decode_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i - 1]))

        self.con_A_decode = nn.Linear(hidden_sizes[0], adj_sizes[1])
        self.con_X_decode = nn.Linear(hidden_sizes[0], att_sizes[1])

        self.pre_A_decode = nn.Linear(adj_sizes[1], adj_sizes[0])
        self.pre_X_decode = nn.Linear(att_sizes[1], att_sizes[0])

    def forward(self, A, X, C1, C2, C3, lamb):
        encode, decode_A, decode_X = self.savector(A, X)
        encode_norm1 = torch.sum(encode * encode, dim=1, keepdim=True)

        L_str1 = (decode_A - A) * C1
        L_str1 = torch.sum(L_str1 * L_str1, dim=1) * self.eta1

        L_att1 = (decode_X - X) * C2
        L_att1 = torch.sum(L_att1 * L_att1, dim=1) * self.eta2

        L_loc1 = (encode_norm1 - 2 * torch.mm(encode, torch.transpose(encode, dim0=0, dim1=1))
                  + torch.transpose(encode_norm1, dim0=0, dim1=1)) * C3
        L_loc1 = torch.sum(L_loc1, dim=1)

        o1 = (L_str1 + L_att1 + L_loc1) * lamb
        return torch.sum(o1), o1.detach()

    def savector(self, A, X):
        encode_A = F.leaky_relu(self.pre_A_encode(A))
        encode_X = F.leaky_relu(self.pre_X_encode(X))

        encode = F.leaky_relu(self.con_A_encode(encode_A)) + F.leaky_relu(self.con_X_encode(encode_X))
        for i in range(len(self.encode_layers)):
            encode = F.leaky_relu(self.hidden_layers[i](encode))

        decode = encode
        for i in range(len(self.decode_layers) - 1):
            decode = F.leaky_relu(self.decode_layers[i](decode))

        decode_A = F.leaky_relu(self.con_A_decode(decode))
        decode_X = F.leaky_relu(self.con_X_decode(decode))

        decode_A = F.leaky_relu(self.pre_A_decode(decode_A))
        decode_X = F.leaky_relu(self.pre_X_decode(decode_X))
        return encode, decode_A, decode_X
