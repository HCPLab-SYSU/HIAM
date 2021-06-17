import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform
from torch.nn.init import xavier_normal_, calculate_gain


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_0 \cdot \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 bias=True,
                 **kwargs):
        super(RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # size = self.num_bases * self.in_channels
        # uniform(size, self.basis)
        # uniform(size, self.att)
        # uniform(size, self.root)
        # xavier_normal_(self.basis, gain=calculate_gain('relu'))
        # xavier_normal_(self.root, gain=calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.basis)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.root)

    def forward(self, x, edge_index, edge_attr, edge_norm=None):
        """"""
        # print("forward:", x.shape, edge_attr.shape)
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_attr, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.to(x_j.device)  # 加cuda
        w = w.view(self.num_relations, self.in_channels, self.out_channels)  # (num_relations, in, out)
        out = torch.einsum('bi,rio->bro', x_j, w)  # x_j: (batch, in)  w:(num_relations, in, out)  out:(batch, num_relations, out)  先转置w再相乘
        # print("message1:", x_j.size(), w.size(), out.size())
        out = (out * edge_attr.unsqueeze(2)).sum(dim=1)
        # print("message2:", out.size(), edge_attr.unsqueeze(2).size())
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if x is None:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        # print("update:", out.size())
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
