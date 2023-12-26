import numpy as np
import scipy as sp
import torch
from torch import nn, optim
from functools import partial
import torch as t
import scipy.io as scio
from collections import namedtuple
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn import metrics

#加高斯核
def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = t.mm(y, y.T)  #torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = t.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne


def kernelToDistance(k):
    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[t.nonzero(k, as_tuple=False)[0]]
    K[t.where(K == 0)] = min_v
    D = t.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S


def laplacian(kernel):
    d1 = sum(kernel)
    D_1 = t.diag(d1)
    L_D_1 = D_1 - kernel
    D_5 = D_1.rsqrt()
    D_5 = t.where(t.isinf(D_5), t.full_like(D_5, 0), D_5)
    L_D_11 = t.mm(D_5, L_D_1)
    L_D_11 = t.mm(L_D_11, D_5)
    return L_D_11


class Sizes(object):
    def __init__(self, D_size, R_size):
        self.D_size = D_size
        self.R_size = R_size

        self.h1_gamma = 2 ** (-5) #γ
        self.h2_gamma = 2 ** (-3)
        self.h3_gamma = 2 ** (-3)

        self.lambda1 = 2 ** (-3) #λ
        self.lambda2 = 2 ** (-4)



def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)


def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)


def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=edge_index.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        # self.bn = nn.BatchNorm1d(in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     nn.init.xavier_uniform_(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight):
        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, self.add_self_loops)
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]
        # x = self.bn(x)
        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class EdgeDropout(nn.Module):
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__() #首先找到EdgeDropout的父类（比如是类nn.Module），然后把类EdgeDropout的对象self转换为类nn.Module的对象，然后“被转换”的类nn.Module对象调用自己的init函数
        assert keep_prob>0 #条件为keep_prob>0时正常执行，否则触发异常
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index.shape[1], device=edge_weight.device)
            mask = torch.floor(mask+self.p).type(torch.bool)
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]/self.p #边的权重
        return edge_index, edge_weight

    def forward2(self, edge_index):
        if self.training:
            mask = ((torch.rand(edge_index._values().size()) + (self.keep_prob)).floor()).type(torch.bool)
            rc = edge_index._indices()[:, mask]
            val = edge_index._values()[mask]/self.p
            return torch.sparse.FloatTensor(rc, val)
        return edge_index

    def __repr__(self):
        return '{}(keep_prob={})'.format(self.__class__.__name__, self.keep_prob)


class ShareGCN(nn.Module):
    def __init__(self, size_u, size_v, in_channels=64, out_channels=64, share=True, normalize=True,
                 dropout=0.4, use_sparse=True, act=nn.ReLU, cached=False, bias=False, add_self_loops=False,
                 **kwargs): #每层有64个隐藏单元，规则退出率为0.4
        super(ShareGCN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u+size_v
        self.share = share
        self.use_sparse = use_sparse
        self.dropout = nn.Dropout(dropout)#↓编码器
        self.u_encoder = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                     normalize=normalize, add_self_loops=add_self_loops,
                                     cached=cached, bias=bias, **kwargs)
        if not self.share:
            self.v_encoder = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                         normalize=normalize, add_self_loops=False,
                                         cached=cached, bias=bias, **kwargs)
        self.act = act(inplace=True) if act else nn.Identity() #激活函数maybe #act=nn.ReLU

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight):
        x = self.dropout(x)
        if self.share:
            edge_index = torch.cat([u_edge_index, v_edge_index], dim=1) #横着拼接，使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐
            edge_weight = torch.cat([u_edge_weight, v_edge_weight], dim=0)#竖着拼接
            if self.use_sparse:
                node_nums = self.num_nodes
                edge_index = SparseTensor(row=edge_index[0], col=edge_index[1],
                                          value=edge_weight,
                                          sparse_sizes=(node_nums, node_nums)).t()
            feature = self.u_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            if self.use_sparse:
                node_nums = self.num_nodes
                u_edge_index = SparseTensor(row=u_edge_index[0], col=u_edge_index[1],
                                           value=u_edge_weight,
                                           sparse_sizes=(node_nums, node_nums)).t()
                v_edge_index = SparseTensor(row=v_edge_index[0], col=v_edge_index[1],
                                            value=v_edge_weight,
                                            sparse_sizes=(node_nums, node_nums)).t()
            feature_u = self.u_encoder(x=x, edge_index=u_edge_index, edge_weight=u_edge_weight)
            feature_v = self.v_encoder(x=x, edge_index=v_edge_index, edge_weight=v_edge_weight)
            feature = torch.cat([feature_u[:self.size_u], feature_v[self.size_u:]])

        output = self.act(feature)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""#""链接预测的解码器模型层。"＂＂

    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, act=nn.Sigmoid):#act=nn.Sigmoid
        super(InnerProductDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, feature1,feature2):
        feature1 = self.dropout(feature1)
        feature2 = self.dropout(feature2)

        outputs = (feature1 + feature2.T) / 2  # F*
        outputs = self.act(outputs) #

        return outputs#, R, D


class InnerProductDecoder_gei(nn.Module):
    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, act=nn.Sigmoid):  # act=nn.Sigmoid
        super(InnerProductDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, feature):
        feature = self.dropout(feature)
        R = feature[:self.size_u]  # 药物 #R:1373*128
        D = feature[self.size_u:]  # 微生物  #D:173*128

        if hasattr(self, "weights"):
            D = self.weights(D)

        x = R @ D.T  # 旧：1373*173 #原始代码
        outputs = self.act(x)  # 旧：1373*173#原始代码
        return outputs, R, D



class Dataset():
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        mask = mask.astype(bool)
        self.stage = stage
        self.one_mask = torch.from_numpy(dataset.interactions>0)
        row, col = np.nonzero(mask&dataset.interactions.astype(bool))
        self.valid_row = torch.tensor(np.unique(row))  #该函数是去除数组中的重复数字，并进行排序之后输出。
        self.valid_col = torch.tensor(np.unique(col))
        if not fill_unkown:
            row_idx, col_idx = np.nonzero(mask)
            self.interaction_edge = torch.LongTensor([row_idx, col_idx]).contiguous()
            self.label = torch.from_numpy(dataset.interactions[mask]).float().contiguous()
            self.valid_mask = torch.ones_like(self.label, dtype=torch.bool)
            self.matrix_mask = torch.from_numpy(mask)
        else:
            row_idx, col_idx = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1])) #torch.meshgrid（）的功能是生成网格，可以用于生成坐标。函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。
            self.interaction_edge = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])  # torch.stack（）官方解释：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
            self.label = torch.clone(torch.from_numpy(dataset.interactions)).float()  #torch.clone返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯
            self.label[~mask] = 0
            self.valid_mask = torch.from_numpy(mask)
            self.matrix_mask = torch.from_numpy(mask)

        self.drug_edge = dataset.drug_edge
        self.microbe_edge = dataset.microbe_edge

        self.u_embedding = torch.from_numpy(dataset.drug_sim).float() #torch.from_numpy(ndarray) → Tensor，即 从numpy.ndarray创建一个张量。
        self.v_embedding = torch.from_numpy(dataset.microbe_sim).float()

        self.mask = torch.from_numpy(mask)
        pos_num = self.label.sum().item()
        neg_num = np.prod(self.mask.shape) - pos_num
        self.pos_weight = neg_num / pos_num

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.mask.shape}, interaction_num={len(self.interaction_edge)}, pos_weight={self.pos_weight})"

    @property
    def size_u(self):
        return self.mask.shape[0]

    @property
    def size_v(self):
        return self.mask.shape[1]

    def get_u_edge(self, union_graph=False):
        edge_index, value, size = self.drug_edge
        if union_graph:
            size = (self.size_u+self.size_v, )*2   #（1546，1546）
        return edge_index, value, size

    def get_v_edge(self, union_graph=False):
        edge_index, value, size = self.microbe_edge
        if union_graph:
            edge_index = edge_index + torch.tensor(np.array([[self.size_u], [self.size_u]]))
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_uv_edge(self, union_graph=False):
        train_mask = self.mask if self.stage=="train" else ~self.mask
        train_one_mask = train_mask & self.one_mask
        edge_index = torch.nonzero(train_one_mask).T
        value = torch.ones(edge_index.shape[1])
        size =  (self.size_u, self.size_v)
        if union_graph:
            edge_index = edge_index + torch.tensor([[0], [self.size_u]])
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_vu_edge(self, union_graph=False):
        edge_index, value, size = self.get_uv_edge(union_graph=union_graph)
        edge_index = reversed(edge_index)
        return edge_index, value, size

    def get_union_edge(self, union_type="u-uv-vu-v"):
        types = union_type.split("-")
        edges = []
        size = (self.size_u+self.size_v, )*2
        for type in types:
            assert type in ["u","v","uv","vu"]
            edge = self.__getattribute__(f"get_{type}_edge")(union_graph=True)
            edges.append(edge)
        edge_index = torch.cat([edge[0].int() for edge in edges], dim=1)
        value = torch.cat([edge[1] for edge in edges], dim=0) #torch.cat（）在给定维度上对输入的张量序列seq 进行连接操作。
        return edge_index, value, size

    @staticmethod
    def collate_fn(batch):
        return batch



FullGraphData = namedtuple("FullGraphData", ["u_edge", "v_edge",
                                             "embedding", "edge",
                                             "uv_edge", "vu_edge",
                                              "label", "interaction_pair", "valid_mask"])


class FullGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, **kwargs):
        super(FullGraphDataset, self).__init__(dataset, mask, fill_unkown=True, **kwargs)
        assert fill_unkown, "fill_unkown need True!"
        self.data = self.build_data()

    def build_data(self):
        u_edge = self.get_u_edge(union_graph=True)                    #构建三个网络，药物药物，微生物微生物，药物微生物
        v_edge = self.get_v_edge(union_graph=True)
        uv_edge = self.get_uv_edge(union_graph=True)
        vu_edge = self.get_vu_edge(union_graph=True)
        edge = self.get_union_edge(union_type="u-uv-vu-v")
        x = self.get_union_edge(union_type="u-v")
           #x可能就是H0,      GCN(A,H,W)
        x = torch.sparse_coo_tensor(indices=x[0], values=x[1], size=x[2])#torch.sparse_coo_tensor创建稀疏矩阵
        # x = x.to_dense()

        norm_x = gcn_norm(edge_index=x, add_self_loops=False).to_dense()
        x = norm_x * torch.norm(x) / torch.norm(norm_x)  #归一化

        data = FullGraphData(u_edge=u_edge,
                             v_edge=v_edge,
                             uv_edge=uv_edge,
                             vu_edge=vu_edge,
                             edge=edge,
                             label=self.label,
                             valid_mask=self.valid_mask,
                             interaction_pair=self.interaction_edge,
                             embedding=x,
                             )
        return data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data


class SmoothDecoder(nn.Module):
    def __init__(self, size_u, size_v, k=20, act=nn.Sigmoid):
        super(SmoothDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.k = k
        self.act = act() if act is not None else nn.Identity()

    def merge_neighbor_feature(self, sims, features, k=5):#融合邻居特征
        assert sims.shape[0] == features.shape[0] and sims.shape[1] == sims.shape[0]
        if k<0:
            k = sims.shape[1]
        N = features.shape[0]
        value, idx = torch.topk(sims, dim=1, k=k)
        col = idx.reshape(-1)
        features = features[col].view(N, k, -1) * value.view(N, k, 1)
        features = features.sum(dim=1)
        features = features / value.sum(dim=1).view(N, 1)
        return features

    def forward(self, u, v, batch:FullGraphData):
        if not hasattr(self, "sim"): #hasattr() 函数用于判断对象是否包含对应的属性。
            indices = torch.cat([batch.u_edge[0], batch.v_edge[0]], dim=1)
            values = torch.cat([batch.u_edge[1], batch.v_edge[1]], dim=0)
            size = batch.u_edge[2]
            sim = torch.sparse_coo_tensor(indices, values, size).to_dense()
            self.register_buffer("sim", sim)
        if not hasattr(self, "mask"):
            if self.training:
                feature = torch.cat([u, v], dim=0)#竖着拼
                interactions = torch.sparse_coo_tensor(indices=batch.interaction_pair,
                                                       values=batch.label.reshape(-1),
                                                       size=(self.size_u, self.size_v)).to_dense()
                index = torch.nonzero(interactions)#找出tensor中非零的元素的索引
                u_idx, v_idx = index[:, 0].unique(), index[:, 1].unique()
                v_idx = v_idx+self.size_u
                mask = torch.zeros(feature.shape[0], 1, dtype=torch.bool, device=feature.device)
                mask[u_idx] = True
                mask[v_idx] = True
        elif not self.training:
            feature = torch.cat([u, v], dim=0)
            merged_feature = self.merge_neighbor_feature(self.sim, feature, self.k)
            feature = torch.where(self.mask, feature, merged_feature)
            u = feature[:self.size_u]
            v = feature[self.size_u:]

        x = u@v.T
        outputs = self.act(x)
        return outputs, u, v


class BaseModel(pl.LightningModule):
    DATASET_TYPE: None

    def __init__(self):
        super(BaseModel, self).__init__()

    def select_topk(self, data, k=-1):   #topk
        if k is None or k <= 0:
            return data
        assert k <= data.shape[1]
        val, col = torch.topk(data, k=k)
        col = col.reshape(-1)
        row = torch.ones(1, k, dtype=torch.int) * torch.arange(data.shape[0]).view(-1, 1)
        row = row.view(-1).to(device=data.device) #.view(-1)变成一维   data.device  用  cuda
        new_data = torch.zeros_like(data)   #维度一样，全为0    torch.zeros_like:生成和括号内变量维度维度一致的全是零的内容
        new_data[row, col] = data[row, col]
        return new_data

    def merge_neighbor_feature(self, sims, features, k=5):  #融合邻居特征，k=5
        assert sims.shape[0] == features.shape[0] and sims.shape[1] == sims.shape[0]
        if k<0:
            k = sims.shape[1]
        N = features.shape[0]
        value, idx = torch.topk(sims, dim=1, k=k)      #沿给定dim维度返回输入张量sims中 k 个最大值。
        col = idx.reshape(-1)
        features = features[col].view(N, k, -1) * value.view(N, k, 1)  #torch.FloatTensor of size N*K*1
        features = features.sum(dim=1)
        features = features / value.sum(dim=1).view(N, 1)
        return features

    def neighbor_smooth(self, sims, features, replace_rate=0.2):    #边缘退出率为0.2
        merged_u = self.merge_neighbor_feature(sims, features)
        mask = torch.rand(merged_u.shape[0], device=sims.device)
        mask = torch.floor(mask + replace_rate).type(torch.bool)
        new_features = torch.where(mask, merged_u, features)   #torch.where()函数的作用是按照一定的规则合并两个tensor类型。
        #torch.where(condition，a，b)   其中输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
        return new_features

    def laplacian_matrix(self, S):   #拉普拉斯矩阵
        x = torch.sum(S, dim=0)   #torch.sum()对输入的tensor数据的某一维度求和
        y = torch.sum(S, dim=1)
        L = 0.5*(torch.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def graph_loss_fn(self, x, edge, topk=None, cache_name=None, reduction="mean"):  #图损失
        if not hasattr(self, f"_{cache_name}") :
            adj = torch.sparse_coo_tensor(*edge).to_dense()
            adj = adj-torch.diag(torch.diag(adj))
            adj = self.select_topk(adj, k=topk) #选择前k个
            la = self.laplacian_matrix(adj)
            if cache_name:
                self.register_buffer(f"_{cache_name}", la)
        else:
            la = getattr(self, f"_{cache_name}")
            assert la.shape==edge[2]

        graph_loss = torch.trace(x.T@la@x)   #返回输入二维矩阵对角线元素的和。
        graph_loss = graph_loss/(x.shape[0]**2) if reduction=="mean" else graph_loss
        return graph_loss

    def mse_loss_fn(self, predict, label, pos_weight):   #均方差
        predict = predict.view(-1)
        label = label.view(-1)
        pos_mask = label>0
        loss = F.mse_loss(predict, label, reduction="none")
        loss_pos = loss[pos_mask].mean()
        loss_neg = loss[~pos_mask].mean()
        loss_mse = loss_pos*pos_weight+loss_neg
        return {"loss_mse":loss_mse,
                "loss_mse_pos":loss_pos,
                "loss_mse_neg":loss_neg,
                "loss":loss_mse}

    def bce_loss_fn(self, predict, label, pos_weight):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        weight = pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight).double()
        return {"loss_bce":loss,
                "loss":loss}

    def getGipKernel_loss_fn(self, target, predict, drug_lap, mic_lap, alpha1, alpha2):
        loss_ls = torch.norm((target - predict), p='fro') ** 2

        drug_reg = torch.trace(torch.mm(torch.mm(alpha1.T, drug_lap), alpha1))
        mic_reg = torch.trace(torch.mm(torch.mm(alpha2.T, mic_lap), alpha2))
        graph_reg =(2 ** (-3)) * drug_reg + (2 ** (-4)) * mic_reg

        loss_sum = loss_ls + graph_reg
        return {"loss_sum": loss_sum,
                "loss": loss_sum}





    def focal_loss_fn(self, predict, label, alpha, gamma):
        predict = predict.view(-1)
        label = label.view(-1)
        ce_loss = F.binary_cross_entropy(
            predict, label, reduction="none"
        )
        p_t = predict*label+(1-predict)*(1-label)
        loss = ce_loss*((1-p_t)**gamma)
        alpha_t = alpha * label + (1-alpha)*(1-label)
        focal_loss = (alpha_t * loss).mean()
        return {"loss_focal":focal_loss,
                "loss":focal_loss}

    def rank_loss_fn(self, predict, label, margin=0.8, reduction='mean'):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        pos_mask = label > 0
        pos = predict[pos_mask]
        neg = predict[~pos_mask]
        neg_mask = torch.randint(0, neg.shape[0], (pos.shape[0],), device=label.device)
        neg = neg[neg_mask]

        rank_loss = F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos),
                                          margin=margin, reduction=reduction)
        return {"loss_rank":rank_loss,
                "loss":rank_loss}

    def get_epoch_auroc_aupr(self, outputs):
        predict = [output["predict"].detach() for output in outputs]
        label = [output["label"] for output in outputs]
        predict = torch.cat(predict).cpu().view(-1)
        label = torch.cat(label).cpu().view(-1)
        aupr = metrics.average_precision_score(y_true=label, y_score=predict)
        auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
        return auroc, aupr

    def get_epoch_loss(self, outputs):
        loss_keys = [key for key in outputs[0] if key.startswith("loss")]
        loss_info = {key: [output[key].detach().cpu() for output in outputs if not torch.isnan(output[key])] for key in loss_keys}
        loss_info = {key: sum(value)/len(value) for key, value in loss_info.items()}
        return loss_info

    def training_epoch_end(self, outputs):
        stage = "train"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        self.log(f"{stage}/loss", loss_info["loss"], prog_bar=True)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        stage = "val"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        self.log(f"{stage}/loss", loss_info["loss"], prog_bar=True)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


def load_MDAD(root_dir="dataset/MKGCN_MicrobeDrugA/MDAD"):

    rr = np.loadtxt( 'dataset/MKGCN_MicrobeDrugA/MDAD/drugsimilarity.txt', dtype=np.float32,delimiter='\t')  # 数组(1373，1373）-1885129
    dd = np.loadtxt( 'dataset/MKGCN_MicrobeDrugA/MDAD/microbesimilarity.txt',dtype=np.float32, delimiter='\t')  # 数组（173,173）-29929
    adj_triple = np.loadtxt( 'dataset/MKGCN_MicrobeDrugA/MDAD/adj.txt',dtype=np.float32)  # 元组(2470,3)
    rd = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),  # 1373,173
                                    shape=(len(rr), len(dd))).toarray()
    dname = np.arange(173)
    rname = np.arange(1373)
    return {"drug": rr,
            "microbe": dd,  # 微生物
            "Wrname": rname,
            "Wdname": dname,
            "didr": rd.T}


def load_aBiofilm(root_dir="dataset/MKGCN_MicrobeDrugA/aBiofilm"):
    """drug:1720, MICROBES:140 association:2884
    """
    rr = np.loadtxt('dataset/MKGCN_MicrobeDrugA/aBiofilm/drugsimilarity.txt', dtype=np.float32,
                    delimiter='\t')  # 数组(1373，1373）-1885129
    dd = np.loadtxt('dataset/MKGCN_MicrobeDrugA/aBiofilm/microbesimilarity.txt', dtype=np.float32,
                    delimiter='\t')  # 数组（173,173）-29929
    adj_triple = np.loadtxt('dataset/MKGCN_MicrobeDrugA/aBiofilm/adj.txt', dtype=np.float32)  # 元组(2470,3)
    rd = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),  # 1373,173
                       shape=(len(rr), len(dd))).toarray()        # rr = read_data_txt(os.path.join(root_dir, "drug_features.txt"))

    dname = np.arange(140)
    rname = np.arange(1720)
    return {"drug":rr,
            "microbe":dd, #微生物
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}


def load_DrugVirus(root_dir="dataset/MKGCN_MicrobeDrugA/DrugVirus"):
    """drug:175, MICROBES:95 association:933
      """
    rr = np.loadtxt('dataset/MKGCN_MicrobeDrugA/DrugVirus/drugsimilarity.txt', dtype=np.float32,
                    delimiter='\t')
    dd = np.loadtxt('dataset/MKGCN_MicrobeDrugA/DrugVirus/microbesimilarity.txt', dtype=np.float32,
                    delimiter='\t')
    adj_triple = np.loadtxt('dataset/MKGCN_MicrobeDrugA/DrugVirus/adj.txt', dtype=np.float32)  # 元组(2470,3)
    rd = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                       shape=(len(rr), len(dd))).toarray()

    dname = np.arange(95)
    rname = np.arange(175)
    return {"drug": rr,
            "microbe": dd,  # 微生物
            "Wrname": rname,
            "Wdname": dname,
            "didr": rd.T}


class DRDataset():
    def __init__(self, dataset_name="MDAD", drug_neighbor_num=15, microbe_neighbor_num=15):
        assert dataset_name in ["aBiofilm","MDAD","DrugVirus","egatmda_dataset"]
        self.dataset_name = dataset_name
        if dataset_name=="aBiofilm":
            old_data=load_aBiofilm()
        elif dataset_name == "MDAD":
            old_data = load_MDAD()
        elif dataset_name == "DrugVirus":
            old_data = load_DrugVirus()

        else:
            old_data = scio.loadmat(f"dataset/{dataset_name}.mat")

        self.drug_sim = old_data["drug"].astype(np.float)   #药物相似
        self.microbe_sim = old_data["microbe"].astype(np.float) #微生物相似
        self.drug_name = old_data["Wrname"].reshape(-1)
        #self.drug_name_new = old_data["Wrname"].reshape(-1) #新加
        self.drug_num = len(self.drug_name)
        self.microbe_name = old_data["Wdname"].reshape(-1)
        self.microbe_num = len(self.microbe_name)
        self.interactions = old_data["didr"].T #转置

        self.drug_edge = self.build_graph(self.drug_sim, drug_neighbor_num)
        self.microbe_edge = self.build_graph(self.microbe_sim, microbe_neighbor_num)
        pos_num = self.interactions.sum()  #阳性样本个数
        neg_num = np.prod(self.interactions.shape) - pos_num  #阴性样本个数：np.prod所有维度的乘积-阳性样本的个数
        self.pos_weight = neg_num / pos_num
        print(f"dataset:{dataset_name}, drug:{self.drug_num}, microbe:{self.microbe_num}, pos weight:{self.pos_weight}")

    def build_graph(self, sim, num_neighbor):  #构建图
        if num_neighbor>sim.shape[0] or num_neighbor<0:#sim.shape[0]是相似性矩阵
            num_neighbor = sim.shape[0]   #本来取15个，但是如果和其相似的没有15个，那么有几个取几个
        neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]     #这里-sim是数组，首先我们要明白的是argpartition函数输出的是一个索引数组
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])  #.repeat重复数组中的元素  row行
        col_index = neighbor.reshape(-1)   #变成一维的了
        edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(int))
        values = torch.ones(edge_index.shape[1])
        values = torch.from_numpy(sim[row_index, col_index]).float()*values
        return (edge_index, values, sim.shape)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset config")
        parser.add_argument("--dataset_name", default="MDAD",
                                   choices=["aBiofilm","MDAD","DrugVirus","egatmda_dataset"])
        parser.add_argument("--drug_neighbor_num", default=15, type=int)
        parser.add_argument("--microbe_neighbor_num", default=15, type=int)
        return parent_parser


class HKFGCN(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):#添加模型特定的参数
        parser = parent_parser.add_argument_group("HKFGCN model config")#HKFGCN 模型配置
        parser.add_argument("--embedding_dim", default=128, type=int)
        parser.add_argument("--layer_num", default=3, type=int)
        parser.add_argument("--lr", type=float, default=0.01)#默认0.01
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--edge_dropout", default=0, type=float) #边的抛弃率
        parser.add_argument("--neighbor_num", type=int, default=15) #k=15
        parser.add_argument("--smooth", default=False, action="store_true")
        parser.add_argument("--gnn_mode", default="gcnt")
        return parent_parser

    def __init__(self, size_u, size_v,lambda1=2 ** (-3),lambda2=2 ** (-4),
                 act=nn.ReLU, dropout=0, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0, lr=0.01, layer_num=3,
                 pos_weight=1.0,
                 gnn_mode="gcnt", smooth=False, **kwargs):
        super(HKFGCN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u + size_v
        self.use_embedding = False
        self.in_dim = self.num_nodes  #输入维度等于节点数
        self.smooth = smooth

        cached = True if edge_dropout==0.0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num

        self.register_buffer("pos_weight", torch.tensor(pos_weight))  #一种是反向传播不需要被optimizer更新，称之为 buffer


        self.edge_dropout = EdgeDropout(keep_prob=1-edge_dropout)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)
        self.lambda1 =lambda1  # lambda   →     λ
        self.lambda2 =lambda2
        self.drug_l = []
        self.mic_l = []

        self.drug_kernels = []
        self.mic_kernels = []
        self.alpha1 = torch.randn(self.size_u,self.size_v)  # α #返回一个张量 drug_size*mic_size，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。
        self.alpha2 = torch.randn(self.size_v, self.size_u) #训练得到参数矩阵
        self.loss_fn = partial(self.getGipKernel_loss_fn)
        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,      #域内编码
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode="gcn") ]
        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,      #域间编码
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode="gcn") )

            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached,gnn_mode=gnn_mode) )
        self.intra_encoders = nn.ModuleList(intra_encoder)
        self.inter_encoders = nn.ModuleList(inter_encoder)
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num) #层注意力
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout)

        self.smooth_decoder = SmoothDecoder(size_u=size_u, size_v=size_v, k=kwargs["neighbor_num"])
        self.save_hyperparameters()  #保存参数

    def step(self, batch:FullGraphData):
        x = batch.embedding
        u_edge_index, u_edge_weight = batch.u_edge[:2]
        v_edge_index, v_edge_weight = batch.v_edge[:2]
        ur_edge_index, ur_edge_weight = batch.uv_edge[:2]
        vr_edge_index, vr_edge_weight = batch.vu_edge[:2]
        label = batch.label
        predict, u, v = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight)

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # torch.set_printoptions(profile="full")
        #
        # a, idx1 = torch.sort(predict[:,0], descending=True)  # descending为alse，升序，为True，降序
        # idx = idx1[:50]
        # print(idx)

        if self.smooth:
            predict, u, v = self.smooth_decoder(u, v, batch)

        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)].cpu()
            label = label[batch.valid_mask]

        ans = self.loss_fn(target=label, predict=predict, drug_lap=self.drug_l, mic_lap=self.mic_l, alpha1=self.alpha1, alpha2=self.alpha2)
        ans["predict"] = predict.reshape(-1)
        ans["label"] = label.reshape(-1)
        ad = DRDataset()
        self.alpha1=torch.mm(
            torch.mm((torch.mm(self.drug_kernels, self.drug_kernels) + self.lambda1 * self.drug_l).inverse(), self.drug_kernels),
            2 * torch.from_numpy(ad.interactions) - torch.mm(self.alpha2.T, self.mic_kernels.T)).detach()
        self.alpha2 = torch.mm(torch.mm((torch.mm(self.mic_kernels, self.mic_kernels) + self.lambda2 * self.mic_l).inverse(), self.mic_kernels),
                            2 * torch.from_numpy(ad.interactions).T - torch.mm(self.alpha1.T, self.drug_kernels.T)).detach()

        return ans  #返回预测的和标签


    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):
        ur_edge_index, ur_edge_weight = self.edge_dropout(ur_edge_index, ur_edge_weight)
        vr_edge_index, vr_edge_weight = self.edge_dropout(vr_edge_index, vr_edge_weight)

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        drug_kernels = [getGipKernel(short_embedding[:self.size_u].clone(), 0, 2 ** (-5), True)]
        mic_kernels = [getGipKernel(short_embedding[self.size_u:].clone(), 0, 2 ** (-5), True)]
        for inter_encoder, intra_encoder in zip(self.inter_encoders, self.intra_encoders):
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                          v_edge_index=v_edge_index, v_edge_weight=v_edge_weight)
            inter_feature = inter_encoder(x, u_edge_index=ur_edge_index, u_edge_weight=ur_edge_weight,
                                          v_edge_index=vr_edge_index, v_edge_weight=vr_edge_weight)
            x = intra_feature + inter_feature + layer_out[-1]
            drug_kernels.append(getGipKernel(x[:self.size_u].clone(), 0,2 ** (-2), True))
            mic_kernels.append(getGipKernel(x[self.size_u:].clone(), 0, 2 ** (-2), True))
            layer_out.append(x)

        x = torch.stack(layer_out[1:])
        drug_kernels=torch.stack(drug_kernels[1:])
        mic_kernels = torch.stack(mic_kernels[1:])

        attention = torch.softmax(self.attention, dim=0)
        drug_attention=torch.softmax(self.attention, dim=0).cpu()
        mic_attention = torch.softmax(self.attention, dim=0).cpu()
        x = torch.sum(x * attention, dim=0)

        # drug_kernels = torch.sum(drug_kernels * drug_attention, dim=0)
        # mic_kernels = torch.sum(mic_kernels * mic_attention, dim=0)
        drug_kernels = torch.sum(drug_kernels * drug_attention, dim=0)
        mic_kernels = torch.sum(mic_kernels * mic_attention, dim=0)
        # score, u, v = self.decoder(x)
        self.drug_kernels = normalized_kernel(drug_kernels)
        self.mic_kernels = normalized_kernel(mic_kernels)
        self.drug_l = laplacian(drug_kernels)
        self.mic_l = laplacian(mic_kernels)
        out1 =torch.mm(self.drug_kernels, self.alpha1)
        out2 =torch.mm(self.mic_kernels, self.alpha2)
        score=self.decoder(out1,out2)


        return score, drug_kernels, mic_kernels

    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]


