import datetime
import os
import time
import argparse
import scipy as sp
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import metrics
import scipy.io as scio
from model import HKFGCN
from . import DATA_TYPE_REGISTRY
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler


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


class GraphDataIterator(DataLoader): #图数据迭代器
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", batch_size=1024*5, shuffle=False,
                 dataset_type="FullGraphDataset", **kwargs):
        # assert dataset_type in ["FullGraphDataset", "PairGraphDataset"]
        dataset_cls = DATA_TYPE_REGISTRY.get(dataset_type)
        dataset = dataset_cls(dataset, mask, fill_unkown, stage=stage, **kwargs)
        if len(dataset)<batch_size:
            # logging.info(f"dataset size:{len(dataset)}, batch_size:{batch_size} is invalid!")
            batch_size = min(len(dataset), batch_size)
        if shuffle and stage=="train":
            sampler = RandomSampler(dataset) #RandomSampler随机样本的元素。如果没有替换，则从打乱的数据集取样。如果需要替换，则用户可以指定要绘制的num_samples。
        else:
            sampler = SequentialSampler(dataset)#SequentialSampler按顺序采样元素，总是按照相同的顺序 按顺序对数据集采样
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)  #也就是说BatchSampler的作用就是将前面的Sampler采样得到的索引值进行合并，当数量等于一个batch大小后就将这一批的索引值返回。
        super(GraphDataIterator, self).__init__(dataset=dataset, batch_size=None, sampler=batch_sampler,
                                            collate_fn=Dataset.collate_fn, **kwargs)


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


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)  #len（）字符串长度
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)  #mat可以从字符串或列表中生成；array只能从列表中生成
    thresholds_num = thresholds.shape[1] # shape[0]：表示矩阵的行数，shape[1]表示矩阵的列数

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1)) #第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    negative_index = np.where(predict_score_matrix < thresholds.T)  #预测为阴性的样本     #只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标
    positive_index = np.where(predict_score_matrix >= thresholds.T)   #预测为阳性的样本
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    # tnr = TN/(FP+TN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    # # plt.plot(mean_recall, mean_precision)
    # plt.show()
    # precision = 1
    # recall = 23
    # ResultFile = open("base_test_3.csv", "a")  # 创建csv文件
    # writer = csv.writer(ResultFile)  # 创
    # writer.writerow([fpr, tpr])
    # ResultFile.close()


    recall_list = tpr
    precision_list = TP/(TP+FP)
    # PR_dot_matrix = np.mat(sorted(np.column_stack(
    #     (recall_list, precision_list)).tolist())).T
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T

    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1,0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    # plt.plot(recall, precision)
    # plt.show()
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def evaluate(predict, label, is_final=False):
    if not is_final:
        res = get_metrics(real_score=label, predict_score=predict)
    else:
        res = [None]*7
    aupr = metrics.average_precision_score(y_true=label, y_score=predict)

    auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
    # f1_score=metrics.f1_score(y_true=label, y_score=predict)
    # precision, recall, thresholds = precision_recall_curve(label, predict)
    # acc=accuracy_score(y_true=label, y_score=predict)
    # res = get_metrics(real_score=label, predict_score=predict)
    # result = {"aupr": aupr,
    #           "auroc": auroc,
    #           "f1_score": f1_score,
    #           "accuracy": acc,
    #           "recall": recall,
    #           "precision": precision,
    #
    #           "lagcn_aupr": res[0],
    #           "lagcn_auc":res[1],
    #           "lagcn_f1_score":res[2],
    #           "lagcn_accuracy":res[3],
    #           "lagcn_recall":res[4],
    #           "lagcn_specificity":res[5],
    #           "lagcn_precision":res[6]}


    res = get_metrics(real_score=label, predict_score=predict)
    result = {"aupr":aupr,
              "auroc":auroc,
              "f1_score":res[2],
              "accuracy":res[3],
              "recall":res[4],
              "specificity":res[5],
              "precision":res[6]}

    return result


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


class CVDataset(pl.LightningDataModule):  #用來交叉驗證
    """use for cross validation
       split_mode | n_splits |  drug_id   |   microbe_id | description
       global     |   1      |   *        |     *        | case study
       global     |  10      |   *        |     *        | 10 fold
       local      |  -1      |   not None |     *        | local leave one for remove drug
       local      |  -1      |   None     |     not None | local leave one for remove microbe
       local      |   1      |   int      |     *        | local leave one for remove specific drug
       local      |   1      |   None     |     int      | local leave one for remove specific drug
    """
    def __init__(self, dataset, split_mode="global", n_splits=5,
                 drug_idx=None, microbe_idx=None, global_test_all_zero=False,
                 train_fill_unknown=True, seed=666, cached_dir="cached",
                 dataset_type="FullGraphDataset",
                 **kwargs):
        super(CVDataset, self).__init__()
        self.dataset = dataset
        self.split_mode = split_mode
        self.n_splits = n_splits
        self.global_test_all_zero = global_test_all_zero
        self.train_fill_unknown = train_fill_unknown
        self.seed = seed
        self.row_idx = drug_idx  #行
        self.col_idx = microbe_idx  #列
        self.dataset_type = dataset_type
        self.save_dir = os.path.join(cached_dir, dataset.dataset_name,#路径拼接Path20 = os.path.join(Path1,Path2,Path3)   Path20 = home\develop\code
                                     f"{self.split_mode}_{len(self)}_split_{self.row_idx}_{self.col_idx}")
        assert isinstance(n_splits, int) and n_splits>=-1

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("cross validation config")
        parser.add_argument("--split_mode", default="global", choices=["global", "local"])
        parser.add_argument("--n_splits", default=5, type=int)
        parser.add_argument("--drug_idx", default=None, type=int)
        parser.add_argument("--microbe_idx", default=None, type=int)
        parser.add_argument("--global_test_all_zero", default=False, action="store_true", help="全局模式每折测试集是否测试所有未验证关联，默认：不测试")
        parser.add_argument("--train_fill_unknown", default=True, action="store_true", help="训练集中是否将测试集关联填0还是丢弃，默认：丢弃")
        parser.add_argument("--dataset_type", default=None, choices=["FullGraphDataset", "PairGraphDataset"])
        parser.add_argument("--seed", default=666, type=int)
        return parent_parser

    def fold_mask_iterator(self, interactions, mode="global", n_splits=5, row_idx=None, col_idx=None, global_test_all_zero=False, seed=666):
        assert mode in ["global", "local"]
        assert n_splits>=-1 and isinstance(n_splits, int)
        if mode=="global":
            if n_splits==1:
                mask = np.ones_like(interactions, dtype="bool")
                yield mask, mask
            else:
                kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                pos_row, pos_col = np.nonzero(interactions)
                neg_row, neg_col = np.nonzero(1 - interactions)
                assert len(pos_row) + len(neg_row) == np.prod(interactions.shape)
                for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                        kfold.split(neg_row)):
                    train_mask = np.zeros_like(interactions, dtype="bool")
                    test_mask = np.zeros_like(interactions, dtype="bool")
                    if global_test_all_zero:
                        test_neg_idx = np.arange(len(neg_row))
                    train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
                    train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
                    test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
                    test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
                    train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
                    test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
                    train_mask[train_edge[0], train_edge[1]] = True
                    test_mask[test_edge[0], test_edge[1]] = True
                    yield train_mask, test_mask
        elif mode=="local":
            if row_idx is not None:
                row_idxs = list(range(interactions.shape[0])) if n_splits==-1 else [row_idx]
                for idx in row_idxs:
                    yield self.get_fold_local_mask(interactions, row_idx=idx)
            elif col_idx is not None:
                col_idxs = list(range(interactions.shape[1])) if n_splits==-1 else [col_idx]
                for idx in col_idxs:
                    yield self.get_fold_local_mask(interactions, col_idx=idx)
        else:
            raise NotImplemented

    def get_fold_local_mask(self, interactions, row_idx=None, col_idx=None):
        train_mask = np.ones_like(interactions, dtype="bool")
        test_mask = np.zeros_like(interactions, dtype="bool")
        if row_idx is not None:
            train_mask[row_idx, :] = False
            test_mask[np.ones(interactions.shape[1], dtype="int")*row_idx,
                      np.arange(interactions.shape[1])] = True
        elif col_idx is not None:
            train_mask[:,col_idx] = False
            test_mask[np.arange(interactions.shape[0]),
                      np.ones(interactions.shape[0], dtype="int") * col_idx] = True
        return train_mask, test_mask

    def prepare_data(self):
        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        import glob
        if len(glob.glob(os.path.join(save_dir, "split_*.mat")))!=len(self):
            for i, (train_mask, test_mask) in enumerate(self.fold_mask_iterator(interactions=self.dataset.interactions,
                                                                 mode=self.split_mode,
                                                                 n_splits=self.n_splits,
                                                                 global_test_all_zero=self.global_test_all_zero,
                                                                 row_idx=self.row_idx,
                                                                 col_idx=self.col_idx)):
                scio.savemat(os.path.join(save_dir, f"split_{i}.mat"),
                             {"train_mask":train_mask,
                              "test_mask":test_mask},
                             )

        data = scio.loadmat(os.path.join(self.save_dir, f"split_{self.fold_id}.mat"))
        self.train_mask = data["train_mask"]
        self.test_mask = data["test_mask"]

    def train_dataloader(self):
        return GraphDataIterator(self.dataset, self.train_mask, fill_unkown=self.train_fill_unknown,
                                 stage="train", dataset_type=self.dataset_type)

    def val_dataloader(self):
        return GraphDataIterator(self.dataset, self.test_mask, fill_unkown=True,
                                 stage="val", dataset_type=self.dataset_type)

    def __iter__(self):
        for fold_id in range(len(self)):
            self.fold_id = fold_id
            yield self

    def __len__(self):
        if self.split_mode=="global":
            return self.n_splits
        elif self.split_mode=="local":
            if self.n_splits==-1:
                if self.row_idx is not None:
                    return self.dataset.interactions.shape[0]
                elif self.col_idx is not None:
                    return self.dataset.interactions.shape[1]
            else:
                return 1

@torch.no_grad()
def train_test_fn(model, train_loader, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels,  edges = [], [], []
    for batch in train_loader:
        model.train_step(batch)
    for batch in val_loader:
        batch = move_data_to_device(batch, device) #将数据集合传输到给定设备。任何定义设备方法的对象将被移动，而集合中的所有其他对象将保持不变
        output = model.test_step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    # logger.info(f"eval time cost: {eval_end_time_stamp - eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"],auc=metric["lagcn_auc"],f1_score=metric["lagcn_f1_score"],acc=metric["lagcn_accuracy"],recall=metric["lagcn_recall"],specificity=["lagcn_specificity"],precision=["lagcn_precision"])
        scio.savemat(save_file, {"row": edges[0],
                                 "col": edges[1],
                                 "score": scores,
                                 "label": labels,
                                 })
        # logger.info(f"save time cost: {time.time() - eval_end_time_stamp}")
    return scores, labels, edges, metric

@torch.no_grad()
def test_fn(model, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in val_loader:
        batch = move_data_to_device(batch, device) #将一组数据传输到给定的设备。任何为(设备)定义方法的对象都将被移动，集合中的所有其他对象将保持不变。
        output = model.step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    # logger.info(f"eval time cost: {eval_end_time_stamp-eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                      "col": edges[1],
                      "score": scores,
                      "label": labels,
                      })
        # logger.info(f"save time cost: {time.time()-eval_end_time_stamp}")
    return scores, labels, edges, metric


def train_fn(config, model, train_loader, val_loader):
    checkpoint_callback = ModelCheckpoint(monitor="val/auroc",  #该回调函数将在每个epoch后保存模型到filepath  monitor后面写的是用auroc
                                          mode="max",   #当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min
                                          save_top_k=1,
                                          verbose=False,
                                          save_last=True)
    lr_callback = pl.callbacks.LearningRateMonitor("epoch")
    trainer = Trainer(max_epochs=config.epochs,
                      default_root_dir=config.log_dir,
                      profiler=config.profiler,
                      fast_dev_run=False,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_callback],
                      # gpus=config.gpus,
                      check_val_every_n_epoch=1
                      )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    if not hasattr(config, "dirpath"):
        config.dirpath = trainer.checkpoint_callback.dirpath
    # checkpoint and add path
    # checkpoint = torch.load("lightning_logs/version_7/checkpoints/epoch=85.ckpt")
    # trainer.on_load_checkpoint(checkpoint)
    print(model.device)


def train(config, model_cls=HKFGCN):
    time_stamp = time.asctime()

    datasets = DRDataset(dataset_name=config.dataset_name, drug_neighbor_num=config.drug_neighbor_num,
                         microbe_neighbor_num=config.microbe_neighbor_num)
    log_dir = os.path.join(f"{config.comment}", f"{config.split_mode}-{config.n_splits}-fold", f"{config.dataset_name}",
                           f"{config.seed}",f"{model_cls.__name__}", f"{time_stamp.replace(':', '-')}")
    #log_dir=os.path.join(args.checkpoint, datetime.datetime.now().isoformat().replace(':', '-'))
    config.log_dir = log_dir
    config.n_drug = datasets.drug_num
    config.n_microbe = datasets.microbe_num

    config.size_u = datasets.drug_num
    config.size_v = datasets.microbe_num

    # config.gpus = 1 if torch.cuda.is_available() else 0
    config.pos_weight = datasets.pos_weight

    config.time_stamp = time_stamp
    # logger = init_logger(log_dir)
    # logger.info(pformat(vars(config)))
    config.dataset_type = config.dataset_dype if config.dataset_type is not None else model_cls.DATASET_TYPE
    cv_spliter = CVDataset(datasets, split_mode=config.split_mode, n_splits=config.n_splits,
                           drug_idx=config.drug_idx, microbe_idx=config.microbe_idx,
                           train_fill_unknown=config.train_fill_unknown,
                           global_test_all_zero=config.global_test_all_zero, seed=config.seed,
                           dataset_type=config.dataset_type)
    pl.seed_everything(config.seed)
    scores, labels, edges, split_idxs = [], [], [], []
    metrics = {}
    start_time_stamp = time.time()
    for split_id, datamodule in enumerate(cv_spliter):
        # if split_id not in [4, 5]:
        #     continue
        config.split_id = split_id
        split_start_time_stamp = time.time()

        datamodule.prepare_data()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        config.pos_weight = train_loader.dataset.pos_weight




        model = model_cls(**vars(config))
        # model = model.cpu() if config.gpus else model


        # if split_id==0:
            # logger.info(model)
        # logger.info(f"begin train fold {split_id}/{len(cv_spliter)}")
        train_fn(config, model, train_loader=train_loader, val_loader=val_loader)
        # logger.info(f"end train fold {split_id}/{len(cv_spliter)}")
        save_file_format = os.path.join(config.log_dir,
                                        f"{config.dataset_name}-{config.split_id} fold-{{auroc}}-{{aupr}}.mat")
        score, label, edge, metric = test_fn(model, val_loader, save_file_format)
        # score, label, edge, metric = train_test_fn(model, train_loader, val_loader, save_file_format)

        metrics[f"{split_id}"] = metric
        scores.append(score)
        labels.append(label)
        edges.append(edge)
        split_idxs.append(np.ones(len(score), dtype=int)*split_id)
        # logger.info(f"{split_id}/{len(cv_spliter)} folds: {metric}")
        # logger.info(f"{split_id}/{len(cv_spliter)} folds time cost: {time.time()-split_start_time_stamp}")
        if config.debug:
            break
    end_time_stamp = time.time()
    # logger.info(f"total time cost:{end_time_stamp-start_time_stamp}")
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    edges = np.concatenate(edges, axis=1)
    split_idxs = np.concatenate(split_idxs, axis=0)
    final_metric = evaluate(predict=scores, label=labels, is_final=True)
    np.save("E:\PLOT_ROC_AUPR\label.npz",labels)
    np.save("E:\PLOT_ROC_AUPR\predict_HKFGCN.npz",scores)
    metrics["final"] = final_metric
    metrics = pd.DataFrame(metrics).T
    metrics.index.name = "split_id"
    metrics["seed"] = config.seed
    # logger.info(f"final {config.dataset_name}-{config.split_mode}-{config.n_splits}-fold-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}")
    output_file_name = f"final-{config.dataset_name}-{config.split_mode}-{config.n_splits}-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}-fold"
    scio.savemat(os.path.join(log_dir, f"{output_file_name}.mat"),
                 {"row": edges[0],
                  "col": edges[1],
                  "score": scores,
                  "label": labels,
                  "split_idx":split_idxs}
                 )
    with pd.ExcelWriter(os.path.join(log_dir, f"{output_file_name}.xlsx")) as f:
        metrics.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        for key, value in final_metric.items():
            params[key] = value
        params["file"] = output_file_name
        params.to_excel(f, sheet_name="params")

    # logger.info(f"save final results to r'{os.path.join(log_dir, output_file_name)}.mat'")
    # logger.info(f"final results: {final_metric}")


def parse(print_help=False):
    parser = argparse.ArgumentParser() #创建 ArgumentParser() 对象,使用 argparse 的第一步是创建一个 ArgumentParser 对象。ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser.add_argument("--model", default="HKFGCN", type=str)   #调用 add_argument() 方法添加参数
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profiler", default=False, type=str)
    parser.add_argument("--comment", default="runs", type=str, help="experiment name")
    parser = DRDataset.add_argparse_args(parser)
    parser = CVDataset.add_argparse_args(parser)
    parser = HKFGCN.add_model_specific_args(parser)
    args = parser.parse_args()   #ArgumentParser 通过 parse_args() 方法解析参数。
    if print_help:
        parser.print_help()
    return args

def pathIterator(comment):
    split_modes = sorted(os.listdir(comment))
    for split_mode in split_modes:
        for dataset in os.listdir(os.path.join(comment, split_mode)):
            for model in os.listdir(os.path.join(comment, split_mode, dataset)):
                for seed in os.listdir(os.path.join(comment, split_mode, dataset, model)):
                    for time_stamp in os.listdir(os.path.join(comment, split_mode, dataset, model, seed)):
                        yield comment, split_mode, dataset, model, seed, time_stamp

def report(comment, collect_parma=True):
    metrics = []
    params = []
    for comment, split_mode, dataset, model, seed, time_stamp in pathIterator(comment):
        dataset_dir = os.path.join(comment, split_mode, dataset, model, seed, time_stamp)
        valid_files = sorted([ file for file in os.listdir(dataset_dir) if file.endswith(".xlsx") and file.startswith("final")])
        for file in tqdm(valid_files):
            with pd.ExcelFile(os.path.join(dataset_dir, file)) as reader:
                metric = reader.parse(reader.sheet_names[0], index_col=0)
                if collect_parma:
                    param = reader.parse("params", index_col=0)
                    params.append(param)
                    metric["index"] = param.index[0]
                    metric["seed"] = param["seed"].iloc[0]
                metric["model"] = model
                metric["split_mode"] = split_mode
                metric["dataset"] = dataset
                metric["comment"] = comment
                metrics.append(metric)
    metrics = pd.concat(metrics)
    metrics.index.name = "split_id"
    if collect_parma:
        params = pd.concat(params)
    with pd.ExcelWriter(f"{comment}.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="metrics")
        if collect_parma:
            params.to_excel(writer, sheet_name="params")


if __name__=="__main__":

    args = parse(print_help=True)
    train(args)
