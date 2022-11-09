import torch
import random
import numpy as np
import pandas as pd
import os

def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)
    # print(gw_real)
    # print(gw_syn)
    args.dis_metric = 'ours'
    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    else:
        exit('DC error: unknown distance function')
        
    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_results(results,args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path,args.dataset)):
        os.makedirs(os.path.join(args.save_path,args.dataset))
    if not os.path.exists(os.path.join(args.save_path,args.dataset,'detail')):
        os.makedirs(os.path.join(args.save_path,args.dataset,'detail'))
    path = os.path.join(args.save_path,args.dataset,'detail') + '/' + args.arch+'_'+args.method+'_'+str(args.num_tasks)+'_'+ args.manner +'_'+str(args.seed) + '.csv'
    results.to_csv(path,index=False)
    LA = 0
    FM = 0
    for task in range(args.num_tasks):
        LA += float(results[(results['stage'] == task) & (results['task'] == task)]['accuracy'])
        if task != args.num_tasks - 1:
            FM += results[results['stage'] == args.num_tasks - 1]['accuracy'].max() - float(results[(results['stage'] == args.num_tasks - 1) & (results['task'] == task)]['accuracy'])
    ACC = results[results['stage'] == args.num_tasks - 1]['accuracy'].mean()
    LA = LA/args.num_tasks
    FM = FM/(args.num_tasks - 1)
    path = os.path.join(args.save_path,args.dataset) + '/' + args.arch+'_'+args.method+'_'+str(args.num_tasks)+'_'+ args.manner +'_overall' + '.csv'
    with open(path, 'a') as f:
        f.write("{:.2f},{:.2f},{:.2f},{}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed))
    print("{:.2f},{:.2f},{:.2f},{}\n".format(round(ACC,2),round(FM,2),round(LA,2),args.seed))



import os.path as osp
import scipy.sparse as sp
import torch_geometric.transforms as T
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from deeprobust.graph.utils import *
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.datasets import CoraFull, Reddit
import random

def load_cora(name, normalize_features=False, transform=None, if_dpr=True, shuffle = True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = CoraFull(path)

    dataset = dataset[0]
    labels = dataset.y
    features = dataset.x
    scaler = StandardScaler()
    scaler.fit(features)
    features = torch.tensor(scaler.transform(features))
    num_cls = np.unique(labels).shape[0]
    cls_idx = np.arange(num_cls)
    if shuffle:
        random.shuffle(cls_idx)
    cls_idx = torch.LongTensor(cls_idx)
    pyg_data = []
    dpr_data = []
    taskcla = []
    num_task = 5
    for task in range(num_task):
        selected_cls = cls_idx[task*(num_cls//num_task):(task+1)*(num_cls//num_task)]
        condition = torch.BoolTensor([i in selected_cls for i in labels])
        selected_mask = torch.where(condition, torch.tensor(1), torch.tensor(0)).bool()
        edge_index, _ = subgraph(selected_mask, dataset.edge_index, relabel_nodes = True)
        le = LabelEncoder()
        le.fit(labels[selected_mask])
        encoded_cls = torch.LongTensor(le.transform(labels[selected_mask]))
        data = Data(x = features[selected_mask], edge_index = edge_index, y = encoded_cls)
        pyg_data.append(data)
        dpr_data.append(Pyg2Dpr(data))
        taskcla.append((task,selected_cls.size()[0]))
    return dpr_data, pyg_data, taskcla

def load_reddit(name, normalize_features=True, transform=None, if_dpr=True, shuffle = True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = Reddit(path)

    dataset = dataset[0]
    labels = dataset.y
    features = dataset.x
    scaler = StandardScaler()
    scaler.fit(features)
    features = torch.tensor(scaler.transform(features))
    num_cls = np.unique(labels).shape[0]
    cls_idx = np.arange(num_cls)
    if shuffle:
        random.shuffle(cls_idx)
    cls_idx = torch.LongTensor(cls_idx)
    pyg_data = []
    dpr_data = []
    taskcla = []
    num_task = 5
    for task in range(num_task):
        selected_cls = cls_idx[task*(num_cls//num_task):(task+1)*(num_cls//num_task)]
        condition = torch.BoolTensor([i in selected_cls for i in labels])
        selected_mask = torch.where(condition, torch.tensor(1), torch.tensor(0)).bool()
        edge_index, _ = subgraph(selected_mask, dataset.edge_index, relabel_nodes = True)
        # print(selected_mask)
        # print(selected_mask.nonzero().flatten().size())
        # print(edge_index)
        le = LabelEncoder()
        le.fit(labels[selected_mask])
        encoded_cls = torch.LongTensor(le.transform(labels[selected_mask]))
        # print(encoded_cls)
        # print(encoded_cls.size())
        # print(features[selected_mask].size())
        data = Data(x = features[selected_mask], edge_index = edge_index, y = encoded_cls)
        pyg_data.append(data)
        dpr_data.append(Pyg2Dpr(data))
        taskcla.append((task,selected_cls.size()[0]))
    return dpr_data, pyg_data, taskcla

class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        # dataset_name = pyg_data.name
        dataset_name = ''
        # pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv': # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))

        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.3, stratify=self.labels)


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask
