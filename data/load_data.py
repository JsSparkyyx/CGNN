import torch
import dgl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
from .data import GraphData

def load_dataset(dataset, device = 'cuda:0'):
    if dataset == 'reddit':
        g, features, labels = load_reddit()
    elif dataset == 'cora':
        g, features, labels = load_cora()
    elif dataset == 'amazon':
        g, features, labels = load_amazon()
    elif dataset == 'cfd':
        return load_cfd()
    else:
        raise(dataset + ' not supported.')
    data = split_tasks(g, features, labels, device)
    return data

def split_tasks(g, features, labels, device = 'cuda:0', num_task = 5, shuffle = True):
    num_cls = torch.unique(labels).size()[0]
    cls_idx = np.arange(num_cls)
    if shuffle:
        random.shuffle(cls_idx)
    cls_idx = torch.LongTensor(cls_idx)
    data = []
    taskcla = []
    for task in range(num_task):
        selected_cls = cls_idx[task*(num_cls//num_task):(task+1)*(num_cls//num_task)]
        condition = torch.BoolTensor([i in selected_cls for i in labels])
        selected_mask = torch.where(condition, torch.tensor(1), torch.tensor(0)).bool()
        sub_g = dgl.node_subgraph(g,selected_mask, store_ids = False)
        sub_feat = features[selected_mask]
        sub_cls = labels[selected_mask]
        sub_g = dgl.add_self_loop(sub_g)
        train_mask, val_mask, test_mask = generate_mask(sub_cls)
        le = LabelEncoder()
        le.fit(sub_cls.numpy())
        encoded_cls = torch.LongTensor(le.transform(sub_cls))
        data.append(GraphData(sub_g, sub_feat, sub_cls, encoded_cls, train_mask, val_mask, test_mask, device))
        taskcla.append((task,selected_cls.size()[0],sub_feat[train_mask].size()[0]))
    return data, taskcla, features.size()[1]

def generate_mask(labels, split = [0.3,0.1,0.6]):
    idx = np.arange(labels.size()[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=split[0]+split[1],
                                                   test_size=split[2],
                                                   stratify=labels)
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(split[0] / (split[0]+split[1])),
                                          test_size=1 - (split[0] / (split[0]+split[1])),
                                          stratify=labels[idx_train_and_val])
    train_mask = torch.zeros((labels.size()[0], ), dtype=torch.bool)
    val_mask = torch.zeros((labels.size()[0], ), dtype=torch.bool)
    test_mask = torch.zeros((labels.size()[0], ), dtype=torch.bool)
    train_mask[idx_train] = 1
    val_mask[idx_val] = 1
    test_mask[idx_test] = 1
    return train_mask, val_mask, test_mask

def load_cora():
    from dgl.data import CoraFullDataset
    g = CoraFullDataset()[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.pop('label')
    g.ndata.pop('feat')
    
    return g, features, labels

def load_amazon():
    from dgl.data import AmazonCoBuyComputerDataset
    g = AmazonCoBuyComputerDataset()[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.pop('label')
    g.ndata.pop('feat')
    
    return g, features, labels

def load_reddit():
    from dgl.data import RedditDataset
    g = RedditDataset()[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.pop('label')
    g.ndata.pop('feat')
    
    return g, features, labels

def load_cfd(device = 'cuda:0', shuffle = True):
    import pickle
    path = './datasets/cfd'
    regions = pickle.load(open(path+'/data.pickle','rb'))
    data = []
    indexes = list(regions.keys())
    # print(regions.keys())
    if shuffle:
        random.shuffle(indexes)
    print(indexes)
    for task, index in enumerate(indexes):
        region = regions[index]
        card2idx = {}
        for i, name in enumerate(region['cc_num'].unique()):
            card2idx[name] = i
        merch2idx = {}
        for i, name in enumerate(region['merchant'].unique()):
            merch2idx[name] = i
            
        g = dgl.heterograph({
            ('transaction', 'tc', 'card'): (np.array(region['trans_num']), np.array([card2idx[i] for i in region['cc_num']])),
            ('card', 'ct', 'transaction'): (np.array([card2idx[i] for i in region['cc_num']]), np.array(region['trans_num'])),
            ('transaction', 'tm', 'merchant'): (np.array(region['trans_num']), np.array(region['merchant_id'])),
            ('merchant', 'mt', 'transaction'): (np.array(region['merchant_id']), np.array(region['trans_num'])),
            ('transaction', 'th', 'hour'): (np.array(region['trans_num']), np.array(region['trans_date_trans_time'])),
            ('hour', 'ht', 'transaction'): (np.array(region['trans_date_trans_time']), np.array(region['trans_num']))
        })
        
        transaction = region[['trans_num', 'trans_date_trans_time', 'amt',
                                'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'category']]
        transaction_dummies = pd.get_dummies(transaction.drop_duplicates(
        )[['trans_date_trans_time', 'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'category']])
        features = np.array(transaction_dummies)
        features -= np.mean(features, axis=0)
        features /= np.std(features, axis=0)
        features = torch.FloatTensor(features)
        labels = np.array(region['is_fraud'])
        labels = torch.LongTensor(labels)

        def get_binary_mask(total_size, indices):
            mask = torch.zeros(total_size)
            mask[indices] = 1
            return mask.bool()
        label_ids = [0, 1]
        float_mask = np.zeros(len(labels))
        for label_id in label_ids:
            label_mask = (labels == label_id)
            float_mask[label_mask] = np.random.permutation(np.linspace(0, 1, label_mask.sum()))
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]
        num_nodes = labels.shape[0]
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)

        train_idx = torch.nonzero(train_mask)
        val_idx = torch.nonzero(val_mask)
        test_idx = torch.nonzero(test_mask)
        num_nodes = labels.shape[0]
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)
        data.append(GraphData(g, features, labels, labels, train_mask, val_mask, test_mask, device))
    return (data, [['tc', 'ct'], ['th', 'ht'], ['tm', 'mt']], indexes), 2, features.shape[1]
        
# g, features, labels = load_cora()
# data, taskcla, in_feat = split_tasks(g, features, labels)
# (seed_nodes, output_nodes, blocks), sub_g = data[0].class_sampling(1)
# g = data[0].g
# features = data[0].features
# labels = data[0].encoded_labels
# print(g)
# print(seed_nodes)
# print(blocks[0].srcdata[dgl.NID])
# print(sub_g.nodes())
# print(sub_g.srcdata[dgl.NID])
# print(labels[sub_g.srcdata[dgl.NID][seed_nodes]])
# print(labels[sub_g.srcdata[dgl.NID][output_nodes]])
# print(data)
# print(taskcla)
# print(in_feat)
# print(torch.FloatTensor(3, 3))