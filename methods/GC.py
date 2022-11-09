import torch
import torch.nn.functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
from models.PGE import PGE
import importlib
from utils import match_loss
import numpy as np

class GCManager:
    def __init__(self, in_feat, taskcla, args):
        self.args = args
        self.current_task = 0
        self.taskcla = taskcla
        self.in_feat = in_feat
        self.pge = torch.nn.ModuleList()
        self.feat_syn = []
        self.num_class_dict = []
        self.syn_class_indices = []
        self.labels_syn = []
        self.optimizer_feat = []
        self.optimizer_pge = []
        self.ce = torch.nn.CrossEntropyLoss()
        for task, n_class, num_node in taskcla:
            feat_syn = torch.nn.Parameter(torch.FloatTensor(int(num_node*args.reduction_rate), in_feat).to(self.args.device))
            feat_syn.data.copy_(torch.randn(feat_syn.size()))
            pge = PGE(in_feat, int(num_node*args.reduction_rate), device = self.args.device).to(self.args.device)
            optimizer_feat = torch.optim.Adam([feat_syn], lr=args.lr_cr)
            optimizer_pge = torch.optim.Adam(pge.parameters(), lr=args.lr_cr)
            self.feat_syn.append(feat_syn)
            self.pge.append(pge)
            self.optimizer_feat.append(optimizer_feat)
            self.optimizer_pge.append(optimizer_pge)

    def generate_label(self, labels):
        from collections import Counter
        labels = labels.cpu().numpy()
        counter = Counter(labels)
        num_class_dict = {}
        n = len(labels)
        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
        labels_syn = torch.LongTensor(labels_syn).to(self.args.device)
        self.syn_class_indices.append(syn_class_indices)
        self.num_class_dict.append(num_class_dict)
        self.labels_syn.append(labels_syn)
        return labels_syn, syn_class_indices

    def get_sub_adj_feat(self, data, features, labels_syn, train_mask):
        idx_selected = []

        from collections import Counter;
        counter = Counter(labels_syn.cpu().numpy())
        for c in counter.keys():
            seed_nodes, output_nodes, blocks = data.class_sampling(c, batch_size = counter[c])
            tmp = list(output_nodes.cpu().numpy())
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        adj_knn = None
        return features, adj_knn    

    def evaluation_cr(self, data, task):
        args = self.args
        in_feat = self.in_feat
        _, n_class, _ = self.taskcla[task]
        feat_syn, pge, labels_syn = self.feat_syn[task].detach(), self.pge[task], self.labels_syn[task]

        arc = importlib.import_module(f'models.{args.arch}')
        if args.arch == 'GAT':
            model = arc.NET(in_feat, args.gat_hidden, heads = args.gat_head, num_layers = args.gat_layer, n_class = n_class).to(args.device)
        elif args.arch == 'SAGE':
            model = arc.NET(in_feat, args.sage_hidden, n_layers = args.sage_layer, aggregator_type = args.aggregator_type).to(args.device)
        elif args.arch == 'GCN':
            model = arc.NET(in_feat, args.gcn_hidden, n_layers = args.gcn_layer).to(args.device)
        g_syn, edge_weight = pge.inference(feat_syn)
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_cr)

        model.train()
        for epoch in range(args.epochs):
            model.zero_grad()
            logits = model(g_syn, feat_syn, edge_weight = edge_weight)
            loss = self.ce(logits, labels_syn)
            loss.backward()
            optimizer_model.step()

        model.eval()
        g, features, _, labels, train_mask, val_mask, test_mask = data.retrieve_data()
        logits = model(g, features)
        prob, prediction = torch.max(logits, dim=1)
        prediction = prediction[test_mask].cpu().numpy()
        labels = labels[test_mask].cpu().numpy()

        acc = accuracy_score(labels, prediction)
        mif1 = f1_score(labels, prediction, average='micro')
        maf1 = f1_score(labels, prediction, average='macro')
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)

    def train_task(self, data, task):
        args = self.args
        in_feat = self.in_feat
        g, features, _, labels, train_mask, val_mask, test_mask = data.retrieve_data()
        labels_syn, syn_class_indices = self.generate_label(labels[train_mask])
        _, n_class, _ = self.taskcla[task]
        pge = self.pge[task]
        self.feat_syn[task].data.copy_(self.get_sub_adj_feat(data, features, labels_syn, train_mask)[0])
        feat_syn = self.feat_syn[task]
        
        for epoch in trange(args.cr_epoch, leave=False):
            arc = importlib.import_module(f'models.{args.arch}')
            if args.arch == 'GAT':
                model = arc.NET(in_feat, args.gat_hidden, heads = args.gat_head, num_layers = args.gat_layer, n_class = n_class).to(args.device)
            elif args.arch == 'SAGE':
                model = arc.NET(in_feat, args.sage_hidden, n_layers = args.sage_layer, aggregator_type = args.aggregator_type).to(args.device)
            elif args.arch == 'GCN':
                model = arc.NET(in_feat, args.gcn_hidden, n_layers = args.gcn_layer).to(args.device)
            optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_cr)
            optimizer_feat = self.optimizer_feat[task]
            optimizer_pge = self.optimizer_pge[task]

            model.train()
            loss_avg = 0
            for _ in range(args.data_epoch):
                g_syn, edge_weight = pge(feat_syn)

                loss = torch.tensor(0.0).to(args.device)
                for cls in range(n_class):
                    seed_nodes, output_nodes, blocks = data.class_sampling(cls)
                    logits = model(blocks, features[seed_nodes], mini_batch = True)
                    loss_real = F.nll_loss(logits, labels[output_nodes])
                    gw_real = torch.autograd.grad(loss_real, list(model.parameters()))
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    logits_syn = model(g_syn, feat_syn, edge_weight = edge_weight)
                    ind = syn_class_indices[cls]
                    loss_syn = F.nll_loss(logits_syn[ind[0]: ind[1]], labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, list(model.parameters()), create_graph=True)
                    coeff = self.num_class_dict[task][cls] / max(self.num_class_dict[task].values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=args.device)
                loss_avg += loss.item()

                optimizer_feat.zero_grad()
                optimizer_pge.zero_grad()
                loss.backward()

                if epoch % 50 < 10:
                        optimizer_pge.step()
                else:
                    optimizer_feat.step()
                
                if _ == args.data_epoch - 1:
                    break

                feat_syn_inner = feat_syn.detach()
                g_syn, edge_weight = pge.inference(feat_syn)
                for j in range(args.arch_epoch):
                    optimizer_model.zero_grad()
                    output_syn_inner = model(g_syn, feat_syn_inner, edge_weight = edge_weight)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()
            
            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation_cr(data, task)
                loss_avg /= (data.nclass*outer_loop)
                print()
                print('Test, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))