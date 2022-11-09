import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import dgl

class Manager(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 taskcla,
                 arch,
                 args):
        super(Manager, self).__init__()
        self.arch = arch
        self.current_task = 0
        self.predict = torch.nn.ModuleList()
        self.offsets = []
        self.n_class = []
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        offset = 0
        for task, n_class, _ in taskcla:
            self.offsets.append(offset)
            self.n_class.append(n_class)
            offset += n_class
            self.predict.append(torch.nn.Linear(in_feat,n_class))
        # self.predict = torch.nn.Linear(in_feat,offset)

        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # self.opt = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
    
    def forward(self, g, features, task, mini_batch = False, edge_weight = None):
        h = self.arch(g, features, mini_batch = mini_batch, edge_weight = edge_weight)
        logits = self.predict[task](h)
        # logits = self.predict(h)

        return logits

    def get_data(self, task, args):
        if args.dataset == 'cora':
            args.reduction_rate = 0.1
        elif args.dataset == 'reddit':
            args.reduction_rate = 0.01 
        feat_syn = torch.load('cgnn/feat_{}_{}_{}_{}.pt'.format(args.dataset,args.reduction_rate,args.seed,task))
        label_syn = torch.load('cgnn/labels_{}_{}_{}_{}.pt'.format(args.dataset,args.reduction_rate,args.seed,task))
        adj_syn = torch.load('cgnn/adj_{}_{}_{}_{}.pt'.format(args.dataset,args.reduction_rate,args.seed,task))
        adj_syn[adj_syn < 0.05] = 0
        g_selfloop = adj_syn + torch.eye(adj_syn.shape[0]).to(args.device)
        edge_index = g_selfloop.nonzero().T
        g_syn = dgl.graph((edge_index[0],edge_index[1])).to(args.device)
        edge_weight = g_selfloop[edge_index[0],edge_index[1]]
        feat_syn = feat_syn.to(args.device)
        label_syn = label_syn.to(args.device)
        return g_syn, feat_syn, label_syn, edge_weight
        # with open('cgnn/data_{}_{}_{}_{}.pkl'.format(args.dataset,args.reduction_rate,args.seed,args.seed,task),'rb') as f:
        #     data = pickle.load(f)
        # with open('cgnn/taskcla_{}_{}_{}_{}.pkl'.format(args.dataset,args.reduction_rate,args.seed,args.seed,task),'rb') as f:
        #     taskcla = pickle.load(f)
        # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        # adj, features, labels = data.adj, data.features, data.labels
        # g = dgl.from_scipy(adj).to(device)
        # g = dgl.add_self_loop(g)
        # features = torch.Tensor(features).to(device)
        # labels = torch.LongTensor(labels).to(device)
        # idx_train = torch.LongTensor(idx_train).to(device)
        # idx_val = torch.LongTensor(idx_val).to(device)
        # idx_test = torch.LongTensor(idx_test).to(device)
        # train_mask = torch.zeros((labels.shape[0], ), dtype=torch.bool)
        # val_mask = torch.zeros((labels.shape[0], ), dtype=torch.bool)
        # test_mask = torch.zeros((labels.shape[0], ), dtype=torch.bool)
        # train_mask[idx_train] = 1
        # val_mask[idx_val] = 1
        # test_mask[idx_test] = 1
        # return feat_syn, label_syn, adj_syn, g, features, labels, train_mask, val_mask, test_mask
        

    def train_with_eval(self, g, features, task, labels, train_mask, val_mask, args):
        if task != 0:
            gs_syn = []
            offset = 0
            for i in range(task-1,task):
                g_syn, feat_syn, label_syn, edge_weight = self.get_data(i,args)
                g_syn.ndata['x'] = feat_syn
                g_syn.ndata['y'] = label_syn + offset
                g_syn.ndata['m'] = torch.ones(g_syn.num_nodes(), dtype=torch.bool).to(args.device)
                g_syn.edata['w'] = edge_weight
                gs_syn.append(g_syn)
                offset += self.n_class[task-1]
            g.ndata['x'] = features
            g.ndata['y'] = labels + offset
            g.ndata['m'] = train_mask
            g.edata['w'] = torch.ones(g.num_edges()).to(args.device)
            gs = dgl.batch([g]+gs_syn)
            # feat_syn, label_syn, adj_syn, g, features, labels, train_mask, val_mask, test_mask = self.get_data(task,args)
        self.train()

        if task == 0:
            epoch = 200
            # epoch = args.epochs
        else:
            # epoch = 200
            epoch = args.epochs

        offset = self.offsets[task]
        n_class = self.n_class[task]
        with trange(epoch) as tr:
            for epoch in tr:
                if epoch % 1000 == 0:
                    self.opt = torch.optim.Adam(self.parameters(), lr=self.lr*0.1, weight_decay=self.weight_decay)
                self.opt.zero_grad()
                if task == 0:
                    logits = self.forward(g, features, task)
                    logits = logits
                    loss = self.ce(logits[train_mask, offset:offset+n_class],labels[train_mask])
                else:
                    logits = self.forward(gs, gs.ndata['x'], task, edge_weight = gs.edata['w'])
                    loss = self.ce(logits[gs.ndata['m'],self.offsets[task-1]:offset+n_class],gs.ndata['y'][gs.ndata['m']])

                loss.backward()
                self.opt.step()
                tr.set_postfix({'loss':'%.3f'%loss.item()},refresh=False)
                # if task != 0:
                #     self.zero_grad()
                #     logits = self.forward(g_syn, feat_syn, task-1, edge_weight = edge_weight)
                #     loss = self.ce(logits,label_syn)
                #     loss.backward()
                #     self.opt.step()
                
                if epoch % 50 == 0:
                    acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                    print()
                    print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

    def batch_train_with_eval(self, dataloader, g, features, task, labels, train_mask, val_mask, args):
        if task != 0:
            g_syn, feat_syn, label_syn, edge_weight = self.get_data(task-1,args)
        self.train()

        if task == 0:
            epoch = 200
        else:
            epoch = 3000
        with trange(epoch) as tr:
            for epoch in tr:
                for seed_nodes, output_nodes, blocks in dataloader:
                    self.zero_grad()
                    logits = self.forward(blocks, features[seed_nodes], task, mini_batch = True)
                    loss = self.ce(logits,labels[output_nodes])
                    # loss.backward()
                    # self.opt.step()
                    loss_reg = torch.tensor(0)
                    if task != 0:
                        self.zero_grad()
                        logits = self.forward(g_syn, feat_syn, task-1, edge_weight = edge_weight)
                        loss_reg = self.ce(logits,label_syn)
                        loss += loss_reg*0.01
                    loss.backward()
                    self.opt.step()

                tr.set_postfix({'loss':'%.3f'%loss.item(), 'loss_reg':'%.3f'%loss_reg.item()},refresh=False)
                if epoch % 50 == 0:
                    acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                    print()
                    print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

    @torch.no_grad()
    def evaluation(self, g, features, task, labels, val_mask):
        logits = self.forward(g, features, task)
        offset = self.offsets[task]
        n_class = self.n_class[task]
        # prob, prediction = torch.max(logits[val_mask, offset:offset+n_class], dim=1)
        prob, prediction = torch.max(logits[val_mask], dim=1)
        prediction = prediction.cpu().numpy()
        labels = labels[val_mask].cpu().numpy()
        acc = accuracy_score(labels, prediction)
        mif1 = f1_score(labels, prediction, average='micro')
        maf1 = f1_score(labels, prediction, average='macro')
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)