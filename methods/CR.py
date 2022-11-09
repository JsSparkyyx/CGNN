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
        self.fisher = {}
        self.params = {}
        self.lamb_full = args.ewc_lamb_full
        self.lamb_mini = args.ewc_lamb_mini
        self.predict = torch.nn.ModuleList()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        for task, n_class, _ in taskcla:
            self.predict.append(torch.nn.Linear(in_feat,n_class))

        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def forward(self, g, features, task, mini_batch = False, edge_weight = None):
        h = self.arch(g, features, mini_batch = mini_batch, edge_weight = edge_weight)
        logits = self.predict[task](h)

        return logits

    def get_data(self, task, args):
        if args.dataset == 'cora':
            args.reduction_rate = 0.1
        elif args.dataset == 'reddit':
            args.reduction_rate = 0.01 
        feat_syn = torch.load('cgnn/feat_{}_{}_{}_{}.pt'.format(args.dataset,args.reduction_rate,args.seed,task))
        label_syn = torch.load('cgnn/labels_{}_{}_{}_{}.pt'.format(args.dataset,args.reduction_rate,args.seed,task))
        adj_syn = torch.load('cgnn/adj_{}_{}_{}_{}.pt'.format(args.dataset,args.reduction_rate,args.seed,task))
        def row_normalize_tensor(adj):
            mx = adj + torch.eye(adj.shape[0]).cuda()
            rowsum = mx.sum(1)
            r_inv = rowsum.pow(-1/2).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)
            mx = r_mat_inv @ mx
            mx = mx @ r_mat_inv
            return mx
        # adj_syn = row_normalize_tensor(adj_syn)
        # if args.dataset == 'cora':
        #     adj_syn[adj_syn < 0.05] = 0
        # elif args.dataset == 'reddit':
        #     adj_syn[adj_syn < 0.01] = 0
        g_selfloop = adj_syn + torch.eye(adj_syn.shape[0]).to(args.device)
        edge_index = g_selfloop.nonzero().T
        g_syn = dgl.graph((edge_index[0],edge_index[1])).to(args.device)
        edge_weight = g_selfloop[edge_index[0],edge_index[1]]
        feat_syn = feat_syn.to(args.device)
        label_syn = label_syn.to(args.device)
        return g_syn, feat_syn, label_syn, edge_weight
        
    def calculate_fisher(self, g, features, task, labels, mask):
        self.train()
        self.zero_grad()

        fisher = {}
        params = {}

        logits = self.forward(g, features, task)
        loss = self.ce(logits[mask],labels[mask])
        loss.backward()

        for n,p in self.named_parameters():
            if p.grad is not None:
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                fisher[n] = pg
                params[n] = pd
            else:
                fisher[n] = 0*p.data
                params[n] = 0*p.data
        return fisher, params

    def train_with_eval(self, g, features, task, labels, train_mask, val_mask, args):
        # if task != 0:
        #     # feat_syn, label_syn, adj_syn, g, features, labels, train_mask, val_mask, test_mask = self.get_data(task,args)
        #     g_syn, feat_syn, label_syn, edge_weight = self.get_data(task-1,args)
        self.train()
        g_syn1, feat_syn1, label_syn1, edge_weight1 = self.get_data(task,args)
        if task == 0:
            epoch = 600
            # epoch = args.epochs
        else:
            epoch = 600
            # epoch = args.epochs
        with trange(epoch) as tr:
            for epoch in tr:
                self.opt.zero_grad()
                # logits = self.forward(g, features, task)
                logits = self.forward(g_syn1, feat_syn1, task, edge_weight = edge_weight1)
                logits = logits
                # loss = self.ce(logits[train_mask],labels[train_mask])*0.01
                loss = self.ce(logits,label_syn1)
                loss_reg = torch.tensor(0)
                # if task != 0:
                #     logits = self.forward(g_syn, feat_syn, task-1, edge_weight = edge_weight)
                #     loss_reg = self.ce(logits,label_syn)
                #     loss += loss_reg

                loss_ewc = torch.tensor(0)
                # if task != 0:
                #     for t in range(task):
                #         for n, p in self.named_parameters():
                #             l = self.fisher[t][n]
                #             l = l * (p - self.params[t][n]).pow(2)
                #             loss_ewc += l.sum()
                # loss = loss + self.lamb_full*loss_ewc

                loss.backward()
                self.opt.step()

                if task == 0:
                    tr.set_postfix({'loss':'%.3f'%loss.item()},refresh=False)
                else:
                    tr.set_postfix({'loss':'%.3f'%loss.item(), 'loss_reg':'%.3f'%loss_reg.item(), 'loss_ewc':'%.3f'%(self.lamb_full*loss_ewc).item()},refresh=False)
                if epoch % 50 == 0:
                    acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                    print()
                    print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))
                    # if task != 0:
                    #     acc, mif1, maf1 = self.evaluation(g_syn, feat_syn, task-1, label_syn, torch.ones(g_syn.num_nodes(), dtype=torch.bool).to(args.device), edge_weight = edge_weight)
                    #     print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))
        
        # fisher, params = self.calculate_fisher(g, features, task, labels, train_mask)
        # self.current_task = task
        # self.fisher[self.current_task] = fisher
        # self.params[self.current_task] = params
    
    
    def batch_train_with_eval(self, dataloader, g, features, task, labels, train_mask, val_mask, args):
        if task != 0:
            g_syn, feat_syn, label_syn, edge_weight = self.get_data(task-1,args)
        self.train()

        if task == 0:
            epoch = 200
        else:
            epoch = 1000
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
                        loss += loss_reg*0.1
                    loss.backward()
                    self.opt.step()

                tr.set_postfix({'loss':'%.3f'%loss.item(), 'loss_reg':'%.3f'%loss_reg.item()},refresh=False)
                if epoch % 50 == 0:
                    acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                    print()
                    print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))
                    if task != 0:
                        acc, mif1, maf1 = self.evaluation(g_syn, feat_syn, task-1, label_syn, torch.ones(g_syn.num_nodes(), dtype=torch.bool).to(args.device), edge_weight = edge_weight)
                        print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

    @torch.no_grad()
    def evaluation(self, g, features, task, labels, val_mask, edge_weight = None):
        logits = self.forward(g, features, task, edge_weight = edge_weight)
        prob, prediction = torch.max(logits[val_mask], dim=1)
        prediction = prediction.cpu().numpy()
        labels = labels[val_mask].cpu().numpy()
        acc = accuracy_score(labels, prediction)
        mif1 = f1_score(labels, prediction, average='micro')
        maf1 = f1_score(labels, prediction, average='macro')
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)