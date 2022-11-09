import torch
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score

class Manager(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 taskcla,
                 arch,
                 args,
                 lr = 0.005,
                 weight_decay = 0.001):
        super(Manager, self).__init__()
        self.arch = arch
        self.current_task = 0

        self.predict = torch.nn.ModuleList()
        for task, n_class, _ in taskcla:
            self.predict.append(torch.nn.Linear(in_feat,n_class))

        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def forward(self, g, features, task, mini_batch = False):
        h = self.arch(g, features, mini_batch)
        logits = self.predict[task](h)

        return logits

    def train_with_eval(self, g, features, task, labels, train_mask, val_mask, args):
        self.train()

        for epoch in trange(args.epochs, leave=False):
            self.zero_grad()
            logits = self.forward(g, features, task)
            loss = self.ce(logits[train_mask],labels[train_mask])
            loss.backward()
            self.opt.step()

            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                print()
                print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

        fisher, params = self.calculate_fisher(g, features, task, labels, train_mask)
        self.current_task = task
        self.fisher[self.current_task] = fisher
        self.params[self.current_task] = params

    def batch_train_with_eval(self, dataloader, g, features, task, labels, train_mask, val_mask, args):
        self.train()

        for epoch in trange(args.epochs, leave=False):
            for seed_nodes, output_nodes, blocks in dataloader:
                self.zero_grad()
                logits = self.forward(blocks, features[seed_nodes], task, mini_batch = True)
                loss = self.ce(logits,labels[output_nodes])
                loss.backward()
                self.opt.step()

            if epoch % 50 == 0:
                acc, mif1, maf1 = self.evaluation(g, features, task, labels, val_mask)
                print()
                print('Val, Epoch:{}, Loss:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(epoch, loss.item(), acc, mif1, maf1))

        fisher, params = self.calculate_fisher(g, features, task, labels, train_mask)
        self.current_task = task
        self.fisher[self.current_task] = fisher
        self.params[self.current_task] = params

    @torch.no_grad()
    def evaluation(self, g, features, task, labels, val_mask):
        logits = self.forward(g, features, task)
        prob, prediction = torch.max(logits, dim=1)
        prediction = prediction[val_mask].cpu().numpy()
        labels = labels[val_mask].cpu().numpy()
        acc = accuracy_score(labels, prediction)
        mif1 = f1_score(labels, prediction, average='micro')
        maf1 = f1_score(labels, prediction, average='macro')
        return round(acc*100,2), round(mif1*100,2), round(maf1*100,2)