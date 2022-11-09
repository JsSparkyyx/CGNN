from utils import *
from init_parameters import init_parameters
from data.load_data import *
import importlib

# python main.py --method CR --arch GAT --dataset reddit --manner full_batch --seed 0 --epoch 3000 --lr 0.001 --weight_decay 5e-4
def main(args):
    data, taskcla, in_feat = load_dataset(args.dataset, args.device)
    arc = importlib.import_module(f'models.{args.arch}')
    if args.arch == 'GAT':
        arc = arc.NET(in_feat, num_hidden = args.gat_hidden, heads = args.gat_head, num_layers = args.gat_layer)
        manager = importlib.import_module(f'methods.{args.method}')
        manager = manager.Manager(args.gat_hidden*args.gat_head, taskcla, arc, args).to(args.device)
    elif args.arch == 'SAGE':
        arc = arc.NET(in_feat, args.sage_hidden, n_layers = args.sage_layer, aggregator_type = args.aggregator_type)
        manager = importlib.import_module(f'methods.{args.method}')
        manager = manager.Manager(args.sage_hidden, taskcla, arc, args).to(args.device)
    elif args.arch == 'GCN':
        arc = arc.NET(in_feat, args.gcn_hidden, n_layers = args.gcn_layer)
        manager = importlib.import_module(f'methods.{args.method}')
        manager = manager.Manager(args.gcn_hidden, taskcla, arc, args).to(args.device)

    # from methods.GC import GCManager
    # manager = GCManager(in_feat,taskcla,args)
    # manager.train_task(data[0],0)

    results = pd.DataFrame([],columns=['stage','task','accuracy','micro-f1','macro-f1','seed'])

    if args.manner == 'full_batch':
        for task in range(args.num_tasks):
            print('Train task:{}'.format(task))
            g, features, _, labels, train_mask, val_mask, test_mask = data[task].retrieve_data()
            manager.train_with_eval(g, features, task, labels, train_mask, val_mask, args)
            for previous in range(task+1):
                g, features, _, labels, train_mask, val_mask, test_mask = data[previous].retrieve_data()
                acc, mif1, maf1 = manager.evaluation(g, features, previous, labels, test_mask)
                print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(task, previous, acc, mif1, maf1))
                results.loc[len(results.index)] = [task,previous,acc,mif1,maf1,args.seed]
    elif args.manner == 'mini_batch':
        for task in range(args.num_tasks):
            print('Train task:{}'.format(task))
            dataloader = data[task].graph_sampling(batch_size = args.batch_size, shuffle = True)
            g, features, _, labels, train_mask, val_mask, test_mask = data[task].retrieve_data()
            manager.batch_train_with_eval(dataloader, g, features, task, labels, train_mask, val_mask, args)
            for previous in range(task+1):
                g, features, _, labels, train_mask, val_mask, test_mask = data[previous].retrieve_data()
                acc, mif1, maf1 = manager.evaluation(g, features, previous, labels, test_mask)
                print('Stage:{} Task:{}, ACC:{}, Micro-F1:{}, Macro-F1:{}'.format(task, previous, acc, mif1, maf1))
                results.loc[len(results.index)] = [task,previous,acc,mif1,maf1,args.seed]
    save_results(results,args)

if __name__ == '__main__':
    args = init_parameters()

    # torch.cuda.set_device(args.gpu_id)
    args.device = 'cuda:{}'.format(str(args.gpu_id)) if torch.cuda.is_available() else 'cpu'
    args.fanouts = [int(i) for i in args.fanouts.split(',')]
    set_seed(args.seed)

    main(args)