import argparse

def init_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cora','amazon','reddit','cfd'], default='cfd')
    parser.add_argument('--arch', '--architecture', type=str, choices=['HTG','GAT','GCN','SAGE'], default='HTG')
    parser.add_argument('--manner', type=str, choices=['full_batch','mini_batch'], default='full_batch')
    parser.add_argument('--method', type=str, choices=['CFD','Finetune','CR','EWC','HAT','GEM','MAS'], default='CFD')
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fanouts', type=str, default='10,15')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

    # parameters for CFD
    parser.add_argument('--lamb_distill', type=float, default=25)

    # parameters for CR
    parser.add_argument('--cr_hidden', type=int, default=256)
    parser.add_argument('--cr_epoch', type=int, default=600)
    parser.add_argument('--lr_cr', type=float, default=0.01)
    parser.add_argument('--data_epoch', type=int, default=10)
    parser.add_argument('--arch_epoch', type=int, default=1)
    parser.add_argument('--reduction_rate', type=float, default=0.01)

    # parameters for HTG
    parser.add_argument('--htg_layer', type=int, default=2)
    parser.add_argument('--htg_head', type=int, default=8)
    parser.add_argument('--htg_hidden', type=int, default=32)

    # parameters for GAT
    parser.add_argument('--gat_layer', type=int, default=2)
    parser.add_argument('--gat_head', type=int, default=8)
    parser.add_argument('--gat_hidden', type=int, default=32)
    
    # parameters for GCN
    parser.add_argument('--gcn_layer', type=int, default=2)
    parser.add_argument('--gcn_hidden', type=int, default=256)
    
    # parameters for SAGE
    parser.add_argument('--sage_layer', type=int, default=2)
    parser.add_argument('--sage_hidden', type=int, default=256)
    parser.add_argument('--aggregator_type', type=str, choices=['mean','pool','lstm','gcn'], default='mean')

    # parameters for EWC
    parser.add_argument('--ewc_lamb_full', type=int, default=1000)
    parser.add_argument('--ewc_lamb_mini', type=int, default=1000000)

    # parameters for GEM
    parser.add_argument('--gem_margin', type=float, default=0.5)

    # parameters for MAS
    parser.add_argument('--mas_lamb_full', type=int, default=50000)
    parser.add_argument('--mas_lamb_mini', type=int, default=1000)

    # parameters for DERPP
    parser.add_argument('--derpp_alpha', type=int, default=50000)
    parser.add_argument('--derpp_beta', type=int, default=1000)
    parser.add_argument('--derpp_buffer_size', type=int, default=1000)

    args = parser.parse_args()
    return args