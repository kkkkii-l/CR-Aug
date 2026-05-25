import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch.nn as nn
import torch
torch.use_deterministic_algorithms(True)
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from model import Model 

from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

import pickle


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', type=str, default='photo') 
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)  
parser.add_argument('--seed', type=int, default=171)
parser.add_argument('--embedding_dim', type=int, default=300)   
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)

parser.add_argument('--readout', type=str, default='avg') 
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)  
parser.add_argument('--mean', type=float, default=0.05)
parser.add_argument('--var', type=float, default=0.0)

parser.add_argument('--use_cuda', type=int, default=1)

parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--frequency', type=int, default=20)


args = parser.parse_args()

if args.lr is None:
    args.lr = 1e-3


if args.num_epoch is None:
    if args.dataset in ['photo']: args.num_epoch = 500       
    if args.dataset in ['elliptic']: args.num_epoch = 1200    
    if args.dataset in ['reddit']: args.num_epoch = 200 
    elif args.dataset in ['t_finance']: args.num_epoch = 400  
    elif args.dataset in ['Amazon']: args.num_epoch = 3000     
    elif args.dataset in ['tolokers']: args.num_epoch = 1300      

if args.dataset in ['reddit']:
    args.mean = 0.2  
    args.var = 0.0
else:
    args.mean = 0.05
    args.var = 0.0

# if args.dataset in ['elliptic']:
#     args.ratio = 1.5


print('Dataset: ', args.dataset)

# Set random seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
dgl.seed(args.seed)
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load and preprocess data
adj, features, labels, all_idx, idx_train, idx_val, idx_test, \
ano_label, str_ano_label, attr_ano_label, \
normal_label_idx, abnormal_label_idx, abnormal_idx = load_mat(args.dataset)

if args.dataset in ['Amazon', 'reddit', 'elliptic', 'yelpchi']:   
    features, _ = preprocess_features(features)
else:
    features = features.todense()

dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]


adj_for_community = adj
print('Adj sum:', adj.sum())
adj = normalize_adj(adj) 

adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
features = torch.FloatTensor(features)
adj = torch.FloatTensor(adj)
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])


# =====================================================================

print("Starting Community Detection (Loading from cache)...")
start_comm = time.time()

partition_cache_file = f'./dataset/partition/{args.dataset}_partition.pkl'

if os.path.exists(partition_cache_file):
    print(f"Loading from {partition_cache_file} ...")
    with open(partition_cache_file, 'rb') as f:
        partition = pickle.load(f)
    print(f"success : {time.time() - start_comm:.4f} s")
else:
    raise FileNotFoundError(f"Can not find {partition_cache_file}! Please run the prepare_all_data.sh script first.")


num_communities = max(partition.values()) + 1
print(f"Detected {num_communities} communities.")

# =====================================================================


if args.dataset in ['reddit']: min_comm_size = 10
elif args.dataset in ['photo']: min_comm_size = 10 #20 
elif args.dataset in ['Amazon']: min_comm_size = 10
elif args.dataset in ['elliptic']: min_comm_size = 10
elif args.dataset in ['t_finance']: min_comm_size = 20
elif args.dataset in ['tolokers']: min_comm_size = 10

min_nor_size = min_comm_size*0.2

adj_coo = adj_for_community.tocoo()

values = adj_coo.data
indices = np.vstack((adj_coo.row, adj_coo.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = adj_coo.shape

torch_adj_raw = torch.sparse_coo_tensor(i, v, torch.Size(shape))

comm_alphas, comm_diffs, comm_labels, num_communities = compute_community_diff(
    features=features, 
    partition=partition,          
    idx_train=idx_train,
    normal_label_idx=normal_label_idx,
    nb_nodes=nb_nodes,
    adj=torch_adj_raw,            
    use_cuda=args.use_cuda,                 
    sample_max=5000,                    
    min_comm_size=min_comm_size,      
    min_nor_size=min_nor_size
)


diff_min = torch.min(comm_diffs)
diff_max = torch.max(comm_diffs)
risk_scores = (comm_diffs - diff_min) / (diff_max - diff_min + 1e-8)  

print("Community risk scores:", risk_scores)

if args.use_cuda and torch.cuda.is_available():
    comm_labels = comm_labels.cuda()
    comm_alphas = comm_alphas.cuda()
    comm_diffs = comm_diffs.cuda()
    risk_scores = risk_scores.cuda()
print(f"Total Community Step Time: {time.time() - start_comm:.2f}s")

# =====================================================================

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if isinstance(partition, dict):
    print("Converting partition dict to tensor...")
    part_list = [partition[i] for i in range(nb_nodes)] 
    partition_tensor = torch.LongTensor(part_list)
else:
    partition_tensor = torch.LongTensor(partition)

if isinstance(risk_scores, torch.Tensor):
    risk_scores_tensor = risk_scores.float()
else:
    risk_scores_tensor = torch.FloatTensor(risk_scores)

if args.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    partition_tensor = partition_tensor.to(device)
    risk_scores_tensor = risk_scores_tensor.to(device)
    print(f"Partition & Risks moved to {device}")
else:
    print("Partition & Risks on CPU")


if args.use_cuda and torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()


b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()   
if args.use_cuda and torch.cuda.is_available():
    b_xent = b_xent.cuda()

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    total_time = 0
    for epoch in range(args.num_epoch):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()

        # Train model
        train_flag = True

        emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, partition_tensor, risk_scores_tensor,
                                                                abnormal_label_idx, normal_label_idx, idx_train,
                                                                train_flag, args)


        if epoch > args.num_epoch * args.ratio  and epoch % args.frequency == 0:
            print(f"\n[Epoch {epoch}] Updating Community Risk Scores using learned embeddings...")
            
            new_alphas, new_diffs, new_c_labels, _ = compute_community_diff(
                features=emb.detach(), 
                partition=partition,        
                idx_train=idx_train,
                normal_label_idx=normal_label_idx,
                nb_nodes=nb_nodes,
                adj=torch_adj_raw,          
                use_cuda=args.use_cuda,               
                sample_max=5000,                   
                min_comm_size=min_comm_size,        
                min_nor_size=min_nor_size
            )

            d_min = torch.min(new_diffs)
            d_max = torch.max(new_diffs)
            new_risk_scores = (new_diffs - d_min) / (d_max - d_min + 1e-8)

            if isinstance(new_risk_scores, torch.Tensor):
                risk_scores_tensor = new_risk_scores.float()
            else:
                risk_scores_tensor = torch.FloatTensor(new_risk_scores)
                
            if args.use_cuda and torch.cuda.is_available():
                risk_scores_tensor = risk_scores_tensor.to(device)


        # =========================================================================
        # BCE loss
        # =========================================================================

        lbl = torch.unsqueeze(torch.cat(
            (torch.zeros(len(normal_label_idx)), torch.ones(len(emb_con)))),
            1).unsqueeze(0)
        if args.use_cuda and torch.cuda.is_available():
            lbl = lbl.cuda()

        loss_bce = b_xent(logits, lbl)  
        loss_bce = torch.mean(loss_bce)  

        loss = 1 * loss_bce 

        loss.backward()
        optimiser.step()
        end_time = time.time()
        total_time += end_time - start_time
        print('Total time is', total_time)


        if epoch % 2 == 0:
            logits = np.squeeze(logits.cpu().detach().numpy())
            lbl = np.squeeze(lbl.cpu().detach().numpy())
            auc = roc_auc_score(lbl, logits)
            print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
            print("=====================================================================")


        if epoch % 10 == 0:
            model.eval()
            train_flag = False
            emb, emb_combine, logits, emb_con, emb_abnormal = model(features, adj, partition_tensor, risk_scores_tensor,
                                                                abnormal_label_idx, normal_label_idx, idx_train,
                                                                train_flag, args)
            logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
            auc = roc_auc_score(ano_label[idx_test], logits)
            print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
            AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
            print('Testing AP:', AP)

