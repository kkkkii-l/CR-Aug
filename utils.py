import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter



def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



def load_mat(dataset, train_rate=0.3, val_rate=0.1):

    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    if not sp.issparse(network):
        adj = sp.csr_matrix(network)
    else:
        adj = network

    if not sp.issparse(attr):
        feat = sp.lil_matrix(attr)
    else:
        feat = attr

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))   
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))  
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)

    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))

    # train_ano_idx = idx_train[ano_labels[idx_train] == 1]

    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    all_abnormal_label_idx = [i for i in idx_train if ano_labels[i] == 1]

    rate = 0.5 
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]  # 15%
    abnormal_idx = all_abnormal_label_idx[: int(len(all_abnormal_label_idx) * rate)]
    print('Training rate', rate)


    random.shuffle(normal_label_idx)

    if dataset in ['reddit']:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.5)]  
    else:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.15)]   
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx, abnormal_idx



def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    # nx_graph = nx.from_scipy_sparse_matrix(adj)
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size * 3]
        subv[i].append(i)

    return subv



def compute_community_diff(features, partition, idx_train, normal_label_idx,
                             nb_nodes, adj,  
                             use_cuda=False, 
                             sample_max=5000, min_comm_size=5,
                             min_nor_size=2
                            ):
    
    num_communities = max(partition.values()) + 1

    feat_2d = features.squeeze() if features.dim() == 3 else features
    
    feat_norm = torch.nn.functional.normalize(feat_2d, p=2, dim=1)

    if use_cuda and torch.cuda.is_available():
        feat_norm = feat_norm.cuda()
        adj = adj.cuda()

    if adj.is_sparse:
        sum_neighbor_feats = torch.sparse.mm(adj, feat_norm)
    else:
        sum_neighbor_feats = torch.mm(adj, feat_norm)
    

    numerator = (feat_norm * sum_neighbor_feats).sum(dim=1)

    if adj.is_sparse:
        ones_vec = torch.ones(nb_nodes, 1).to(feat_norm.device)
        degrees = torch.sparse.mm(adj, ones_vec).flatten()
    else:
        degrees = adj.sum(dim=1)
    

    degrees[degrees == 0] = 1.0 
    node_neighbor_sims = numerator / degrees  
    
    node_neighbor_sims_np = node_neighbor_sims.detach().cpu().numpy()
    
    comm_labels_np = np.array([partition[i] for i in range(nb_nodes)])
    comm_labels = torch.LongTensor(comm_labels_np)

    if torch.is_tensor(idx_train):
        idx_train = idx_train.cpu().numpy()
    train_set = set(idx_train)
    
    if torch.is_tensor(normal_label_idx):
        normal_label_idx = normal_label_idx.cpu().numpy()
    normal_nodes = np.array(normal_label_idx)
    normal_set = set(normal_nodes)

    baseline_normal_sim = node_neighbor_sims_np[normal_nodes].mean()
    
    comm_sim_list = []
    diff_list = []   
    
    for c in range(num_communities):
        all_nodes_in_c = np.where(comm_labels_np == c)[0]
        nodes_in_c = np.array([n for n in all_nodes_in_c if n in train_set])

        if len(nodes_in_c) < min_comm_size:
            default_val = 2 
            comm_sim_list.append(default_val)
            diff_list.append(default_val)
            continue
        
        avg_sim = node_neighbor_sims_np[nodes_in_c].mean()
        
        comm_sim_list.append(float(avg_sim))

        normal_nodes_in_c = np.array([n for n in nodes_in_c if n in normal_set])
        
        if len(normal_nodes_in_c) >= min_nor_size:
            sim_normal_local = node_neighbor_sims_np[normal_nodes_in_c].mean()
            
            diff = avg_sim - sim_normal_local
            diff_list.append(float(diff))

        else:

            diff = avg_sim - baseline_normal_sim
            diff_list.append(float(diff))


    comm_sims_tensor = torch.tensor(comm_sim_list)
    
    valid_mask = (comm_sims_tensor != 2)
    if valid_mask.sum() > 0:
        minn = comm_sims_tensor[valid_mask].min().item()
        maxx = comm_sims_tensor[valid_mask].max().item()
        default_sim = (minn + maxx) / 2
    else:
        default_sim = 0.5


    comm_sim_list = [default_sim if x == 2 else x for x in comm_sim_list]
    comm_sims = torch.tensor(comm_sim_list)

    abs_diffs = np.array([abs(x) for x in diff_list])
    mask_invalid = (abs_diffs > 1.0) 
    mask_valid = ~mask_invalid
    
    if mask_valid.sum() > 0:
        mean_diff = abs_diffs[mask_valid].mean()
    else:
        mean_diff = 0.0
        
    
    abs_diffs[mask_invalid] = mean_diff
    comm_diffs = torch.tensor(abs_diffs)
        
    return comm_sims, comm_diffs, comm_labels, num_communities


