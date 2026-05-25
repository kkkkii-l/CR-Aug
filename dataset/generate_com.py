import argparse
import os
import time
import random
import numpy as np
import pickle
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
import community.community_louvain as community_louvain  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate community partitions using Python-Louvain (NetworkX).")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--seed', type=int, default=171, help="Random seed")
    parser.add_argument('--data_dir', type=str, default='./', help="Directory containing the raw .mat files")
    parser.add_argument('--save_dir', type=str, default='./partition', help="Directory to save the generated .pkl files")
    
    args = parser.parse_args()
    
    dataset = args.dataset
    seed = args.seed
    data_dir = args.data_dir
    save_dir = args.save_dir

    set_seed(seed)

    print("=====================================================")
    print(f" 🚀 Processing dataset: {dataset} (NetworkX Version)")
    print("=====================================================")

    data_path = os.path.join(data_dir, f"{dataset}.mat")
    print(f"Loading {data_path} ...")
    try:
        data = sio.loadmat(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find {data_path}. Please check the data directory.")

    network = data['Network'] if ('Network' in data) else data['A']
    if not sp.issparse(network):
        adj = sp.csr_matrix(network)
    else:
        adj = network

    print("Building NetworkX Graph...")
    start_time = time.time()
    
    if hasattr(nx, 'from_scipy_sparse_array'):
        G_nx = nx.from_scipy_sparse_array(adj)
    else:
        G_nx = nx.from_scipy_sparse_matrix(adj)
        
    print(f"Graph built! Time elapsed: {time.time() - start_time:.2f} s")

    print("Starting Community Detection (python-louvain)...")
    louvain_start = time.time()
    

    partition = community_louvain.best_partition(G_nx, random_state=seed)
    
    print(f"Louvain done! Time elapsed: {time.time() - louvain_start:.2f} s")

    num_communities = max(partition.values()) + 1
    print(f"Detected {num_communities} communities.")


    os.makedirs(save_dir, exist_ok=True)  
    save_path = os.path.join(save_dir, f"{dataset}_partition.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(partition, f)

    print(f"✅ Done! Partition successfully saved to: {save_path}\n")