import argparse
import os
import time
import random           
import numpy as np     
import pickle
import scipy.io as sio
import scipy.sparse as sp
import networkit as nk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    nk.setSeed(seed, True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate community partitions using Louvain (Networkit).")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--seed', type=int, default=171, help="Random seed")
    parser.add_argument('--resolution', type=float, default=1.0, help="Resolution (gamma).")
    parser.add_argument('--data_dir', type=str, default='./', help="Directory containing the raw .mat files")
    parser.add_argument('--save_dir', type=str, default='./partition', help="Directory to save the generated .pkl files")
    
    args = parser.parse_args()
    
    dataset = args.dataset
    seed = args.seed
    resolution = args.resolution
    data_dir = args.data_dir
    save_dir = args.save_dir

    set_seed(seed)

    print("=========================================")
    print(f" Processing dataset: {dataset}")
    print("=========================================")

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


    print("Building NetworKit Graph...")
    start_time = time.time()
    coo = adj.tocoo()
    G_nk = nk.Graph(n=adj.shape[0], weighted=False, directed=False)

    for u, v in zip(coo.row, coo.col):
        if u < v:  
            G_nk.addEdge(u, v)
    print(f"Graph built! Time elapsed: {time.time() - start_time:.2f} s")


    print("Running Louvain ...")
    louvain_start = time.time()
    louvain = nk.community.PLM(G_nk, True)
    # louvain = nk.community.PLM(G_nk, refine=True, gamma=resolution)
    louvain.run()
    print(f"Louvain done! Time elapsed: {time.time() - louvain_start:.2f} s")


    partition_list = louvain.getPartition().getVector()
    partition = {i: comm for i, comm in enumerate(partition_list)}

    num_communities = max(partition.values()) + 1
    print(f"Found {num_communities} communities.")


    os.makedirs(save_dir, exist_ok=True)  
    save_path = os.path.join(save_dir, f"{dataset}_partition.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump(partition, f)

    print(f"Done! Partition saved to: {save_path}\n")