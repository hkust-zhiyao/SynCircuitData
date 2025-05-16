import os
import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import glob
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time
import random
def load_graphml_files_with_size_limit(folder_path, sample_size=20, max_nodes=15000):
    
    graphml_files = glob.glob(os.path.join(folder_path, "*.graphml"))
    print(f"found {len(graphml_files)} files in {folder_path}")
    
    
    random.shuffle(graphml_files)
    
    graphs = []
    processed_count = 0
    
    for file_path in tqdm(graphml_files, desc="loading graph files"):
        try:
            G = nx.read_graphml(file_path)
            
            if len(G.nodes) <= max_nodes:
                graphs.append(G)
                processed_count += 1
                
                if processed_count >= sample_size:
                    break
            else:
                print(f"skip file {file_path}, node number {len(G.nodes)} exceeds limit {max_nodes}")
        except Exception as e:
            print(f"cannot load file {file_path}: {e}")
    
    print(f"successfully loaded {len(graphs)} graphs with node number limit")
    return graphs


def load_graphml_files(folder_path, sample_size=20):
    
    graphml_files = glob.glob(os.path.join(folder_path, "*.graphml"))
    print(f"found {len(graphml_files)} files in {folder_path}")
    
    if len(graphml_files) > sample_size:
        print(f"randomly sample {sample_size} files for analysis")
        graphml_files = random.sample(graphml_files, sample_size)
    
    graphs = []
    for file_path in tqdm(graphml_files, desc="加载图文件"):
        try:
            G = nx.read_graphml(file_path)
            graphs.append(G)
        except Exception as e:
            print(f"cannot load file {file_path}: {e}")
    return graphs

def fast_mmd(features1, features2, kernel_bandwidth=1.0, sample_size=1000):
    
    
    if len(features1) > sample_size:
        idx1 = np.random.choice(len(features1), sample_size, replace=False)
        features1 = features1[idx1]
    
    if len(features2) > sample_size:
        idx2 = np.random.choice(len(features2), sample_size, replace=False)
        features2 = features2[idx2]
    
    
    if len(features1.shape) == 1:
        features1 = features1.reshape(-1, 1)
    if len(features2.shape) == 1:
        features2 = features2.reshape(-1, 1)
    
    
    n_x = features1.shape[0]
    n_y = features2.shape[0]
    
    
    gamma = 1.0 / kernel_bandwidth
    X2 = np.sum(features1**2, axis=1).reshape(-1, 1)
    Y2 = np.sum(features2**2, axis=1).reshape(-1, 1)
    
    K_XX = np.exp(-gamma * (X2 + X2.T - 2 * np.dot(features1, features1.T)))
    K_YY = np.exp(-gamma * (Y2 + Y2.T - 2 * np.dot(features2, features2.T)))
    K_XY = np.exp(-gamma * (X2 + Y2.T - 2 * np.dot(features1, features2.T)))
    
    mmd = (np.sum(K_XX) / (n_x * n_x) + 
           np.sum(K_YY) / (n_y * n_y) - 
           2 * np.sum(K_XY) / (n_x * n_y))
    
    return float(mmd)

def extract_graph_features(G):
    
    features = {}
    
    
    in_degrees = np.array([d for n, d in G.in_degree()])
    out_degrees = np.array([d for n, d in G.out_degree()])
    total_degrees = in_degrees + out_degrees
    
    features['in_degrees'] = in_degrees
    features['out_degrees'] = out_degrees
    features['total_degrees'] = total_degrees
    
    
    try:
        features['clustering'] = np.array(list(nx.clustering(G).values()))
    except:
        features['clustering'] = np.array([0])
    
    
    try:
        features['pagerank'] = np.array(list(nx.pagerank(G, max_iter=100).values()))
    except:
        features['pagerank'] = np.array([0])
    
    
    try:
        features['eigenvector_centrality'] = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    except:
        features['eigenvector_centrality'] = np.array([0])
    
    
    try:
        A = nx.to_scipy_sparse_array(G)
        k = min(100, A.shape[0]-1) if A.shape[0] > 1 else 1
        from scipy.sparse.linalg import eigs
        if k > 0:
            eigvals = eigs(A, k=k, return_eigenvectors=False)
            features['eigvals'] = np.real(eigvals)
        else:
            features['eigvals'] = np.array([0])
    except:
        features['eigvals'] = np.array([0])
    
    
    scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
    features['scc_sizes'] = np.array(scc_sizes) if scc_sizes else np.array([0])
    
    
    
    only_in = np.sum((in_degrees > 0) & (out_degrees == 0)) / max(1, len(in_degrees))
    only_out = np.sum((out_degrees > 0) & (in_degrees == 0)) / max(1, len(out_degrees))
    features['terminal_nodes'] = np.array([only_in, only_out])
    
    return features

def compute_mmds_from_features(features1, features2):
    
    results = {}
    
    
    results['degree distribution MMD'] = {
        'in_degree_mmd': fast_mmd(features1['in_degrees'], features2['in_degrees']),
        'out_degree_mmd': fast_mmd(features1['out_degrees'], features2['out_degrees']),
        'total_degree_mmd': fast_mmd(features1['total_degrees'], features2['total_degrees'])
    }
    
    
    results['clustering coefficient MMD'] = fast_mmd(features1['clustering'], features2['clustering'])
    results['spectral features MMD'] = fast_mmd(features1['eigvals'], features2['eigvals'])
    results['PageRank MMD'] = fast_mmd(features1['pagerank'], features2['pagerank'])
    results['eigenvector centrality MMD'] = fast_mmd(features1['eigenvector_centrality'], features2['eigenvector_centrality'])
    results['scc size MMD'] = fast_mmd(features1['scc_sizes'], features2['scc_sizes'])
    results['terminal node distribution MMD'] = fast_mmd(features1['terminal_nodes'], features2['terminal_nodes'])
    
    return results

def process_graph_pair(pair, all_features1, all_features2):
    
    i, j = pair
    return compute_mmds_from_features(all_features1[i], all_features2[j])

def compare_graph_folders(folder1, folder2, sample_graphs=20, sample_pairs=100):
    
    start_time = time.time()
    
    
    print("loading graphs from first folder...")
    graphs1 = load_graphml_files_with_size_limit(folder1, sample_size=sample_graphs, max_nodes=15000)
    print("loading graphs from second folder...")
    graphs2 = load_graphml_files_with_size_limit(folder2, sample_size=sample_graphs, max_nodes=15000)
    
    if not graphs1 or not graphs2:
        return "at least one folder has no valid GraphML files"
    
    
    print("pre-calculating graph features...")
    all_features1 = []
    all_features2 = []
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        all_features1 = list(tqdm(pool.imap(extract_graph_features, graphs1), 
                               total=len(graphs1), desc="processing first folder"))
        all_features2 = list(tqdm(pool.imap(extract_graph_features, graphs2), 
                               total=len(graphs2), desc="processing second folder"))
    
    
    all_pairs = [(i, j) for i in range(len(graphs1)) for j in range(len(graphs2))]
    
    
    if len(all_pairs) > sample_pairs:
        all_pairs = random.sample(all_pairs, sample_pairs)
    
    print(f"calculating MMD for {len(all_pairs)} graph pairs...")
    
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        func = partial(process_graph_pair, all_features1=all_features1, all_features2=all_features2)
        all_results = list(tqdm(pool.imap(func, all_pairs), total=len(all_pairs), desc="calculating MMD"))
    
    
    final_results = {}
    for key in all_results[0].keys():
        if isinstance(all_results[0][key], dict):
            final_results[key] = {}
            for subkey in all_results[0][key].keys():
                final_results[key][subkey] = np.mean([r[key][subkey] for r in all_results])
        else:
            final_results[key] = np.mean([r[key] for r in all_results])
    
    end_time = time.time()
    print(f"total calculation time: {end_time - start_time:.2f} seconds")
    
    return final_results

def main():
    import argparse
    
    folder1 = 'graphml_files'
    

    folder2 = 'selected_skeletons'
    
    results = compare_graph_folders(folder1, folder2, 
                                   sample_graphs=24, sample_pairs=100)
    
    print("\ngraph structure comparison results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue:.6f}")
        else:
            print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    main()