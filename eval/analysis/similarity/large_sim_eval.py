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
            print(f"failed to load file {file_path}: {e}")
    
    print(f"successfully loaded {len(graphs)} graphs within node number limit")
    return graphs

def compute_netrd_graph_distance(G1, G2, method='nbd'):
    
    try:
        import netrd
        if method == 'nbd':
            dist_obj = netrd.distance.NetSimile()
        elif method == 'resistance':
            dist_obj = netrd.distance.ResistancePerturbation()
        elif method == 'portrait':
            dist_obj = netrd.distance.PortraitDivergence()
        else:
            dist_obj = netrd.distance.NetSimile()  
        
        
        G1_ud = G1.to_undirected() if G1.is_directed() else G1
        G2_ud = G2.to_undirected() if G2.is_directed() else G2
        
        return dist_obj.dist(G1_ud, G2_ud)
    except ImportError:
        print("need to install netrd library: pip install netrd")
        return None

def compute_graph_histogram_signatures(G):
    
    features = {}
    
    
    in_degrees = np.array([d for n, d in G.in_degree()])
    out_degrees = np.array([d for n, d in G.out_degree()])
    
    
    max_in_degree = max(in_degrees) if len(in_degrees) > 0 else 0
    max_out_degree = max(out_degrees) if len(out_degrees) > 0 else 0
    
    
    in_bins = np.logspace(0, np.log10(max(max_in_degree, 1) + 1), 10)
    out_bins = np.logspace(0, np.log10(max(max_out_degree, 1) + 1), 10)
    
    in_hist, _ = np.histogram(in_degrees, bins=in_bins, density=True)
    out_hist, _ = np.histogram(out_degrees, bins=out_bins, density=True)
    
    features['in_degree_hist'] = in_hist
    features['out_degree_hist'] = out_hist
    
    
    sample_size = min(1000, len(G.nodes))
    sampled_nodes = random.sample(list(G.nodes), sample_size)
    
    
    clustering_coefs = []
    for node in sampled_nodes:
        try:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                clustering_coefs.append(nx.clustering(G, node))
        except:
            pass
    
    if clustering_coefs:
        clustering_hist, _ = np.histogram(clustering_coefs, bins=10, range=(0, 1), density=True)
        features['clustering_hist'] = clustering_hist
    else:
        features['clustering_hist'] = np.zeros(10)
    
    
    try:
        
        scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
        if scc_sizes:
            largest_scc = max(scc_sizes)
            scc_ratio = largest_scc / len(G.nodes)
            features['scc_ratio'] = np.array([scc_ratio])
            
            
            if len(scc_sizes) > 1:
                scc_bins = np.logspace(0, np.log10(largest_scc + 1), 10)
                scc_hist, _ = np.histogram(scc_sizes, bins=scc_bins, density=True)
                features['scc_size_hist'] = scc_hist
            else:
                features['scc_size_hist'] = np.zeros(10)
        else:
            features['scc_ratio'] = np.array([0])
            features['scc_size_hist'] = np.zeros(10)
    except:
        features['scc_ratio'] = np.array([0])
        features['scc_size_hist'] = np.zeros(10)
    
    try:
        pageranks = nx.pagerank(G, alpha=0.85)
        pagerank_values = list(pageranks.values())
        
        if pagerank_values:
            min_pr = min(pagerank_values)
            max_pr = max(pagerank_values)
            pr_bins = np.logspace(np.log10(max(min_pr, 1e-10)), np.log10(max(max_pr, 1e-9)), 10)
            pr_hist, _ = np.histogram(pagerank_values, bins=pr_bins, density=True)
            features['pagerank_hist'] = pr_hist
        else:
            features['pagerank_hist'] = np.zeros(10)
    except:
        features['pagerank_hist'] = np.zeros(10)
    
    features['density'] = np.array([nx.density(G)])
    features['reciprocity'] = np.array([nx.reciprocity(G)]) if G.is_directed() else np.array([0])
    
    
    sources = np.sum(in_degrees == 0) / len(G.nodes)
    sinks = np.sum(out_degrees == 0) / len(G.nodes)
    features['terminal_ratios'] = np.array([sources, sinks])
    
    return features

def compute_histogram_distance(hist1, hist2):
    
    
    max_len = max(len(hist1), len(hist2))
    h1 = np.zeros(max_len)
    h2 = np.zeros(max_len)
    h1[:len(hist1)] = hist1
    h2[:len(hist2)] = hist2
    
    
    from scipy.spatial.distance import jensenshannon
    
    if np.sum(h1) == 0:
        h1 += 1e-10
    if np.sum(h2) == 0:
        h2 += 1e-10
        
    
    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)
    
    return jensenshannon(h1, h2)

def compute_histogram_signatures_distance(features1, features2):
    
    results = {}
    
    
    
    results['in-degree distribution distance'] = compute_histogram_distance(features1['in_degree_hist'], features2['in_degree_hist'])
    results['out-degree distribution distance'] = compute_histogram_distance(features1['out_degree_hist'], features2['out_degree_hist'])
    results['clustering coefficient distribution distance'] = compute_histogram_distance(features1['clustering_hist'], features2['clustering_hist'])
    # results['scc size distribution distance'] = compute_histogram_distance(features1['scc_size_hist'], features2['scc_size_hist'])
    
    
    # results['density difference'] = np.abs(features1['density'] - features2['density'])[0]
    # results['reciprocity difference'] = np.abs(features1['reciprocity'] - features2['reciprocity'])[0]
    # results['scc ratio difference'] = np.abs(features1['scc_ratio'] - features2['scc_ratio'])[0]
    
    
    term_dist = np.linalg.norm(features1['terminal_ratios'] - features2['terminal_ratios'])
    results['terminal node distribution distance'] = term_dist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return results

def graph_sampling(G, sample_nodes=1000, sample_type='random'):
    
    if len(G.nodes) <= sample_nodes:
        return G
    
    if sample_type == 'random':
        
        nodes = random.sample(list(G.nodes), sample_nodes)
        return G.subgraph(nodes)
    elif sample_type == 'snowball':
        
        start_node = random.choice(list(G.nodes))
        sampled_nodes = set([start_node])
        frontier = set([start_node])
        
        while len(sampled_nodes) < sample_nodes and frontier:
            current = frontier.pop()
            neighbors = set(G.neighbors(current)) - sampled_nodes
            if len(neighbors) > 0:
                
                neighbors_to_add = random.sample(list(neighbors), 
                                               min(10, len(neighbors)))
                frontier.update(neighbors_to_add)
                sampled_nodes.update(neighbors_to_add)
                
            if len(sampled_nodes) >= sample_nodes:
                break
                
            if not frontier and len(sampled_nodes) < sample_nodes:
                
                remaining = set(G.nodes) - sampled_nodes
                if remaining:
                    new_start = random.choice(list(remaining))
                    frontier.add(new_start)
                    sampled_nodes.add(new_start)
        
        return G.subgraph(list(sampled_nodes))
    else:
        return G

def process_large_graph_comparison(G1, G2, method='histogram'):
    
    
    sample_size = min(5000, min(len(G1.nodes), len(G2.nodes)))
    
    if method == 'netrd':
        
        G1_sample = graph_sampling(G1, sample_nodes=sample_size)
        G2_sample = graph_sampling(G2, sample_nodes=sample_size)
        return compute_netrd_graph_distance(G1_sample, G2_sample)
    
    elif method == 'histogram':
        
        features1 = compute_graph_histogram_signatures(G1)
        features2 = compute_graph_histogram_signatures(G2)
        return compute_histogram_signatures_distance(features1, features2)
    
    elif method == 'spectral':
        
        
        from scipy.linalg import eigvalsh
        
        G1_sample = graph_sampling(G1, sample_nodes=sample_size)
        G2_sample = graph_sampling(G2, sample_nodes=sample_size)
        
        L1 = nx.normalized_laplacian_matrix(G1_sample.to_undirected()).todense()
        L2 = nx.normalized_laplacian_matrix(G2_sample.to_undirected()).todense()
        
        k = min(50, L1.shape[0]-1, L2.shape[0]-1)
        if k <= 0:
            return {"谱距离": 1.0}  
            
        
        eigs1 = eigvalsh(L1)[:k+1]
        eigs2 = eigvalsh(L2)[:k+1]
        
        
        max_len = max(len(eigs1), len(eigs2))
        padded_eigs1 = np.zeros(max_len)
        padded_eigs2 = np.zeros(max_len)
        padded_eigs1[:len(eigs1)] = eigs1
        padded_eigs2[:len(eigs2)] = eigs2
        
        
        spectral_dist = np.linalg.norm(padded_eigs1 - padded_eigs2) / max_len
        return {"谱距离": spectral_dist}

def compare_large_graph_folders(folder1, folder2, sample_graphs=10, method='histogram'):
    
    start_time = time.time()
    
    
    print("加载第一个文件夹中的图...")
    graphs1 = load_graphml_files_with_size_limit(folder1, sample_size=sample_graphs, max_nodes=50000)
    print("加载第二个文件夹中的图...")
    graphs2 = load_graphml_files_with_size_limit(folder2, sample_size=sample_graphs, max_nodes=50000)
    
    if not graphs1 or not graphs2:
        return "至少一个文件夹中没有有效的 GraphML 文件"
    
    
    all_results = []
    total_pairs = len(graphs1) * len(graphs2)
    pair_counter = 0
    
    print(f"计算 {total_pairs} 对图的距离...")
    for i, G1 in enumerate(graphs1):
        for j, G2 in enumerate(graphs2):
            pair_counter += 1
            print(f"处理图对 {pair_counter}/{total_pairs}：{len(G1.nodes)} 节点与 {len(G2.nodes)} 节点")
            
            
            result = process_large_graph_comparison(G1, G2, method=method)
            all_results.append(result)
    
    
    final_results = {}
    for key in all_results[0].keys():
        final_results[key] = np.mean([r[key] for r in all_results])
    
    end_time = time.time()
    print(f"总计算时间: {end_time - start_time:.2f} 秒")
    
    return final_results

def main():
    import argparse
    
    folder1 = '/home/sliudx/project/rtl_aug/nips-circuitgen/analysis/graph_sim_eval/real_asts'
    
    folder2 = '/home/sliudx/project/rtl_aug/nips-circuitgen/59-gen/511_samples_ast/ast'
    
    results = compare_large_graph_folders(folder1, folder2, 
                                       sample_graphs=20, method='spectral')
    
    print("\n图结构比较结果:")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    main()
