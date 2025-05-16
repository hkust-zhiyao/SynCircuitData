import numpy as np
import scipy.sparse as sp
import networkx as nx


import numpy as np
import scipy.sparse as sp
import networkx as nx
import powerlaw


def max_degree(A):
    
    try:
        degrees = A.sum(axis=-1)
        return np.max(degrees)
    except:
        return np.nan


def min_degree(A):
    
    try:
        degrees = A.sum(axis=-1)
        return np.min(degrees)
    except:
        return np.nan


def average_degree(A):
    
    try:
        degrees = A.sum(axis=-1)
        return np.mean(degrees)
    except:
        return np.nan


def LCC(A):
    
    try:
        G = nx.from_scipy_sparse_matrix(A)
        return max([len(c) for c in nx.connected_components(G)])
    except:
        return np.nan


def wedge_count(A):
    
    try:
        degrees = np.array(A.sum(axis=-1))
        return 0.5 * np.dot(degrees.T, degrees - 1).reshape([]) 
    except:
        return np.nan


def claw_count(A):
    
    try:
        degrees = np.array(A.sum(axis=-1))
        return 1 / 6 * np.sum(degrees * (degrees - 1) * (degrees - 2))
    except:
        return np.nan


def triangle_count(A):
    
    try:
        A_graph = nx.from_scipy_sparse_matrix(A)
        triangles = nx.triangles(A_graph)
        t = np.sum(list(triangles.values())) / 3
        return int(t)
    except:
        return np.nan


def square_count(A):
    
    try:
        A_squared = A @ A
        common_neighbors = sp.triu(A_squared, k=1).tocsr()
        num_common_neighbors = np.array(
            common_neighbors[common_neighbors.nonzero()]
        ).reshape(-1)
        return np.dot(num_common_neighbors, num_common_neighbors - 1) / 4
    except:
        return np.nan


def power_law_alpha(A):
    
    try:
        degrees = np.array(A.sum(axis=-1)).flatten()
        return powerlaw.Fit(
            degrees, xmin=max(np.min(degrees), 1), verbose=False
        ).power_law.alpha
    except:
        return np.nan


def gini(A):
    
    try:
        N = A.shape[0]
        degrees_sorted = np.sort(np.array(A.sum(axis=-1)).flatten())
        return (
            2 * np.dot(degrees_sorted, np.arange(1, N + 1)) / (N * np.sum(degrees_sorted))
            - (N + 1) / N
        )
    except:
        return np.nan


def edge_distribution_entropy(A):
    
    try:
        N = A.shape[0]
        degrees = np.array(A.sum(axis=-1)).flatten()
        degrees /= degrees.sum()
        return -np.dot(np.log(degrees), degrees) / np.log(N)
    except:
        return np.nan


def assortativity(A):
    
    try:
        G = nx.from_scipy_sparse_matrix(A)
        return nx.degree_assortativity_coefficient(G)
    except:
        return np.nan


def clustering_coefficient(A):
    
    try:
        return 3 * triangle_count(A) / wedge_count(A)   
    except:
        return np.nan


def cpl(A):
    
    try:
        P = sp.csgraph.shortest_path(A)
        return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()
    except:
        return np.nan


def compute_graph_statistics(A):
    
    statistics = {"d_max": max_degree(A),
                  "d_min": min_degree(A),
                  "d": average_degree(A),
                  "LCC": LCC(A),
                  "wedge_count": wedge_count(A),
                  "claw_count": claw_count(A),
                  "triangle_count": triangle_count(A),
                  "square_count": square_count(A),
                  "power_law_exp": power_law_alpha(A),
                  "gini": gini(A),
                  "rel_edge_distr_entropy": edge_distribution_entropy(A),
                  "assortativity": assortativity(A),
                  "clustering_coefficient": clustering_coefficient(A),
                  "cpl": cpl(A)} 
    return statistics
    
def edge_overlap(A, B):
    
    try:
        return A.multiply(B).sum() / 2
    except:
        return np.nan