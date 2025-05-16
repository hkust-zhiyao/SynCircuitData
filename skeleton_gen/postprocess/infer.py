import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
import glob
from collections import defaultdict, Counter
from tqdm import tqdm
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import get_laplacian, to_dense_adj
import random
from scipy.stats import wasserstein_distance
from scipy.linalg import eigvalsh


from train import TypeGNNEncoder, WidthGNNEncoder, EdgeGNNEncoder, NodeTypePredictor, NodeWidthPredictor, EdgeDirectionPredictor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EnhancedPositionalEncoding:
    def __init__(self, dim=16):
        self.dim = dim
    
    def compute_node2vec_embeddings(self, edge_index, num_nodes, embedding_dim=16):
        
        model = Node2Vec(
            edge_index, 
            embedding_dim=embedding_dim, 
            walk_length=10,
            context_size=5, 
            walks_per_node=10,
            num_negative_samples=1, 
            p=1, 
            q=1, 
            sparse=True
        ).to(device)
        
        loader = model.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        
        model.train()
        for epoch in range(5):  
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                
        
        with torch.no_grad():
            node_embeddings = model().detach()
            
        return node_embeddings
    
    def compute_positional_encodings(self, data):
        
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = edge_index.device  
        
        
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', 
                                                num_nodes=num_nodes)
        lap = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        
        
        try:
            
            eigenvalues, eigenvectors = torch.linalg.eigh(lap)
            
            pe = eigenvectors[:, 1:self.dim+1]  
            if pe.shape[1] < self.dim:  
                padding = torch.zeros((num_nodes, self.dim - pe.shape[1]), device=device)
                pe = torch.cat([pe, padding], dim=1)
        except:
            
            pe = torch.randn(num_nodes, self.dim, device=device)
            
        return pe
    
    def apply_to_data(self, data):
        
        
        data_device = data.x.device
        
        
        node2vec_emb = self.compute_node2vec_embeddings(
            data.edge_index, data.num_nodes, self.dim
        )
        
        
        laplacian_pe = self.compute_positional_encodings(data)
        
        
        node2vec_emb = node2vec_emb.to(data_device)
        laplacian_pe = laplacian_pe.to(data_device)
        
        
        if len(node2vec_emb) != data.num_nodes:
            print(f"Warning: Node2Vec embedding size ({len(node2vec_emb)}) does not match node count ({data.num_nodes}), adjusting...")
            corrected_emb = torch.zeros((data.num_nodes, self.dim), device=data_device)
            
            min_nodes = min(len(node2vec_emb), data.num_nodes)
            corrected_emb[:min_nodes] = node2vec_emb[:min_nodes]
            node2vec_emb = corrected_emb
        
        
        if len(laplacian_pe) != data.num_nodes:
            print(f"Warning: Laplacian encoding size ({len(laplacian_pe)}) does not match node count ({data.num_nodes}), adjusting...")
            corrected_pe = torch.zeros((data.num_nodes, self.dim), device=data_device)
            min_nodes = min(len(laplacian_pe), data.num_nodes)
            corrected_pe[:min_nodes] = laplacian_pe[:min_nodes]
            laplacian_pe = corrected_pe
        
        
        enhanced_x = torch.cat([data.x, node2vec_emb, laplacian_pe], dim=1)
            
        data.x = enhanced_x
        return data


def convert_undirected_graph_to_pyg(G):
    
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    
    
    x = torch.ones((len(G.nodes()), 1), dtype=torch.float)
    
    
    edge_index = []
    node_pairs = []  
    
    for src, dst in G.edges():
        i, j = node_indices[src], node_indices[dst]
        edge_index.append([i, j])
        edge_index.append([j, i])  
        node_pairs.append((src, dst))
    
    if len(edge_index) == 0:
        return None, None, None
    
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    
    data = Data(x=x, edge_index=edge_index)
    
    return data, node_indices, node_pairs


def validate_constraints(G, node_types):
    
    if not G:
        return False
    
    for node, node_type in node_types.items():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        
        if node_type == 'input' and in_degree > 0:
            return False
        
        
        if node_type == 'output' and out_degree > 0:
            return False
    
    return True

def apply_constraints(G, node_types):
    
    G_new = G.copy()
    changes = 0
    
    
    for node, type_val in node_types.items():
        if type_val == 'input':
            in_edges = list(G_new.in_edges(node))
            for u, v in in_edges:
                G_new.remove_edge(u, v)
                G_new.add_edge(v, u)
                changes += 1
    
    
    for node, type_val in node_types.items():
        if type_val == 'output':
            out_edges = list(G_new.out_edges(node))
            for u, v in out_edges:
                G_new.remove_edge(u, v)
                G_new.add_edge(v, u)
                changes += 1
    
    return G_new, changes


def compute_laplacian_spectrum(G, k=20):
    
    
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
        
    
    L = nx.normalized_laplacian_matrix(G_undirected).todense()
    
    
    eigvals = np.real(eigvalsh(L))
    
    
    if len(eigvals) < k:
        return np.pad(eigvals, (0, k - len(eigvals)))
    else:
        return eigvals[:k]


def compute_graph_similarity(G1, G2, k=20):
    
    spectrum1 = compute_laplacian_spectrum(G1, k)
    spectrum2 = compute_laplacian_spectrum(G2, k)
    
    
    distance = wasserstein_distance(spectrum1, spectrum2)
    
    return distance


def load_reference_graphs(ref_dir):
    
    ref_graphs = []
    
    for file_path in tqdm(glob.glob(os.path.join(ref_dir, "*.graphml")), desc="Loading reference graphs"):
        try:
            
            G_directed = nx.read_graphml(file_path)
            
            
            node_types = nx.get_node_attributes(G_directed, 'type')
            node_widths = nx.get_node_attributes(G_directed, 'width')
            
            
            type_counter = Counter(node_types.values())
            total_nodes = len(node_types)
            type_ratio = {t: count/total_nodes for t, count in type_counter.items()}
            
            
            width_classes = defaultdict(int)
            for w in node_widths.values():
                if isinstance(w, str):
                    w = int(w)
                width_idx, _ = map_width_to_class(w)
                width_classes[width_idx] += 1
            
            width_ratio = {w: count/total_nodes for w, count in width_classes.items()}
            
            
            G_undirected = G_directed.to_undirected()
            
            ref_graphs.append({
                'name': os.path.basename(file_path),
                'graph_directed': G_directed,
                'graph_undirected': G_undirected,
                'type_ratio': type_ratio,
                'width_ratio': width_ratio,
                'total_nodes': total_nodes
            })
            
        except Exception as e:
            print(f"Failed to load reference graph {file_path}: {e}")
    
    print(f"Successfully loaded {len(ref_graphs)} reference graphs")
    return ref_graphs


def find_most_similar_reference(G_undirected, ref_graphs):

    best_similarity = float('inf')
    best_ref = None
    
    for ref in ref_graphs:
        similarity = compute_graph_similarity(G_undirected, ref['graph_undirected'])
        if similarity < best_similarity:
            best_similarity = similarity
            best_ref = ref
    
    return best_ref, best_similarity


def assign_node_types(type_probs, ref_type_ratio, num_nodes):

    
    for t in ['input', 'output', 'reg']:
        if t not in ref_type_ratio:
            ref_type_ratio[t] = 0.0
    
    
    target_inputs = max(1, int(ref_type_ratio.get('input', 0) * num_nodes * (1 + random.uniform(-0.5, 0.5))))
    target_outputs = max(1, int(ref_type_ratio.get('output', 0) * num_nodes * (1 + random.uniform(-0.05, 0.05))))
    
    
    if target_inputs + target_outputs > num_nodes:
        ratio = num_nodes / (target_inputs + target_outputs) * 0.9  
        target_inputs = max(1, int(target_inputs * ratio))
        target_outputs = max(1, int(target_outputs * ratio))
    
    
    input_probs = type_probs[:, 0].cpu().numpy()  
    input_indices = np.argsort(-input_probs)[:target_inputs]  
    
    
    output_probs = type_probs[:, 1].cpu().numpy()  
    
    mask = np.ones(num_nodes, dtype=bool)
    mask[input_indices] = False
    masked_output_probs = output_probs.copy()
    masked_output_probs[~mask] = -1  
    
    output_indices = np.argsort(-masked_output_probs)[:target_outputs]  
    
    
    assigned_indices = set(input_indices) | set(output_indices)
    reg_indices = [i for i in range(num_nodes) if i not in assigned_indices]
    
    
    node_types = np.zeros(num_nodes, dtype=int)
    node_types[input_indices] = 0  
    node_types[output_indices] = 1  
    node_types[reg_indices] = 2     
    
    return node_types

def assign_node_widths(width_probs, ref_width_ratio, num_nodes, width_classes):

    width_probs_np = width_probs.cpu().numpy()
    assigned_widths = np.zeros(num_nodes, dtype=int)
    
    
    remaining_indices = set(range(num_nodes))
    
    
    for width_idx, width_val in enumerate(width_classes):
        if width_idx not in ref_width_ratio:
            continue
            
        
        target_count = int(ref_width_ratio[width_idx] * num_nodes * (1 + random.uniform(-0.1, 0.1)))
        
        if not remaining_indices or target_count <= 0:
            continue
            
        
        current_probs = width_probs_np[:, width_idx].copy()
        
        
        for idx in range(num_nodes):
            if idx not in remaining_indices:
                current_probs[idx] = -1
                
        
        selected = np.argsort(-current_probs)[:min(target_count, len(remaining_indices))]
        
        
        for idx in selected:
            assigned_widths[idx] = width_idx
            remaining_indices.remove(idx)
    
    
    for idx in remaining_indices:
        
        assigned_widths[idx] = np.argmax(width_probs_np[idx])
    
    return assigned_widths

def map_width_to_class(width):
    
    from train import WIDTH_CLASSES
    width_array = np.array(WIDTH_CLASSES)
    
    if isinstance(width, str):
        width = int(width)
        
    idx = np.argmin(np.abs(width_array - width))
    return idx, WIDTH_CLASSES[idx]

def inference(input_dir, output_dir, ref_dir="graphml_files"):

    print("Loading model and reference graphs...")
    
    ref_graphs = load_reference_graphs(ref_dir)
    
    checkpoint = torch.load('best_graph_model.pt', map_location=device)
    
    pe_dim = checkpoint.get('pe_dim', 16)
    width_classes = checkpoint.get('width_classes', [1, 2, 4, 8, 16, 32, 64, 128, 256])
    
    in_channels = 1 + 2 * pe_dim 
    hidden_channels = 32
    out_channels = 32
    

    type_encoder = TypeGNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    width_encoder = WidthGNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    edge_encoder = EdgeGNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    
    type_predictor = NodeTypePredictor(out_channels, hidden_channels).to(device)
    width_predictor = NodeWidthPredictor(out_channels, hidden_channels).to(device)
    edge_predictor = EdgeDirectionPredictor(out_channels, hidden_channels).to(device)
    
    type_encoder.load_state_dict(checkpoint['type_encoder'])
    width_encoder.load_state_dict(checkpoint['width_encoder'])
    edge_encoder.load_state_dict(checkpoint['edge_encoder'])
    type_predictor.load_state_dict(checkpoint['type_predictor'])
    width_predictor.load_state_dict(checkpoint['width_predictor'])
    edge_predictor.load_state_dict(checkpoint['edge_predictor'])
    
    type_encoder.eval()
    width_encoder.eval()
    edge_encoder.eval()
    type_predictor.eval()
    width_predictor.eval()
    edge_predictor.eval()
    
    pe_generator = EnhancedPositionalEncoding(dim=pe_dim)
    
    os.makedirs(output_dir, exist_ok=True)
    

    for file_path in tqdm(glob.glob(os.path.join(input_dir, "*.graphml")), desc="Processing graph files"):
        filename = os.path.basename(file_path)
        print(f"\nProcessing file: {filename}")
        
        G_undirected = nx.read_graphml(file_path)
        
        best_ref, similarity = find_most_similar_reference(G_undirected, ref_graphs)
        print(f"Found the most similar reference graph: {best_ref['name']}, similarity distance: {similarity:.4f}")
        

        pyg_data, node_indices, node_pairs = convert_undirected_graph_to_pyg(G_undirected)
        if pyg_data is None:
            print(f"Skipping empty graph: {filename}")
            continue
        
        pyg_data = pe_generator.apply_to_data(pyg_data)
        
        reverse_indices = {i: node for node, i in node_indices.items()}
        
        with torch.no_grad():
            pyg_data = pyg_data.to(device)
            
            type_embeddings = type_encoder(pyg_data.x, pyg_data.edge_index)
            type_logits = type_predictor(type_embeddings)
            type_probs = F.softmax(type_logits, dim=1)
            
            width_embeddings = width_encoder(pyg_data.x, pyg_data.edge_index)
            width_logits = width_predictor(width_embeddings)
            width_probs = F.softmax(width_logits, dim=1)
            
            node_types_pred = assign_node_types(
                type_probs, 
                best_ref['type_ratio'],
                pyg_data.num_nodes
            )
            
            node_width_classes_pred = assign_node_widths(
                width_probs,
                best_ref['width_ratio'],
                pyg_data.num_nodes,
                width_classes
            )
            
            G_directed = nx.DiGraph()
            
            for idx in range(len(node_types_pred)):
                node = reverse_indices[idx]
                
                type_idx = node_types_pred[idx]
                if type_idx == 0:
                    node_type = 'input'
                elif type_idx == 1:
                    node_type = 'output'
                else:
                    node_type = 'reg'
                
                width_idx = node_width_classes_pred[idx]
                base_width = width_classes[width_idx]
                min_variation = max(1, int(base_width * 0.8))
                max_variation = int(base_width * 1.2)
                
                width = random.randint(min_variation, max_variation)
                
                
                G_directed.add_node(node, type=node_type, width=width)
            
            edge_embeddings = edge_encoder(pyg_data.x, pyg_data.edge_index)
            

            for src, dst in node_pairs:
                src_idx, dst_idx = node_indices[src], node_indices[dst]
                

                src_embed = edge_embeddings[src_idx]
                dst_embed = edge_embeddings[dst_idx]
                
                src_embed = src_embed.unsqueeze(0)
                dst_embed = dst_embed.unsqueeze(0)
                

                prob_src_to_dst = edge_predictor(src_embed, dst_embed).item()

                prob_dst_to_src = edge_predictor(dst_embed, src_embed).item()
                

                if prob_src_to_dst > prob_dst_to_src:
                    G_directed.add_edge(src, dst)
                else:
                    G_directed.add_edge(dst, src)
        
        is_valid = True
        if is_valid:
            
            output_path = os.path.join(output_dir, filename)
            nx.write_graphml(G_directed, output_path)
            print(f"Successfully processed and saved: {output_path}")
        else:
            print(f"Warning: The graph {filename} cannot satisfy all constraints")

if __name__ == "__main__":

    input_directory = "collected_graphs"
    output_directory = "proccessed_graphs"
    ref_directory = "graphml_files"
    
    inference(input_directory, output_directory, ref_directory)