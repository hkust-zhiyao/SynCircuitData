

import os
import json
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm

def load_graph_data(graphml_path, json_path, text_dim=300):
    
    pass

def load_all_graphs(folder_dir, text_dim=300):
    
    pass

def copy_design_files(folder):
    
    pass

def save_node_emb_with_func(combined_data, graph_embs, output_file):
    
    pass

def load_graph_data_pyg(graphml_path):
    
    
    graph_name = os.path.splitext(os.path.basename(graphml_path))[0]
    
    G = nx.read_graphml(graphml_path)
    
    
    source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    source_node = source_nodes[0] if source_nodes else list(G.nodes())[0]
    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    sink_node = sink_nodes[0] if sink_nodes else list(G.nodes())[0]
    
    
    try:
        source_path_dict = nx.single_source_shortest_path_length(G, source_node)
    except Exception:
        source_path_dict = {n: -1 for n in G.nodes()}
    
    try:
        G_rev = G.reverse(copy=True)
        sink_path_dict = nx.single_source_shortest_path_length(G_rev, sink_node)
    except Exception:
        sink_path_dict = {n: -1 for n in G.nodes()}
    
    
    width_list = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]
    round_list = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32]
    
    node_names = []
    tpe_list = []
    node_features = []
    
    for node, attr in G.nodes(data=True):
        
        node_names.append(node)
        
        tpe_list.append(attr.get("tpe", ""))
        
        
        try:
            width_val = int(attr.get("width", 0))
        except:
            width_val = 0
        def round_width(val):
            try:
                val = float(val)
            except:
                val = 1
            if val < 0:
                val = 1
            return min(width_list, key=lambda x: abs(x - val))
        width = round_width(width_val)
        width_idx = width_list.index(width)
        width_onehot = [0] * len(width_list)
        width_onehot[width_idx] = 1
        
        
        def round_value(val):
            try:
                val = float(val)
            except:
                val = 0
            if val < 0:
                val = 0
            return min(round_list, key=lambda x: abs(x - val))
        
        rounded_source = round_value(source_path_dict.get(node, -1))
        rounded_sink = round_value(sink_path_dict.get(node, -1))
        source_onehot = [0] * len(round_list)
        sink_onehot = [0] * len(round_list)
        try:
            source_index = round_list.index(rounded_source)
        except Exception:
            source_index = 0
        try:
            sink_index = round_list.index(rounded_sink)
        except Exception:
            sink_index = 0
        source_onehot[source_index] = 1
        sink_onehot[sink_index] = 1
        pos_encoding = source_onehot + sink_onehot
        
        combined_feature = width_onehot + pos_encoding  
        
        node_features.append(combined_feature)
    
    
    new_G = nx.DiGraph()
    new_G.add_nodes_from(list(G.nodes()))
    new_G.add_edges_from(G.edges())
    data_obj = from_networkx(new_G)
    data_obj.x = torch.tensor(node_features, dtype=torch.float)
    data_obj.node_names = node_names
    data_obj.tpe_list = tpe_list
    return data_obj

