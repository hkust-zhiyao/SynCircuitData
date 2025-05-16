import os
import networkx as nx
import glob
from tqdm import tqdm
import copy
import numpy as np

def remove_isolated_nodes(G):
    
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return G, isolated_nodes

def check_node_constraints(G):
    
    node_types = nx.get_node_attributes(G, 'type')
    violations = {
        'input_with_parents': [],
        'output_without_children': [],
        'reg_without_parents': [],
        'reg_without_children': []
    }
    
    for node, node_type in node_types.items():
        if node_type == 'input':
            
            parents = list(G.predecessors(node))
            if parents:
                violations['input_with_parents'].append((node, parents))
        
        elif node_type == 'output':
            
            children = list(G.successors(node))
            if children:
                violations['output_without_children'].append((node, children))
        
        elif node_type == 'reg':
            
            parents = list(G.predecessors(node))
            children = list(G.successors(node))
            
            if not parents:
                violations['reg_without_parents'].append(node)
            
            if not children:
                violations['reg_without_children'].append(node)
    
    return violations

def calculate_cost(violations):
    
    total_violations = (
        len(violations['input_with_parents']) + 
        len(violations['output_without_children']) + 
        len(violations['reg_without_parents']) + 
        len(violations['reg_without_children'])
    )
    return total_violations

def swap_node_types(G, node1, node2):
    
    node_types = nx.get_node_attributes(G, 'type')
    type1, type2 = node_types[node1], node_types[node2]
    
    
    nx.set_node_attributes(G, {node1: type2, node2: type1}, 'type')
    return G

def fix_graph_optimally(G):
    
    original_G = copy.deepcopy(G)
    node_types = nx.get_node_attributes(G, 'type')
    
    
    input_nodes = [n for n, t in node_types.items() if t == 'input']
    output_nodes = [n for n, t in node_types.items() if t == 'output']
    reg_nodes = [n for n, t in node_types.items() if t == 'reg']
    
    
    violations = check_node_constraints(G)
    initial_cost = calculate_cost(violations)
    
    if initial_cost == 0:
        return G, 0  
    
    
    best_G = None
    min_edge_changes = float('inf')
    
    
    all_nodes = list(G.nodes())
    node_pairs = [(n1, n2) for i, n1 in enumerate(all_nodes) for n2 in all_nodes[i+1:]]
    
    for node1, node2 in node_pairs:
        
        if node_types[node1] == node_types[node2]:
            continue
        
        
        test_G = copy.deepcopy(G)
        test_G = swap_node_types(test_G, node1, node2)
        
        
        fixed_G, edge_changes = fix_edges(test_G)
        
        
        new_violations = check_node_constraints(fixed_G)
        if calculate_cost(new_violations) == 0 and edge_changes < min_edge_changes:
            min_edge_changes = edge_changes
            best_G = fixed_G
    
    
    if best_G is None:
        best_G, min_edge_changes = fix_edges(original_G)
    
    return best_G, min_edge_changes

def fix_edges(G):
    
    node_types = nx.get_node_attributes(G, 'type')
    edge_changes = 0
    
    
    fixed_G = copy.deepcopy(G)
    
    
    for node, node_type in node_types.items():
        if node_type == 'input':
            parents = list(fixed_G.predecessors(node))
            for parent in parents:
                fixed_G.remove_edge(parent, node)
                edge_changes += 1
    
    
    for node, node_type in node_types.items():
        if node_type == 'output':
            children = list(fixed_G.successors(node))
            for child in children:
                fixed_G.remove_edge(node, child)
                edge_changes += 1
    
    
    for node, node_type in node_types.items():
        if node_type == 'reg':
            parents = list(fixed_G.predecessors(node))
            children = list(fixed_G.successors(node))
            
            
            if not parents:
                
                potential_parents = [n for n, t in node_types.items() 
                                    if (t == 'reg' or t == 'input') and n != node]
                if potential_parents:
                    
                    parent = potential_parents[0]
                    fixed_G.add_edge(parent, node)
                    edge_changes += 1
            
            
            if not children:
                
                potential_children = [n for n, t in node_types.items() 
                                     if (t == 'reg' or t == 'output') and n != node]
                if potential_children:
                    
                    child = potential_children[0]
                    fixed_G.add_edge(node, child)
                    edge_changes += 1
    
    return fixed_G, edge_changes

def process_all_graphs(input_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    graphml_files = glob.glob(os.path.join(input_dir, "*.graphml"))
    
    total_files = len(graphml_files)
    processed_files = 0
    fixed_files = 0
    
    for file_path in tqdm(graphml_files, desc="processing graphs"):
        filename = os.path.basename(file_path)
        
        try:
            
            G = nx.read_graphml(file_path)
            
            
            G, isolated_nodes = remove_isolated_nodes(G)
            if isolated_nodes:
                print(f"removed {len(isolated_nodes)} isolated nodes from {filename}")
            
            
            fixed_G, edge_changes = fix_graph_optimally(G)
            
            
            output_path = os.path.join(output_dir, filename)
            nx.write_graphml(fixed_G, output_path)
            
            processed_files += 1
            if edge_changes > 0:
                fixed_files += 1
                print(f"fixed {filename}, changed {edge_changes} edges")
            
        except Exception as e:
            print(f"error when processing {filename}: {str(e)}")
    
    print(f"processed! {processed_files}/{total_files} graphs, fixed {fixed_files} graphs.")

if __name__ == "__main__":
    input_directory = "proccessed_graphs"
    output_directory = "valid_graphs"
    
    process_all_graphs(input_directory, output_directory)
