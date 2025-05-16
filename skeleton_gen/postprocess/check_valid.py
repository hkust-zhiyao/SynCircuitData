import os
import glob
import networkx as nx
from tqdm import tqdm

def validate_constraints(G, node_types):
    for node in G.nodes():
        
        if G.degree(node) == 0:
            return False, f"Node {node} is isolated"
        
        node_type = node_types.get(node)
        
        
        if node_type == 'input':
            if list(G.predecessors(node)):
                return False, f"Input node {node} has a parent node"
                
        
        elif node_type == 'output':
            if list(G.successors(node)):
                return False, f"Output node {node} has a child node"
                
        
        elif node_type == 'reg':
            if not list(G.predecessors(node)):
                return False, f"reg node {node} has no parent node"
            if not list(G.successors(node)):
                return False, f"reg node {node} has no child node"
    
    return True, "The graph satisfies all constraints"

def process_graphs(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
  
    graphml_files = glob.glob(os.path.join(input_dir, "*.graphml"))
    
    valid_count = 0
    invalid_count = 0
    
    print(f"Processing {len(graphml_files)} graph files...")
    
    for file_path in tqdm(graphml_files, desc="Validating graph files"):
        filename = os.path.basename(file_path)
        
        try:

            G = nx.read_graphml(file_path)
            
            node_types = nx.get_node_attributes(G, 'type')
            
            is_valid, message = validate_constraints(G, node_types)
            
            if is_valid:
                output_filename = f"skeleton_{valid_count}.graphml"
                output_path = os.path.join(output_dir, output_filename)
                
                for i, node in enumerate(G.nodes()):
                    G.nodes[node]['name'] = f"node_{i}"
                
                nx.write_graphml(G, output_path)
                valid_count += 1
            else:
                print(f"Invalid graph {filename}: {message}")
                invalid_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            invalid_count += 1

    print(f"Processing completed! Valid graphs: {valid_count}, Invalid graphs: {invalid_count}")

if __name__ == "__main__":
    input_directory = "fixed_graphs"
    output_directory = "valid_skeletons"
    
    process_graphs(input_directory, output_directory)


