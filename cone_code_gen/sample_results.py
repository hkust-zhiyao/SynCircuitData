import os
import json
import random
import glob
import shutil
import networkx as nx
from pathlib import Path
from tqdm import tqdm


valid_designs_json = "valid_designs.json"
fixed_designs_dir = "selected_skeletons"
gen_results_dir = "results"
output_base_dir = "sample_results"  


N = 12000  

def find_node_files(design):
    
    pattern = os.path.join(gen_results_dir, f"**/*new_{design}_cone_*")
    matching_dirs = glob.glob(pattern, recursive=True)
    return matching_dirs

def extract_output_reg_nodes(graphml_path):
    
    try:
        G = nx.read_graphml(graphml_path)
        
        target_nodes = []
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', '')
            if node_type in ['reg', 'output']:
                target_nodes.append(attrs.get('name', ''))
        
        return target_nodes
    except Exception as e:
        print(f"error processing GraphML file: {str(e)}")
        return []

def sample_and_create_test(design, test_index):
    
    
    test_dir = os.path.join(output_base_dir, f"test_{test_index}")
    os.makedirs(test_dir, exist_ok=True)
    
    
    graphml_path = os.path.join(fixed_designs_dir, f"{design}.graphml")
    if not os.path.exists(graphml_path):
        print(f"error: cannot find GraphML file {graphml_path}")
        return False
    
    
    shutil.copy2(graphml_path, os.path.join(test_dir, f"test_{test_index}_ast_sir.graphml"))
    print(f"copied GraphML file to {test_dir}")
    
    
    target_nodes = extract_output_reg_nodes(graphml_path)
    if not target_nodes:
        print(f"warning: no reg or output nodes found in GraphML")
        return False
    
    print(f"found {len(target_nodes)} reg/output nodes")
    
    
    node_folders = find_node_files(design)
    if not node_folders:
        print(f"error: no node folders found for design {design}")
        return False
    
    print(f"found {len(node_folders)} matching node folders")
    
    
    nodes_found = 0
    for node_name in target_nodes:
        candidate_files = []
        
        
        for folder in node_folders:
            node_file = os.path.join(folder, f"{node_name}.v")
            if os.path.exists(node_file):
                candidate_files.append(node_file)
        
        
        if candidate_files:
            selected_file = random.choice(candidate_files)
            shutil.copy2(selected_file, os.path.join(test_dir, f"{node_name}.v"))
            nodes_found += 1
            print(f"   copied {node_name}.v (selected from {len(candidate_files)} candidate files)")
        else:
            print(f"   warning: no Verilog file found for node {node_name}")
    
    print(f"completed test {test_index} for design {design}: found {nodes_found}/{len(target_nodes)} nodes")
    return nodes_found > 0

def main():
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    
    try:
        with open(valid_designs_json, 'r') as f:
            designs = json.load(f)
        print(f"successfully read {len(designs)} valid design names")
    except Exception as e:
        print(f"error: cannot read the design list: {str(e)}")
        return
    
    
    if len(designs) < 1:
        print("error: the design list is empty")
        return
    
    
    successful_tests = 0
    for i in tqdm(range(N)):
        
        selected_design = random.choice(designs)
        print(f"\n[{i+1}/{N}] create test {i} for design {selected_design}")
        
        if sample_and_create_test(selected_design, i):
            successful_tests += 1
    
    print(f"\nprocess completed! {successful_tests}/{N} test projects created")

if __name__ == "__main__":
    main()