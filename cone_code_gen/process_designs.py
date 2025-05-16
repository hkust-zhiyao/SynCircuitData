
import os
import json
import networkx as nx
from tqdm import tqdm
import re
import numpy as np
import argparse
from utils import load_graph_data_pyg
from graph_sage import GraphSAGE

import torch
from llm_request import generate_code_for_node_with_ref
def load_reference_embeddings(ref_filepath):
    with open(ref_filepath, 'r', encoding='utf-8') as f:
        ref_dict = json.load(f)
    ref_keys = list(ref_dict.keys())
    ref_embs = []
    for key in ref_keys:
        emb = np.array(ref_dict[key]["x"])
        ref_embs.append(emb)
    ref_embs = np.array(ref_embs)
    return ref_dict, ref_keys, ref_embs
def process_design(design_folder, design_names_file, output_folder, api_key, iter_num):

    model_ckpt = "./models/checkpoints_triplet_20_200/GraphSAGE_iter_1100.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    valid_folder = "selected_skeletons"
    
    design_names_path = os.path.join(valid_folder, design_names_file)
    with open(design_names_path, 'r', encoding='utf-8') as f:
        design_names = json.load(f)
    ref_filepath = "node_emb.json"
    ref_dict, ref_keys, ref_embs = load_reference_embeddings(ref_filepath)


    
    for i, design_name in enumerate(design_names):
        print(f"处理设计: {design_name} (索引: {i} / {len(design_names)})")
        
        
        
        
        graphml_file = os.path.join(valid_folder, f"{design_name}.graphml")
        if not os.path.exists(graphml_file):
            print(f"GraphML文件 {graphml_file} 不存在，跳过。")
            continue
        data = load_graph_data_pyg(graphml_file)
        
        g = nx.read_graphml(graphml_file)
        

        
        in_channels = data.x.size(1)
        model = GraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=64, num_layers=3).to(device)
        state_dict = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            node_embs = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()


        
        output_dir = os.path.join(output_folder, f"new_{design_name}_cone_{iter_num}")
        os.makedirs(output_dir, exist_ok=True)
        
        

        all_nodes = []
        for node in g.nodes():
            node_type = g.nodes[node].get('type', '')
            
            if node_type in ['reg', 'output']:
                all_nodes.append(node)
        
        
        for idx, node in enumerate(data.node_names):
            

            
            node_emb = node_embs[idx]
            
            dists = np.linalg.norm(ref_embs - node_emb, axis=1)
            min_idx = np.argmin(dists)
            chosen_key = ref_keys[min_idx]
            
            
            parts = chosen_key.split('+')
            if len(parts) >= 2:
                ref_node = parts[0]  
                ref_design = '+'.join(parts[1:])  
                print(f"found the most similar reference: node={ref_node}, design={ref_design}")
            else:
                print(f"warning: chosen_key format is incorrect: {chosen_key}")
                ref_node = ""
                ref_design = ""
            
            
            ref_code_dir = os.path.join(design_folder, f"{ref_design}_cone")
            ref_code_path = os.path.join(ref_code_dir, f"{ref_node}.txt")
            ref_func_path = os.path.join(ref_code_dir, f"{ref_node}_func.txt")
            ref_depth_path = os.path.join(design_folder, f"{ref_design}_depth.json")

            with open(ref_code_path, 'r', encoding='utf-8') as f:
                ref_code = f.read()
            
            ref_depth = 10  
            try:
                if os.path.exists(ref_depth_path):
                    with open(ref_depth_path, 'r', encoding='utf-8') as f:
                        depth_info = json.load(f)
                    
                    if ref_node in depth_info:
                        ref_depth = depth_info[ref_node]
                        print(f"read the depth of reference node {ref_node} from depth file: {ref_depth}")
                    else:
                        print(f"depth file does not contain reference node {ref_node}, use default depth {ref_depth}")
                else:
                    print(f"depth file {ref_depth_path} does not exist, use default depth {ref_depth}")
            except Exception as e:
                print(f"error reading depth file: {e}, use default depth {ref_depth}")
            with open(ref_func_path, 'r', encoding='utf-8') as f:
                ref_func = f.read()
            
            
            arithmetic_ops = {
                '+': 0,  
                '-': 0,  
                '*': 0,  
                '/': 0,  
                '%': 0   
            }
            
            
            for op in arithmetic_ops:
                
                if op in ['+', '-', '*']:
                    
                    pattern = r'[^a-zA-Z0-9_]' + re.escape(op) + r'[^a-zA-Z0-9_]'
                    arithmetic_ops[op] = len(re.findall(pattern, ref_code))
                else:
                    
                    arithmetic_ops[op] = ref_code.count(op)
            
            
            total_arithmetic_ops = sum(arithmetic_ops.values())
            
            print(f"the arithmetic operator statistics of reference code for node {node}:")
            for op, count in arithmetic_ops.items():
                print(f"  {op}: {count}")
            print(f"  total: {total_arithmetic_ops}")
            extra_info = f"the arithmetic operator statistics of reference code for node {node}: {arithmetic_ops}\n总计: {total_arithmetic_ops}，你写的代码在+,-,*，/, %这几个运算符使用上要模仿参考代码的数量，可以多一些或少一些。"
            
            
            ref_code_lines = ref_code.splitlines()
            if len(ref_code_lines) > 100:
                print(f"the reference code for node {node} has {len(ref_code_lines)} lines, more than 100 lines, so we sample 100 lines")
                
                import random
                sampled_indices = sorted(random.sample(range(len(ref_code_lines)), 100))
                sampled_lines = [ref_code_lines[i] for i in sampled_indices]
                ref_code = '\n'.join(sampled_lines)
                print(f"after sampling, the reference code for node {node} has {len(sampled_lines)} lines")
                
            
            
            try:
                
                print(f"generate code for node {node}, target depth: {ref_depth}")
                
                
                generated_code = generate_code_for_node_with_ref(
                    api_key, g, node, ref_func, ref_code, ref_depth, output_dir, extra_info=extra_info
                )
                
            
                    
            except Exception as e:
                print(f"error processing node {node}: {str(e)}")
                raise  
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="process design files and generate code for nodes.")
    parser.add_argument("--design_folder", required=True, help="the directory of design related files")
    parser.add_argument("--design_names_file", required=True, help="the JSON file containing design names (relative to design_folder)")
    parser.add_argument("--output_folder", required=True, help="the output directory of this instance")
    parser.add_argument("--api_key", type=str, required=True, help="the API key of this instance", default="")
    

    args = parser.parse_args()

    
    os.makedirs(args.output_folder, exist_ok=True)

    
    print(f"design folder: {args.design_folder}")
    print(f"design names file: {args.design_names_file}")
    print(f"output folder: {args.output_folder}")
    

    
    process_design(args.design_folder, args.design_names_file, args.output_folder, args.api_key, 0)
    process_design(args.design_folder, args.design_names_file, args.output_folder, args.api_key, 1)
    