import os
import re
import networkx as nx
import json
def generate_verilog_top(
    in_signals,
    out_signals,
    reg_signals,
    submodules,
    output_file="cone/top.v"
):
    code = []
    
    
    code.append("module top(")
    
    code.append("    // Input ports")
    for sig, width in in_signals:
        code.append(f"    input [{width-1}:0] node_{sig},")
    code.append("    input clock,")
    
    
    code.append("\n    // Output ports")
    for i, (sig, width) in enumerate(out_signals):
        code.append(f"    output reg [{width-1}:0] node_{sig}" + ("," if i < len(out_signals)-1 else ""))
    code.append(");\n")
    
    
    code.append("    // Internal registers")
    for sig, width in reg_signals:
        code.append(f"    reg [{width-1}:0] node_{sig};")
    
    
    code.append("\n    // Submodule instantiations")
    for mod_name, ports in submodules.items():
        code.append(f"\n    // Instance of {mod_name}")
        code.append(f"    {mod_name} {mod_name}_inst (")
        connections = []
        for port in ports:
            
            connections.append(f".{port}({port})")
        code.append(",\n        ".join(connections))
        code.append("    );")
    
    code.append("\nendmodule")
    
    return "\n".join(code)
def extract_code(text):

    try:
        start = text.index("##code##") + len("##code##")
        end = text.index("##end code##")
        return text[start:end].strip()
    except ValueError:
        return ''

def extract_submodule_ports(file_path):
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r"module\s+(\w+)\s*\((.*?)\);", content, re.DOTALL)
    if not match:
        return []
    ports_str = match.group(2)
    ports = []
    
    for item in ports_str.split(','):
        item = item.strip()
        
        item = re.split(r'//', item)[0].strip()
        if not item:
            continue
        tokens = item.split()
        if not tokens:
            continue
        
        port_name = tokens[-1].strip(' ,)')
        ports.append(port_name)
    return ports



def generate_top_module_code(graph_path, cone_dir='cone'):
    
    
    
    g = nx.read_graphml(graph_path)
    in_ports = []
    regs = []
    out_ports = []
    
    for node in g.nodes():
        if g.nodes[node].get('type') != 'input' and g.out_degree(node) == 0:
            g.nodes[node]['type'] = 'output'
    for node in g.nodes():
        node_type = g.nodes[node].get('type', '')
        width = g.nodes[node].get('width', None)
        if width is None:
            continue
        if node_type == 'input':
            in_ports.append((node, width))
        elif node_type == 'reg':
            regs.append((node, width))
        elif node_type == 'output':
            out_ports.append((node, width))
    
    
    submodules_detail = {}
    if os.path.exists(cone_dir):
        for filename in os.listdir(cone_dir):
            if filename.endswith('.v') and filename != 'top.v':
                mod_name = os.path.splitext(filename)[0]
                file_path = os.path.join(cone_dir, filename)
                ports = extract_submodule_ports(file_path)
                submodules_detail[mod_name] = ports
    else:
        os.makedirs(cone_dir, exist_ok=True)
    

    code = generate_verilog_top(in_ports, out_ports, regs, submodules_detail)
    
    top_file_path = os.path.join(cone_dir, "top.v")
    with open(top_file_path, 'w') as f:
        f.write(code)
    
    print(f"top module code generated and saved to {top_file_path}")
    return code


def process_folder(folder_path):
    
    import os
    import glob
    
    
    test_folders = glob.glob(os.path.join(folder_path, "test_*"))
    
    
    test_folders.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    print(f"在 {folder_path} 中找到 {len(test_folders)} 个测试文件夹")
    
    
    for test_folder in test_folders:
        test_name = os.path.basename(test_folder)
        print(f"正在处理测试: {test_name}")
        
        
        graph_path = os.path.join(test_folder, f"{test_name}_ast_sir.graphml")
        
        
        if not os.path.exists(graph_path):
            print(f"警告: 找不到图文件 {graph_path}，跳过此测试")
            continue
        
        try:
            
            generate_top_module_code(graph_path, test_folder)
            print(f"测试 {test_name} 处理完成")
        except Exception as e:
            print(f"处理测试 {test_name} 时出错: {str(e)}")
            continue

def process_sample_results():
    
    sample_results_dir = "/home/sliudx/project/rtl_aug/nips-circuitgen/59-gen/511_12000_sample_results"
    process_folder(sample_results_dir)
process_sample_results()










