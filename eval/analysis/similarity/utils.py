import networkx as nx
import json
from collections import deque

import networkx as nx
from collections import deque

import networkx as nx
from collections import deque
def split_reg_nodes(g):
    
    g_new = nx.DiGraph()
    
    
    for node in g.nodes():
        if g.nodes[node]['type'] != 'reg':
            g_new.add_node(node, **g.nodes[node])
    
    
    for node in g.nodes():
        if g.nodes[node]['type'] == 'reg':
            
            node_attr = g.nodes[node].copy()
            
            
            des_name = f"{node}_des"
            src_name = f"{node}_src" 
            
            
            g_new.add_node(des_name, **node_attr)
            g_new.add_node(src_name, **node_attr)
            g_new.nodes[des_name]['name'] = des_name
            g_new.nodes[src_name]['name'] = src_name
            
            for pred in g.predecessors(node):
                if g.nodes[pred]['type'] == 'reg':
                    g_new.add_edge(f"{pred}_src", des_name)
                else:
                    g_new.add_edge(pred, des_name)
            
            
            for succ in g.successors(node):
                if g.nodes[succ]['type'] == 'reg':
                    g_new.add_edge(src_name, f"{succ}_des")
                else:
                    g_new.add_edge(src_name, succ)
    
    
    for u, v in g.edges():
        if g.nodes[u]['type'] != 'reg' and g.nodes[v]['type'] != 'reg':
            g_new.add_edge(u, v)
            
    return g_new
def count_weakly_connected_components(g):
    
    
    weakly_connected_components = list(nx.weakly_connected_components(g))
    
    
    component_sizes = [len(component) for component in weakly_connected_components]
    
    print(f"图中共有 {len(weakly_connected_components)} 个弱连通分量")
    print("各连通分量的节点数量:")
    for i, size in enumerate(component_sizes):
        print(f"连通分量 {i+1}: {size} 个节点")
        
    return len(weakly_connected_components), component_sizes


def create_compressed_graph_no_reg_interm(G):
    

    
    valid_types = {"input", "reg", "output"}
    nodes_of_interest = [n for n, data in G.nodes(data=True) 
                         if data.get("type") in valid_types]

    
    subG = nx.DiGraph()
    for node in nodes_of_interest:
        subG.add_node(node, **G.nodes[node])

    
    
    def exists_path_no_reg_interm(s, t):
        if s == t:
            
            return False
        visited = set()
        queue = deque([s])
        while queue:
            current = queue.popleft()
            if current == t:
                return True
            for nxt in G.successors(current):
                if nxt not in visited:
                    
                    if nxt != t and G.nodes[nxt].get("type") == "reg":
                        continue
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    
    for s in nodes_of_interest:
        for t in nodes_of_interest:
            if s != t and exists_path_no_reg_interm(s, t):
                subG.add_edge(s, t)

    return subG


def visualize_and_save_graph(G, figsize=(20,20), save_path='graph.png'):
    
    import matplotlib.pyplot as plt
    
    
    plt.figure(figsize=figsize) 
    
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=2000)
    
    
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20)
    
    
    labels = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        label = f"{node}\n{node_data.get('type', 'N/A')}\nw:{node_data.get('width', 'N/A')}"
        labels[node] = label
    
    
    nx.draw_networkx_labels(G, pos, 
                           labels,
                           font_size=10,
                           font_family='sans-serif')
    
    plt.title("Graph Visualization", fontsize=16)
    plt.axis('off')
    
    
    plt.savefig(save_path, format="PNG", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存至 {save_path}")
def print_reg_nodes_predecessors(G):
    
    print("正在统计reg类型节点的父节点数量...")
    
    
    for node in G.nodes():
        
        
        
        
        
        
        if G.nodes[node].get('type') == 'reg':
            
            predecessors = list(G.predecessors(node))
            print(f"reg节点 {node} 的父节点数量: {len(predecessors)}")
def remove_const_nodes(G):
    
    
    G_new = nx.DiGraph()
    
    
    for node in G.nodes():
        if G.nodes[node].get('type') != 'const':
            G_new.add_node(node, **G.nodes[node])
    
    
    for u, v in G.edges():
        if (G.nodes[u].get('type') != 'const' and 
            G.nodes[v].get('type') != 'const'):
            G_new.add_edge(u, v)
    
    print(f"已删除所有type为const的节点")
    print(f"原图节点数: {G.number_of_nodes()}")
    print(f"新图节点数: {G_new.number_of_nodes()}")
    
    return G_new
































