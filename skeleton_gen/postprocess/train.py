import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, Node2Vec
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.utils import get_laplacian, to_dense_adj


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


WIDTH_CLASSES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def map_width_to_class(width):
    width_array = np.array(WIDTH_CLASSES)
    idx = np.argmin(np.abs(width_array - width))
    return idx, WIDTH_CLASSES[idx]


class TypeGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TypeGNNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class WidthGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(WidthGNNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class EdgeGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EdgeGNNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class NodeTypePredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(NodeTypePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
        )
        self.type_pred = nn.Linear(hidden_dim, 3)
    
    def forward(self, x):
        h = self.mlp(x)
        type_logits = self.type_pred(h)
        return type_logits

class NodeWidthPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(NodeWidthPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
        )
        
        self.width_pred = nn.Linear(hidden_dim, len(WIDTH_CLASSES))
    
    def forward(self, x):
        h = self.mlp(x)
        width_logits = self.width_pred(h)
        return width_logits

class EdgeDirectionPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(EdgeDirectionPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_i, x_j):
        
        x = torch.cat([x_i, x_j], dim=1)
        return self.mlp(x)


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
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        
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
        
        
        enhanced_x = torch.cat([data.x, node2vec_emb, laplacian_pe], dim=1)
            
        data.x = enhanced_x
        return data


def read_graphml_files(directory):
    all_data = []
    
    for file_path in tqdm(glob.glob(os.path.join(directory, "*.graphml")), desc="read graphml files"):
        
        G = nx.read_graphml(file_path)
        
        
        node_indices = {node: i for i, node in enumerate(G.nodes())}
        x = torch.ones((len(G.nodes()), 1), dtype=torch.float)  
        
        
        node_types = []
        node_widths = []
        node_width_classes = []
        
        for node, attrs in G.nodes(data=True):
            type_value = attrs.get('type', 'reg')  
            width_value = float(attrs.get('width', 1))  
            
            
            if type_value == 'input':
                type_idx = 0
            elif type_value == 'output':
                type_idx = 1
            else:  
                type_idx = 2
                
            
            width_class_idx, _ = map_width_to_class(width_value)
            
            node_types.append(type_idx)
            node_widths.append(float(width_value))
            node_width_classes.append(width_class_idx)
        
        
        edge_index = []
        edge_labels = []
        
        for src, dst in G.edges():
            
            i, j = node_indices[src], node_indices[dst]
            edge_index.append([i, j])
            edge_labels.append(1.0)  
            
            
            edge_index.append([j, i])
            edge_labels.append(0.0)  
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_labels = torch.tensor(edge_labels, dtype=torch.float)
            
            
            node_types = torch.tensor(node_types, dtype=torch.long)
            node_widths = torch.tensor(node_widths, dtype=torch.float).view(-1, 1)
            node_width_classes = torch.tensor(node_width_classes, dtype=torch.long)
            
            
            data = Data(x=x, 
                      edge_index=edge_index, 
                      node_types=node_types,
                      node_widths=node_widths,
                      node_width_classes=node_width_classes,
                      edge_labels=edge_labels)
            
            all_data.append(data)
    
    return all_data


def apply_positional_encoding(data_list, pe_dim=16):
    pe_generator = EnhancedPositionalEncoding(dim=pe_dim)
    enhanced_data_list = []
    
    print("applying positional encoding...")
    for data in tqdm(data_list):
        
        enhanced_data = pe_generator.apply_to_data(data)
        enhanced_data_list.append(enhanced_data)
    
    return enhanced_data_list


def train():
    
    print("reading training data...")
    train_data = read_graphml_files('graphml_files')
    
    
    pe_dim = 16
    train_data = apply_positional_encoding(train_data, pe_dim)
    
    
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    
    
    in_channels = 1 + 2 * pe_dim  
    hidden_channels = 32
    out_channels = 32
    
    
    type_encoder = TypeGNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    width_encoder = WidthGNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    edge_encoder = EdgeGNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    
    
    type_predictor = NodeTypePredictor(out_channels, hidden_channels).to(device)
    width_predictor = NodeWidthPredictor(out_channels, hidden_channels).to(device)
    edge_predictor = EdgeDirectionPredictor(out_channels, hidden_channels).to(device)
    
    
    type_optimizer = torch.optim.Adam(list(type_encoder.parameters()) + 
                                      list(type_predictor.parameters()), lr=0.001)
    width_optimizer = torch.optim.Adam(list(width_encoder.parameters()) + 
                                       list(width_predictor.parameters()), lr=0.001)
    edge_optimizer = torch.optim.Adam(list(edge_encoder.parameters()) + 
                                      list(edge_predictor.parameters()), lr=0.001)
    
    
    type_criterion = nn.CrossEntropyLoss()
    width_criterion = nn.CrossEntropyLoss()  
    edge_criterion = nn.BCELoss()
    
    
    best_val_loss = float('inf')
    
    print("start training...")
    for epoch in range(160):
        
        type_encoder.train()
        width_encoder.train()
        edge_encoder.train()
        type_predictor.train()
        width_predictor.train()
        edge_predictor.train()
        
        type_total_loss = 0
        width_total_loss = 0
        edge_total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            
            
            type_optimizer.zero_grad()
            type_embeddings = type_encoder(data.x, data.edge_index)
            type_logits = type_predictor(type_embeddings)
            type_loss = type_criterion(type_logits, data.node_types)
            type_loss.backward()
            type_optimizer.step()
            type_total_loss += type_loss.item()
            
            
            width_optimizer.zero_grad()
            width_embeddings = width_encoder(data.x, data.edge_index)
            width_logits = width_predictor(width_embeddings)
            width_loss = width_criterion(width_logits, data.node_width_classes)
            width_loss.backward()
            width_optimizer.step()
            width_total_loss += width_loss.item()
            
            
            edge_optimizer.zero_grad()
            edge_embeddings = edge_encoder(data.x, data.edge_index)
            src_nodes, dst_nodes = data.edge_index
            src_embeddings = edge_embeddings[src_nodes]
            dst_embeddings = edge_embeddings[dst_nodes]
            
            edge_preds = edge_predictor(src_embeddings, dst_embeddings)
            edge_loss = edge_criterion(edge_preds.view(-1), data.edge_labels)
            edge_loss.backward()
            edge_optimizer.step()
            edge_total_loss += edge_loss.item()
        
        
        type_encoder.eval()
        width_encoder.eval()
        edge_encoder.eval()
        type_predictor.eval()
        width_predictor.eval()
        edge_predictor.eval()
        
        type_val_loss = 0
        width_val_loss = 0
        edge_val_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                
                
                type_embeddings = type_encoder(data.x, data.edge_index)
                type_logits = type_predictor(type_embeddings)
                type_loss = type_criterion(type_logits, data.node_types)
                type_val_loss += type_loss.item()
                
                
                width_embeddings = width_encoder(data.x, data.edge_index)
                width_logits = width_predictor(width_embeddings)
                width_loss = width_criterion(width_logits, data.node_width_classes)
                width_val_loss += width_loss.item()
                
                
                edge_embeddings = edge_encoder(data.x, data.edge_index)
                src_nodes, dst_nodes = data.edge_index
                src_embeddings = edge_embeddings[src_nodes]
                dst_embeddings = edge_embeddings[dst_nodes]
                
                edge_preds = edge_predictor(src_embeddings, dst_embeddings)
                edge_loss = edge_criterion(edge_preds.view(-1), data.edge_labels)
                edge_val_loss += edge_loss.item()
        
        
        total_val_loss = type_val_loss + width_val_loss + edge_val_loss
        
        print(f'  The {epoch+1} epoch:')
        print(f'  type training loss: {type_total_loss/len(train_loader):.4f}, validation loss: {type_val_loss/len(val_loader):.4f}')
        print(f'  width training loss: {width_total_loss/len(train_loader):.4f}, validation loss: {width_val_loss/len(val_loader):.4f}')
        print(f'  edge direction training loss: {edge_total_loss/len(train_loader):.4f}, validation loss: {edge_val_loss/len(val_loader):.4f}')
        print(f'  total validation loss: {total_val_loss/len(val_loader):.4f}')
        
        
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            
            
            print("saving the best model...")
            torch.save({
                'type_encoder': type_encoder.state_dict(),
                'width_encoder': width_encoder.state_dict(),
                'edge_encoder': edge_encoder.state_dict(),
                'type_predictor': type_predictor.state_dict(),
                'width_predictor': width_predictor.state_dict(),
                'edge_predictor': edge_predictor.state_dict(),
                'pe_dim': pe_dim,
                'width_classes': WIDTH_CLASSES
            }, 'best_graph_model.pt')
    
    print("training completed!")

if __name__ == "__main__":
    train()