import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(nn1))
        for _ in range(num_layers - 1):
            nn_layer = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                                     nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn_layer))
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = 0.1

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return x