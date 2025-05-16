


import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

class GINConv(nn.Module):
    r
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):
        r
        with graph.local_scope():
            aggregate_fn = self.concat_edge_msg
            
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))


            diff = torch.tensor(graph.dstdata['neigh'].shape[1: ]) - torch.tensor(feat_dst.shape[1: ])
            zeros = torch.zeros(feat_dst.shape[0], *diff).to(feat_dst.device)
            feat_dst = torch.cat([feat_dst, zeros], dim=1)
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst

    def concat_edge_msg(self, edges):
        if self.edge_feat_loc not in edges.data:
            return {'m': edges.src['h']}
        else:
            m = torch.cat([edges.src['h'], edges.data[self.edge_feat_loc]], dim=1)
            return {'m': m}


class ApplyNodeFunc(nn.Module):
    
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        
        super(MLP, self).__init__()
        self.linear_or_not = True  
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            
            self.linear = nn.Linear(input_dim, output_dim)

        else:
            
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            
            return self.linear(x)
        else:
            
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 graph_pooling_type, neighbor_pooling_type, edge_feat_dim=0,
                 final_dropout=0.0, learn_eps=False, output_dim=1, **kwargs):
        

        super().__init__()
        def init_weights_orthogonal(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
            elif isinstance(m, MLP):
                if hasattr(m, 'linears'):
                    m.linears.apply(init_weights_orthogonal)
                else:
                    m.linear.apply(init_weights_orthogonal)
            elif isinstance(m, nn.ModuleList):
                pass
            else:
                raise Exception()

        self.num_layers = num_layers
        self.learn_eps = learn_eps

        
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        
        
        
        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim + edge_feat_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim + edge_feat_dim, hidden_dim, hidden_dim)
            if kwargs['init'] == 'orthogonal':
                init_weights_orthogonal(mlp)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        
        
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))


        if kwargs['init'] == 'orthogonal':
            self.linears_prediction.apply(init_weights_orthogonal)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        
        hidden_rep = [h]

        
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer

    def get_graph_embed(self, g, h):
        self.eval()
        with torch.no_grad():
            
            hidden_rep = []
            
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            
            graph_embed = torch.Tensor([]).to(self.device)
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            return graph_embed

    def get_graph_embed_no_cat(self, g, h):
        self.eval()
        with torch.no_grad():
            hidden_rep = []
            
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            
            
            
            
            

            
            return self.pool(g, hidden_rep[-1]).to(self.device)

    @property
    def edge_feat_loc(self):
        return self.ginlayers[0].edge_feat_loc

    @edge_feat_loc.setter
    def edge_feat_loc(self, loc):
        for layer in self.ginlayers:
            layer.edge_feat_loc = loc
