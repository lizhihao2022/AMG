import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import unbatch

import einops

from .mlp import MLP
from utils.graph import global_sample, local_sample


class AttentionGraphBlock(nn.Module):
    def __init__(self, feature_width, input_features=1, output_features=1, num_heads=8,
                 batch_norm=True, act='relu', **kwargs):
        super(AttentionGraphBlock, self).__init__()
        self.graph_conv = GATv2Conv(input_features, feature_width, heads=num_heads, concat=False, negative_slope=0.2, dropout=0.0)
        self.ln_1 = nn.LayerNorm(feature_width)
        self.ffn = MLP(feature_width, feature_width, output_features, num_layers=1, batch_norm=batch_norm, act=act, **kwargs)
        self.ln_2 = nn.LayerNorm(input_features)
    
    def forward(self, x, edge_index):
        shortcut = x
        x = self.graph_conv(x, edge_index)
        x = self.ln_1(x + shortcut)
        shortcut = x
        x = self.ffn(x)
        x = self.ln_2(x + shortcut)
        return x


class PhysicsGraphBlock(nn.Module):
    def __init__(self, feature_width, num_heads=8, num_phys=32, dropout=0., **kwargs):
        super(PhysicsGraphBlock, self).__init__()
        hidden_width = feature_width * num_heads
        self.feature_width = feature_width
        self.num_heads = num_heads
        self.num_phys = num_phys
        self.softmax = nn.Softmax(dim=-1)
        self.scale = feature_width ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([num_heads, 1, 1]) * 0.5)
                
        self.l_in = nn.Linear(feature_width, hidden_width)
        self.l_token = nn.Linear(feature_width, hidden_width)
        self.l_phy = nn.Linear(feature_width, num_phys)
        for l in [self.l_phy]:
            torch.nn.init.orthogonal_(l.weight)
        
        self.q = nn.Linear(feature_width, feature_width, bias=False)
        self.k = nn.Linear(feature_width, feature_width, bias=False)
        self.v = nn.Linear(feature_width, feature_width, bias=False)
        
        self.l_out = nn.Linear(hidden_width, feature_width)
        
        self.ln_1 = nn.LayerNorm(feature_width)
        self.ffn = MLP(feature_width, feature_width, feature_width, num_layers=1, act='relu')
        self.ln_2 = nn.LayerNorm(feature_width)
        
    def single_in(self, x):
        phy_x = einops.rearrange(self.l_in(x), 'n (h c) -> h n c', h=self.num_heads)
        phy_weights = self.softmax(self.l_phy(phy_x) / self.temperature) # H N M
        
        phy_norm = phy_weights.sum(1).unsqueeze(-1) # H M 1
        phy_norm = (phy_norm + 1e-5).repeat(1, 1, self.feature_width) # H M C
        
        phy_token = einops.rearrange(self.l_token(x), 'n (h c) -> h n c', h=self.num_heads)
        phy_token = torch.einsum("hnc,hnm->hmc", phy_token, phy_weights)
        phy_token = phy_token / phy_norm
        
        return phy_token.unsqueeze(0), phy_weights
    
    def single_out(self, phy_token, phy_weights):
        out = torch.einsum("hmc,hnm->hnc", phy_token, phy_weights)
        out = einops.rearrange(out, 'h n c -> n (h c)')
        out = self.l_out(out)
        
        return out
    
    def forward(self, x, batch):
        shortcut = x
        x_batch = unbatch(x, batch)
        B = len(x_batch)
        phy_token_batch = []
        phy_weights_batch = []
        
        for x in x_batch:
            phy_token, phy_weights = self.single_in(x)
            phy_token_batch.append(phy_token)
            phy_weights_batch.append(phy_weights)
        
        phy_token = torch.cat(phy_token_batch, dim=0)
        phy_q = self.q(phy_token)
        phy_k = self.k(phy_token)
        phy_v = self.v(phy_token)
        dots = torch.matmul(phy_q, phy_k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        phy_y = torch.matmul(attn, phy_v)
        
        out_batch = []
        for i in range(B):
            out = self.single_out(phy_y[i], phy_weights_batch[i])
            out_batch.append(out)
        
        x = torch.cat(out_batch, dim=0)
        x = self.ln_1(x + shortcut)
        shortcut = x
        out = self.ln_2(self.ffn(x) + shortcut)
        
        return out


class MultiscaleGraphBlock(nn.Module):
    def __init__(self, feature_width, input_features=1, output_features=1, 
                 batch_norm=True, act='relu', num_phys=32, num_heads=8,
                 local_nodes=512, local_ratio=0.25, local_k=4, local_cos=False, local_pos=True,
                 global_ratio=0.25, global_k=8, global_cos=True, global_pos=False,
                 **kwargs):
        super(MultiscaleGraphBlock, self).__init__()        
        self.local_nodes = local_nodes
        self.local_ratio = local_ratio
        self.local_k = local_k
        self.local_cos = local_cos
        self.local_pos = local_pos
        
        self.global_ratio = global_ratio
        self.global_k = global_k
        self.global_cos = global_cos
        self.global_pos = global_pos
        
        self.num_phys = num_phys
        self.phy_aggr = PhysicsGraphBlock(feature_width, num_heads=num_heads, num_phys=num_phys, dropout=0.0)
        self.global_aggr = AttentionGraphBlock(feature_width, input_features=feature_width, output_features=feature_width, num_heads=num_heads, batch_norm=batch_norm, act=act, **kwargs)
        self.local_aggr = AttentionGraphBlock(feature_width, input_features=feature_width, output_features=feature_width, num_heads=num_heads, batch_norm=batch_norm, act=act, **kwargs)
        
        self.ln_1 = nn.LayerNorm(input_features)
        self.ln_2 = nn.LayerNorm(input_features)
        
        self.mlp = MLP(input_features, feature_width, output_features, num_layers=0, batch_norm=batch_norm, act=act, **kwargs)
        
    def forward(self, x, pos, batch=None):
        x_in = x
        x = self.phy_aggr(x, batch)
        
        local_edge_index, _, _, _ = local_sample(x, pos, sample_nodes=self.local_nodes, k=self.local_k, 
                                                 ratio=self.local_ratio, batch=batch, cosine=self.local_cos,
                                                 use_pos=self.local_pos)
        x = self.local_aggr(x, local_edge_index)

        global_edge_index, _, _, _ = global_sample(x, pos, ratio=self.global_ratio, k=self.global_k, 
                                                   batch=batch, cosine=self.global_cos, 
                                                   use_pos=self.global_pos)
        
        x = self.global_aggr(x, global_edge_index)
        
        x = self.mlp(self.ln_2(x + x_in))
        
        return x


class Grapher(nn.Module):
    def __init__(self, feature_width, num_layers, 
                 pos_dim=2, input_features=1, output_features=1, 
                 batch_norm=True, act='relu', 
                 local_nodes=512, local_ratio=0.25, local_k=8, local_cos=False, local_pos=True,
                 global_ratio=0.25, global_k=8, global_cos=True, global_pos=False, 
                 num_phys=32, num_heads=8, **kwargs):
        super(Grapher, self).__init__()
        self.num_layers = num_layers

        self.in_mlp = MLP(input_features + pos_dim, feature_width * 2, feature_width, num_layers=0, batch_norm=batch_norm, act=act, **kwargs)
        self.blocks = nn.ModuleList([
            MultiscaleGraphBlock(feature_width, input_features=feature_width, output_features=feature_width, 
                                 batch_norm=batch_norm, act=act, local_nodes=local_nodes, 
                                 local_ratio=local_ratio, local_cos=local_cos, local_pos=local_pos,
                                 local_k=local_k, global_ratio=global_ratio, global_k=global_k, 
                                 global_cos=global_cos, global_pos=global_pos, 
                                 num_phys=num_phys, num_heads=num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(feature_width)
        self.out_mlp = MLP(feature_width, feature_width, output_features, num_layers=0, batch_norm=batch_norm, act=act, **kwargs)
        
    def forward(self, data):
        x = torch.cat([data.x, data.pos], dim=-1)
        x = self.in_mlp(x)
        x_in = x

        for i in range(self.num_layers):
            x = self.blocks[i](x, data.pos, batch=data.batch)

        x_out = self.out_mlp(self.ln(x + x_in))

        return x_out
