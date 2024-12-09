import torch
from torch_geometric.nn.pool import fps
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import knn_graph


def construct_coordinate(data, **kwargs):
    x_features = data.x.size(-1)
    y_features = data.y.size(-1)
    pos_features = data.pos.size(-1)
    data.x = data.x.reshape(-1, x_features)
    data.y = data.y.reshape(-1, y_features)
    data.pos = data.pos.reshape(-1, pos_features)
    return data


def compute_feature_map(y, pos, ratio=0.25, batch=None):
    index_down = fps(pos, ratio=ratio, batch=batch)
    pos_down = pos[index_down]
    y_down = y[index_down]
    y_up = knn_interpolate(x=y_down, pos_x=pos_down, pos_y=pos, batch_x=batch[index_down] if batch is not None else None, batch_y=batch)

    fm = torch.abs(y - y_up)
    fm = torch.sum(fm, dim=1)
    return fm, index_down


def local_sample(x, pos, sample_nodes=512, k=8, ratio=0.25, batch=None, cosine=False, use_pos=False):
    fm, _ = compute_feature_map(x, pos, ratio, batch)
    
    if batch is not None:
        batch_size = batch.max().item() + 1
        sampled_indices = None
        sampled_batch = None

        for b in range(batch_size):
            mask = (batch == b)
            fm_batch = fm[mask]
            _, indices_topk_local = torch.topk(fm_batch, k=min(sample_nodes, fm_batch.size(0)), largest=True)
            
            if sampled_indices is None:
                sampled_indices = torch.nonzero(mask).to(x.device).flatten()[indices_topk_local].to(indices_topk_local.device)
                sampled_batch = torch.tensor([b] * indices_topk_local.size(0)).to(x.device)
            else:
                sampled_indices = torch.cat([sampled_indices, torch.nonzero(mask).flatten()[indices_topk_local]], dim=0)
                sampled_batch = torch.cat([sampled_batch, torch.tensor([b] * indices_topk_local.size(0)).to(x.device)], dim=0)
        if use_pos:
            sampled_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
        else:
            sampled_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
        edge_index = torch.cat([
            sampled_indices[sampled_edge_index[0]].view(1, -1), 
            sampled_indices[sampled_edge_index[1]].view(1, -1)], dim=0)
    else:
        _, indices_topk_local = torch.topk(fm, k=min(sample_nodes, fm.size(0)), largest=True)
        sampled_indices = torch.nonzero(fm).flatten()[indices_topk_local]
        if use_pos:
            sampled_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, cosine=cosine)
        else:
            sampled_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, cosine=cosine)
        edge_index = torch.cat([
            sampled_indices[sampled_edge_index[0]].view(1, -1), 
            sampled_indices[sampled_edge_index[1]].view(1, -1)], dim=0)
        sampled_batch = None
    
    return edge_index, sampled_edge_index, sampled_indices, sampled_batch


def global_sample(x, pos, ratio=0.25, k=8, batch=None, cosine=False, use_pos=False):
    sampled_indices = fps(pos, ratio=ratio, batch=batch)
    sampled_batch = batch[sampled_indices] if batch is not None else None
    
    if use_pos:
        sampled_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    else:
        sampled_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    edge_index = torch.cat([
        sampled_indices[sampled_edge_index[0]].view(1, -1), 
        sampled_indices[sampled_edge_index[1]].view(1, -1)], dim=0)
    
    return edge_index, sampled_edge_index, sampled_indices, sampled_batch
