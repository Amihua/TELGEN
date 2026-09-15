import numpy as np
import torch
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from data.utils import log_normalize


class LogNormalize:
    def __init__(self):
        pass

    def __call__(self, data):
        data.gt_primals = log_normalize(data.gt_primals)
        return data


class HeteroAddLaplacianEigenvectorPE:
    def __init__(self, k, attr_name='laplacian_eigenvector_pe'):
        self.k = k
        self.attr_name = attr_name

    def __call__(self, data):
        if self.k == 0:
            return data
        data_homo = data.to_homogeneous()
        del data_homo.edge_weight
        lap = AddLaplacianEigenvectorPE(k=self.k, attr_name=self.attr_name)(data_homo).laplacian_eigenvector_pe

        _, node_slices, _ = to_homogeneous_edge_index(data)
        cons_lap = lap[node_slices['cons'][0]: node_slices['cons'][1], :]
        cons_lap = (cons_lap - cons_lap.mean(0)) / cons_lap.std(0)
        vals_lap = lap[node_slices['vals'][0]: node_slices['vals'][1], :]
        vals_lap = (vals_lap - vals_lap.mean(0)) / vals_lap.std(0)
        obj_lap = lap[node_slices['obj'][0]: node_slices['obj'][1], :]

        data['cons'].laplacian_eigenvector_pe = cons_lap
        data['vals'].laplacian_eigenvector_pe = vals_lap
        data['obj'].laplacian_eigenvector_pe = obj_lap
        return data

    
class HeteroAddLaplacianEigenvectorPE_harp:
    def __init__(self, k, attr_name='laplacian_eigenvector_pe_harp'):
        self.k = k
        self.attr_name = attr_name

    def __call__(self, data):
        if self.k == 0:
            return data
        data_homo = data.to_homogeneous()
        del data_homo.edge_weight
        lap = AddLaplacianEigenvectorPE(k=self.k, attr_name=self.attr_name)(data_homo).laplacian_eigenvector_pe

        _, node_slices, _ = to_homogeneous_edge_index(data)
        cons_lap = lap[node_slices['cons'][0]: node_slices['cons'][1], :]
        cons_lap = (cons_lap - cons_lap.mean(0)) / cons_lap.std(0)
        econs_lap = lap[node_slices['econs'][0]: node_slices['econs'][1], :]
        econs_lap = (econs_lap - econs_lap.mean(0)) / econs_lap.std(0)
        vals_lap = lap[node_slices['vals'][0]: node_slices['vals'][1], :]
        vals_lap = (vals_lap - vals_lap.mean(0)) / vals_lap.std(0)
        obj_lap = lap[node_slices['obj'][0]: node_slices['obj'][1], :]

        data['cons'].laplacian_eigenvector_pe = cons_lap
        data['econs'].laplacian_eigenvector_pe = econs_lap
        data['vals'].laplacian_eigenvector_pe = vals_lap
        data['obj'].laplacian_eigenvector_pe = obj_lap
        return data

    

# in such case, k >= len_seq
class SubSample_pad:
    def __init__(self, k):
        self.k = k
    
    def __call__(self, data):
        len_seq = data.gt_primals.shape[1]
        pad = torch.full(data.gt_primals[:, -1:].shape, float('nan'))
        data.gt_primals = torch.cat([data.gt_primals,
                                     pad.repeat(1, self.k - len_seq)], dim=1)
        if hasattr(data, 'gt_duals'):
            data.gt_duals = torch.cat([data.gt_duals,
                                       pad.repeat(1, self.k - len_seq)], dim=1)
        if hasattr(data, 'gt_slacks'):
            data.gt_slacks = torch.cat([data.gt_slacks,
                                        pad.repeat(1, self.k - len_seq)], dim=1)
        return data

    

class SubSample:
    def __init__(self, k):
        self.k = k
    
    def __call__(self, data):
        len_seq = data.gt_primals.shape[1]
        if self.k == 1:                   # if sample only one step of ipm: use the last step
            data.gt_primals = data.gt_primals[:, -1:]
            if hasattr(data, 'gt_duals'):
                data.gt_duals = data.gt_duals[:, -1:]
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = data.gt_slacks[:, -1:]
        elif self.k == len_seq:           # if the sample size == len(inters) of linprog, use the whole data
            return data
        elif self.k > len_seq:            # if the sample size > len(inters), repeat the last inter until len is equal
            data.gt_primals = torch.cat([data.gt_primals,
                                         data.gt_primals[:, -1:].repeat(1, self.k - len_seq)], dim=1)
            if hasattr(data, 'gt_duals'):
                data.gt_duals = torch.cat([data.gt_duals,
                                           data.gt_duals[:, -1:].repeat(1, self.k - len_seq)], dim=1)
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = torch.cat([data.gt_slacks,
                                            data.gt_slacks[:, -1:].repeat(1, self.k - len_seq)], dim=1)
        else:                             # if the sample size < len(inters), take evenly spaced numbers over a len(data)
            data.gt_primals = data.gt_primals[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)] 
            if hasattr(data, 'gt_duals'):
                data.gt_duals = data.gt_duals[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = data.gt_slacks[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
        return data

    
class SubSample_mix:
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        len_seq = data.gt_primals.shape[1]
        if self.k == 1:                   # if sample only one step of ipm: use the last step
            data.gt_primals = data.gt_primals[:, -1:]
            if hasattr(data, 'gt_duals'):
                data.gt_duals = data.gt_duals[:, -1:]
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = data.gt_slacks[:, -1:]
        elif self.k >= len_seq:           # if the sample size == len(inters) of linprog, use the whole data
            data.gt_primals = torch.cat(data.gt_primals[:, int(self.k/2)+1], data.gt_primals[:, -1:].repeat(1, self.k - int(self.k/2)))
            if hasattr(data, 'gt_duals'):
                data.gt_duals = torch.cat(data.gt_duals[:, int(self.k/2)+1], data.gt_duals[:, -1:].repeat(1, self.k - int(self.k/2)))
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = torch.cat(data.gt_slacks[:, int(self.k/2)+1], data.gt_slacks[:, -1:].repeat(1, self.k - int(self.k/2)))
            return data
        else:                             # if the sample size < len(inters), take evenly spaced numbers over a len(data)
            data.gt_primals = data.gt_primals[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)] 
            if hasattr(data, 'gt_duals'):
                data.gt_duals = data.gt_duals[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
            if hasattr(data, 'gt_slacks'):
                data.gt_slacks = data.gt_slacks[:, np.linspace(1, len_seq - 1, self.k).astype(np.int64)]
        return data


class SubSample_:
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        len_seq = data.gt_primals.shape[1]

        data.gt_primals = data.gt_primals[:, -1:].repeat(1, self.k - len_seq + 1)
        if hasattr(data, 'gt_duals'):
            data.gt_duals = data.gt_duals[:, -1:].repeat(1, self.k - len_seq + 1)
        if hasattr(data, 'gt_slacks'):
            data.gt_slacks = data.gt_slacks[:, -1:].repeat(1, self.k - len_seq + 1)
                
        return data

    
    
    
    