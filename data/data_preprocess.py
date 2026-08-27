import numpy as np
import torch
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from data.utils import log_normalize
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs

# Custom AddLaplacianEigenvectorPE to handle numerical instability
def custom_laplacian_eigenvector_pe(data, k, attr_name='laplacian_eigenvector_pe'):
    """
    A robust, multi-stage implementation of Laplacian Eigenvector Positional Encoding.
    It attempts to compute k eigenvectors, but gracefully degrades to smaller k values
    or zero vectors upon numerical failure.
    """
    if data.num_nodes == 0:
        for node_type in data.node_types:
            data[node_type][attr_name] = torch.empty((0, k), dtype=torch.float32)
        return data

    edge_index, _ = get_laplacian(data.edge_index, num_nodes=data.num_nodes, normalization='sym')
    L = to_scipy_sparse_matrix(edge_index)

    # Multi-stage degradation strategy
    k_tries = [k]
    if k > 4: k_tries.append(4)
    if k > 2: k_tries.append(2)

    eigvecs = None

    for k_try in k_tries:
        if k_try <= 0: continue
        # For sparse matrices, ncv must be > k. A safe value is often 2*k+1 or larger.
        ncv = max(2 * k_try + 1, 40)
        
        try:
            # Find the smallest magnitude eigenvalues
            eigvals, eigvecs_try = eigs(L, k=k_try, which='SM', ncv=ncv)
            
            # If successful, sort, pad if necessary, and break the loop
            eigvecs_try = np.real(eigvecs_try[:, np.argsort(np.abs(np.real(eigvals)))])
            
            # Pad with zeros if we got fewer eigenvectors than originally requested
            if k_try < k:
                print(f"\n[Warning] ARPACK succeeded with a smaller k={k_try}. Padding with zeros.")
                padding = np.zeros((data.num_nodes, k - k_try), dtype=np.float32)
                eigvecs = np.concatenate([eigvecs_try, padding], axis=1)
            else:
                eigvecs = eigvecs_try
            
            # Success, we can stop trying
            break

        except Exception as e:
            print(f"\n[Info] ARPACK failed for k={k_try}. Trying a smaller k. Error: {e}")
            continue # Go to the next smaller k

    # If all attempts failed, fall back to zero vectors
    if eigvecs is None:
        print(f"\n[ERROR] All ARPACK attempts failed for k up to {k}. Falling back to zero vectors.")
        eigvecs = np.zeros((data.num_nodes, k), dtype=np.float32)

    data[attr_name] = torch.from_numpy(eigvecs)
    
    return data


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
        del data_homo.edge_weight # Ensure unweighted graph for PE calculation

        # Use the robust custom function instead of the default PyG one
        data_homo = custom_laplacian_eigenvector_pe(data_homo, k=self.k, attr_name=self.attr_name)
        lap = data_homo[self.attr_name]

        _, node_slices, _ = to_homogeneous_edge_index(data)
        cons_lap = lap[node_slices['cons'][0]: node_slices['cons'][1], :]
        cons_mean = cons_lap.mean(0)
        cons_std = cons_lap.std(0)
        cons_std = torch.where(cons_std == 0, torch.ones_like(cons_std), cons_std)
        cons_lap = (cons_lap - cons_mean) / cons_std
        vals_lap = lap[node_slices['vals'][0]: node_slices['vals'][1], :]
        vals_mean = vals_lap.mean(0)
        vals_std = vals_lap.std(0)
        vals_std = torch.where(vals_std == 0, torch.ones_like(vals_std), vals_std)
        vals_lap = (vals_lap - vals_mean) / vals_std
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
        cons_mean = cons_lap.mean(0)
        cons_std = cons_lap.std(0)
        cons_std = torch.where(cons_std == 0, torch.ones_like(cons_std), cons_std)
        cons_lap = (cons_lap - cons_mean) / cons_std
        econs_lap = lap[node_slices['econs'][0]: node_slices['econs'][1], :]
        econs_mean = econs_lap.mean(0)
        econs_std = econs_lap.std(0)
        econs_std = torch.where(econs_std == 0, torch.ones_like(econs_std), econs_std)
        econs_lap = (econs_lap - econs_mean) / econs_std
        vals_lap = lap[node_slices['vals'][0]: node_slices['vals'][1], :]
        vals_mean = vals_lap.mean(0)
        vals_std = vals_lap.std(0)
        vals_std = torch.where(vals_std == 0, torch.ones_like(vals_std), vals_std)
        vals_lap = (vals_lap - vals_mean) / vals_std
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

    
    
    
    