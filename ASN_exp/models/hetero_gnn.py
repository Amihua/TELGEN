import torch
import torch.nn.functional as F

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.ginconv import GINEConv
from models.utils import MLP
from models.hetero_conv import HeteroConv, HeteroConv_quad

# update order, the smaller the higher priority

def strseq2rank(conv_sequence):
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = 0
    elif conv_sequence == 'cvo':
        v2c = o2c = 0
        c2v = o2v = 1
        c2o = v2o = 2
    elif conv_sequence == 'vco':
        c2v = o2v = 0
        v2c = o2c = 1
        c2o = v2o = 2
    elif conv_sequence == 'ocv':
        c2o = v2o = 0
        v2c = o2c = 1
        c2v = o2v = 2
    elif conv_sequence == 'ovc':
        c2o = v2o = 0
        c2v = o2v = 1
        v2c = o2c = 2
    elif conv_sequence == 'voc':
        c2v = o2v = 0
        c2o = v2o = 1
        v2c = o2c = 2
    elif conv_sequence == 'cov':
        v2c = o2c = 0
        c2o = v2o = 1
        c2v = o2v = 2
    else:
        raise ValueError
    return c2v, v2c, v2o, o2v, c2o, o2c


# o: obj, v: var, c: inequality, e: equality
def strseq2rank_harp(conv_sequence):       # add equality and inequality constraints
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = v2e = e2v = o2e = e2o = 0
    elif conv_sequence == 'cevo':
        v2c = o2c = 0
        v2e = o2e = 1
        c2v = o2v = e2v = 2
        c2o = v2o = e2o = 3
    elif conv_sequence == 'vceo':
        c2v = o2v = e2v = 0
        v2c = o2c = 1
        v2e = o2e = 2
        c2o = v2o = e2o = 3
    elif conv_sequence == 'ocev':
        c2o = v2o = e2o = 0
        v2c = o2c = 1
        v2e = o2e = 2
        c2v = o2v = e2v = 3
    elif conv_sequence == 'ovce':
        c2o = v2o = e2o = 0
        c2v = o2v = e2v = 1
        v2c = o2c = 2
        v2e = o2e = 3
    elif conv_sequence == 'voce':
        c2v = o2v = e2v = 0
        c2o = v2o = e2o = 1
        v2c = o2c = 2
        v2e = o2e = 3
    elif conv_sequence == 'ceov':
        v2c = o2c = 0
        v2e = o2e = 1
        c2o = v2o = e2o = 2
        c2v = o2v = e2v = 3
    else:
        raise ValueError
    return c2v, v2c, e2v, v2e, v2o, o2v, c2o, o2c, e2o, o2e



# this the gnn layer for message passing: defined in models.genconv/gcnconv/ginconv

def get_conv_layer(conv: str,
                   in_dim: int,
                   hid_dim: int,
                   num_mlp_layers: int,
                   use_norm: bool,
                   in_place: bool):
    if conv.lower() == 'genconv':
        def get_conv():
            return GENConv(in_channels=in_dim,
                           out_channels=hid_dim,
                           num_layers=num_mlp_layers,
                           aggr='softmax',
                           msg_norm=use_norm,
                           learn_msg_scale=use_norm,
                           norm='batch' if use_norm else None,
                           bias=True,
                           edge_dim=1,
                           in_place=in_place)
    elif conv.lower() == 'gcnconv':
        def get_conv():
            return GCNConv(in_dim=in_dim,
                           edge_dim=1,
                           hid_dim=hid_dim,
                           num_mlp_layers=num_mlp_layers,
#                            norm='batch' if use_norm else None,
                           norm='instance' if use_norm else None,    # this is for batch=1
                           in_place=in_place)
    elif conv.lower() == 'ginconv':
        def get_conv():
            return GINEConv(in_dim=in_dim,
                            edge_dim=1,
                            hid_dim=hid_dim,
                            num_mlp_layers=num_mlp_layers,
                            norm='batch' if use_norm else None,
                            in_place=in_place)
    else:
        raise NotImplementedError

    return get_conv


# final GMM: 
# encoder (MLP): encode node sets information
# mp (GCN/GEN/GIN): num_layer, output vals/cons of layers of gnn
# decoder (MLP) for decoding vals/cons: share weight or not, share (use one MLP), not share(use num_layer MLPs for each layer)
# the gnn is aggregating information to serve as mp step; the decoder layer are supervised by the ipm intermediate values 

class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res
        self.in_shape = in_shape

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        '''
        # amends part
        for k in ['cons', 'vals', 'obj']:
            x_dict[k] = x_dict[k].reshape(-1, self.in_shape)
        # amends part
        '''
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            x_dict[k] = x_emb

        hiddens = []
        for i in range(self.num_layers):
            #print('num_layers:', i)
            if self.share_conv_weight:
                i = 0

            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            hiddens.append((h2['cons'], h2['vals']))
            #print(h2['cons'].shape, h2['vals'].shape)
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h2[k]) for k in keys}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            x_dict = h

        cons, vals = zip(*hiddens)
        #print(len(hiddens))

        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            #print('pred: vals, cons', vals.shape, cons.shape, vals[0], cons[0])
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.num_layers)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.num_layers)], dim=1)
            #print('pred: vals, cons', vals.shape, cons.shape, vals[0], cons[0])
            return vals, cons

        
# TripartiteHeteroGNN_: for our inner+outer loop network structure
class TripartiteHeteroGNN_(torch.nn.Module):
    def __init__(self,
                 ipm_steps,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()
        
        self.ipm_steps = ipm_steps
        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            #print(k)
            #print(x_dict[k].shape)
            x_emb = self.encoder[k](x_dict[k])
            #print('x_emb', x_emb.shape)
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            x_dict[k] = x_emb

        hiddens = []
        for _ in range(int(self.ipm_steps)):
            # mimic every ipm iteration: within every iteration, there are self.num_layers of GNN serve as mp 
            for i in range(self.num_layers):
                if self.share_conv_weight:      
                    i = 0

                h1 = x_dict
                h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
                keys = h2.keys()
                
                if self.use_res:
                    h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
                else:
                    h = {k: F.relu(h2[k]) for k in keys}
                h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
                x_dict = h
            hiddens.append((h2['cons'], h2['vals']))
        cons, vals = zip(*hiddens)
        if self.share_lin_weight:
            # use this sharing link actually, since var_dict['share_lin_weight']='false', not False
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            vals = F.relu(vals)
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.ipm_steps)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.ipm_steps)], dim=1)
            vals = F.relu(vals)
            return vals, cons




# TripartiteHeteroGNN_harp: for comparision with harp
class TripartiteHeteroGNN_harp(torch.nn.Module):
    def __init__(self,
                 ipm_steps,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()
        
        self.ipm_steps = ipm_steps
        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),    # inequality constraints
                'econs': MLP([pe_dim, hid_dim, hid_dim]),    # equality constraints
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'econs': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

        c2v, v2c, e2v, v2e, v2o, o2v, c2o, o2c, e2o, o2e = strseq2rank_harp(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('econs', 'to', 'vals'): (get_conv(), e2v),
                        ('vals', 'to', 'econs'): (get_conv(), v2e),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                        ('econs', 'to', 'obj'): (get_conv(), e2o),
                        ('obj', 'to', 'econs'): (get_conv(), o2e),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])  # [2*180] + [180]*[4-1] + [1] 
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])  # [2*180] + [180]*[4-1] + [1] 
            self.pred_econs = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]) # [2*180] + [180]*[4-1] + [1] 
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            self.pred_econs = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_econs.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'econs', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            x_dict[k] = x_emb

        hiddens = []
        for _ in range(int(self.ipm_steps)):
            for i in range(self.num_layers):
                if self.share_conv_weight:      
                    i = 0

                h1 = x_dict
                h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
                keys = h2.keys()
                
                if self.use_res:
                    h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
                else:
                    h = {k: F.relu(h2[k]) for k in keys}
                h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
                x_dict = h
            hiddens.append((h2['cons'], h2['econs'], h2['vals']))
        cons, vals = zip(*hiddens)
        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            econs = self.pred_econs(torch.stack(cons, dim=0))
            vals = F.relu(vals)
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.ipm_steps)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.ipm_steps)], dim=1)
            econs = torch.cat([self.pred_econs[i](econs[i]) for i in range(self.ipm_steps)], dim=1)
            vals = F.relu(vals)
            return vals, cons, econs

        
# QuadpartiteHeteroGNN: for comparision with harp
class QuadpartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 ipm_steps,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()
        
        self.ipm_steps = ipm_steps
        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res

        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),    # inequality constraints
                'econs': MLP([pe_dim, hid_dim, hid_dim]),    # equality constraints
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'econs': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

        c2v, v2c, e2v, v2e, v2o, o2v, c2o, o2c, e2o, o2e = strseq2rank_harp(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv_quad({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('econs', 'to', 'vals'): (get_conv(), e2v),
                        ('vals', 'to', 'econs'): (get_conv(), v2e),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                        ('econs', 'to', 'obj'): (get_conv(), e2o),
                        ('obj', 'to', 'econs'): (get_conv(), o2e),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])  # [2*180] + [180]*[4-1] + [1] 
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])  # [2*180] + [180]*[4-1] + [1] 
            self.pred_econs = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]) # [2*180] + [180]*[4-1] + [1] 
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            self.pred_econs = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_econs.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'econs', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            x_dict[k] = x_emb
        hiddens = []
        for _ in range(int(self.ipm_steps)):
            for i in range(self.num_layers):
                if self.share_conv_weight:      
                    i = 0

                h1 = x_dict
                h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
                keys = h2.keys()
                
                if self.use_res:
                    h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
                else:
                    h = {k: F.relu(h2[k]) for k in keys}
                h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
                x_dict = h
            hiddens.append((h2['cons'], h2['econs'], h2['vals']))
        cons, econs, vals = zip(*hiddens)
        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            econs = self.pred_econs(torch.stack(econs, dim=0))
            vals = F.relu(vals)
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.ipm_steps)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.ipm_steps)], dim=1)
            econs = torch.cat([self.pred_econs[i](econs[i]) for i in range(self.ipm_steps)], dim=1)
            vals = F.relu(vals)
            return vals, cons, econs




# TripartiteHeteroGNN_: for our inner+outer loop network structure
class TripartiteHeteroGNN_new(torch.nn.Module):
    def __init__(self,
                 ipm_steps,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel',
                 add_layer=0):
        
        super().__init__()
        
        self.ipm_steps = ipm_steps
        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res
        self.add_layer = add_layer
        
        if pe_dim > 0:
            self.pe_encoder = torch.nn.ModuleDict({
                'vals': MLP([pe_dim, hid_dim, hid_dim]),
                'cons': MLP([pe_dim, hid_dim, hid_dim]),
                'obj': MLP([pe_dim, hid_dim, hid_dim])})
            in_emb_dim = hid_dim
        else:
            self.pe_encoder = None
            in_emb_dim = 2 * hid_dim

#         self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
#                                             'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
#                                             'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})
        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='instance'),
                                            'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='instance'),
                                            'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='instance')})

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, 2 * hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k])
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            x_dict[k] = x_emb

        hiddens = []
        for _ in range(int(self.add_layer)):
            for i in range(self.num_layers):
                if self.share_conv_weight:      
                    i = 0

                h1 = x_dict
                h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
                keys = h2.keys()
                
                if self.use_res:
                    h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
                else:
                    h = {k: F.relu(h2[k]) for k in keys}
                h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
                x_dict = h
            hiddens.append((h2['cons'], h2['vals']))
        cons, vals = zip(*hiddens)
        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            vals = F.relu(vals)
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.ipm_steps)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.ipm_steps)], dim=1)
            vals = F.relu(vals)
            return vals, cons

        
        


class BipartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True, ):
        super().__init__()

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res

        if pe_dim > 0:
            raise NotImplementedError

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, hid_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, hid_dim], norm='batch')})

        get_conv = get_conv_layer(conv, hid_dim, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), 0),
                        ('vals', 'to', 'cons'): (get_conv(), 0),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([hid_dim] * num_pred_layers + [1])
            self.pred_cons = MLP([hid_dim] * num_pred_layers + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([hid_dim] * num_pred_layers + [1]))
                self.pred_cons.append(MLP([hid_dim] * num_pred_layers + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals']:
            x_emb = self.encoder[k](x_dict[k])
            x_dict[k] = x_emb

        hiddens = []
        for i in range(self.num_layers):
            if self.share_conv_weight:
                i = 0

            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            hiddens.append((h2['cons'], h2['vals']))
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h2[k]) for k in keys}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            x_dict = h

        cons, vals = zip(*hiddens)

        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.num_layers)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.num_layers)], dim=1)
        return vals, cons
