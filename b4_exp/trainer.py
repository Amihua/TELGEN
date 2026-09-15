from functools import partial

import numpy as np
import torch
from torch_scatter import scatter
import copy

from data.utils import barrier_function
import time


class Trainer:
    def __init__(self,
                 device,
                 loss_target,
                 loss_type,
                 micro_batch,
                 ipm_steps,
                 ipm_alpha,
                 loss_weight):
        assert 0. <= ipm_alpha <= 1.
        self.ipm_steps = ipm_steps           # number of steps being supervised
        self.ipm_alpha = ipm_alpha
        self.step_weight = torch.tensor([ipm_alpha ** (ipm_steps - l - 1)
                                         for l in range(ipm_steps)],
                                        dtype=torch.float, device=device)[None]    # step decay factor
        self.device = device
        # self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0
        self.device = device
        self.loss_target = loss_target.split('+')
        self.loss_weight = loss_weight
        if loss_type == 'l2':
            self.loss_func = partial(torch.pow, exponent=2)
        elif loss_type == 'l1':
            self.loss_func = torch.abs
        else:
            raise ValueError
        self.micro_batch = micro_batch

    
    # self.get_loss_: loss, primal_loss, obj_loss, cons_loss
    def train_(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        primal_losses = 0.
        obj_losses = 0.
        cons_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)    # predicted obj and con
            loss, primal_loss, obj_loss, cons_loss = self.get_loss_(vals, data)

            train_losses += loss.detach() * data.num_graphs
            primal_losses += primal_loss.detach() * data.num_graphs
            obj_losses += obj_loss.detach() * data.num_graphs
            cons_losses += cons_loss.detach() * data.num_graphs
            
            num_graphs += data.num_graphs
            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs, primal_losses.item() / num_graphs, obj_losses.item() / num_graphs, cons_losses.item() / num_graphs
    
     
    def get_loss_(self, vals, data):
        loss = 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target: 
            constraint_gap = self.get_constraint_violation(vals, data)
            cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss, primal_loss, obj_loss, cons_loss
    
    
    def get_constraint_violation(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        constraint_gap = Ax - data.rhs[:, None]                   # data.rhs=b
        constraint_gap = torch.relu(constraint_gap)               # ax-b > 0, gap>0, otw, gap=0
        return constraint_gap

    
    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.obj_const[:, None] * pred                   # data.obj_const = c
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')     # scatter(src, index, dim, reduce): inplace matrix operation on src using index and reduce, add all ipm step gap for each instance
        x_gt = data.gt_primals[:, -self.ipm_steps:]                  # data.gt_primals = the intermediate values of x (decision variables) in inters
        c_times_xgt = data.obj_const[:, None] * x_gt                
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt                          # this gap is a vector of ipm_steps length for every instances

    
    def obj_metric(self, dataloader, model):
        model.eval()

        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)

    
    def constraint_metric(self, dataloader, model):
        """
        minimize ||Ax - b||^p in case of equality constraints
         ||relu(Ax - b)||^p in case of inequality

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))

        return np.concatenate(cons_gap, axis=0)
    

    # eval_metrics_: return normalized obj gap 
    @torch.no_grad()
    def eval_metrics_(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        objs_gap = []
        objs_nocgap = []
        b_vals = []
        ttt = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            
            tt = time.time()
            vals, _ = model(data)
            ttt += time.time()-tt
            
            con_gap = self.get_constraint_violation(vals, data)
            obj_gap = self.get_obj_metric(data, vals, hard_non_negative=True)
            
            if torch.max(con_gap) > 0:
                obj_nocgap = obj_gap * (1/(1+torch.max(con_gap)))
            else:
                obj_nocgap = obj_gap
                
            cons_gap.append(np.abs(con_gap.detach().cpu().numpy()))
            objs_gap.append(np.abs(obj_gap.detach().cpu().numpy()))
            objs_nocgap.append(np.abs(obj_nocgap.detach().cpu().numpy()))
        print('time used:', ttt)
            
        objs_gap = np.concatenate(objs_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        objs_nocgap = np.concatenate(objs_nocgap, axis=0)
        return objs_gap, cons_gap, objs_nocgap
    
