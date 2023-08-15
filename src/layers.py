import torch.nn.functional as F
import torch
import numpy as np
import dgl.function as fn
import dgl
import torch.nn as nn
import time


class DRGlobal_net(nn.Module):
    def __init__(self, num_E, h_dim, device):
        super(DRGlobal_net, self).__init__()
        self.num_E = num_E
        self.h_dim = h_dim
        self.device = device
        self.rel_gru = nn.GRUCell(2* self.h_dim, self.h_dim)
        self.rel_gru2 = nn.GRUCell(3 * self.h_dim, self.h_dim)

    def forward(self, g, rel_embs, ent_embs, e_r_bias, num_rels, g_idx=0):
        # r_input = torch.zeros(rel_embs.size(), device=self.device)
        # # concurrent relations
        # for r_type, r_to_r in zip(g.uniq_r, g.r_to_r):
        #     x = torch.index_select(rel_embs, 0, torch.tensor(r_to_r, device=self.device))
        #     x_mean = torch.mean(x, dim=0, keepdim=True)
        #     r_input[r_type] = x_mean
        # r_input = torch.cat((rel_embs, r_input), dim=1)  # (num_rels*2, h_dim*2)
        # if g_idx == 0:
        #     self.h_0 = self.rel_gru(r_input, rel_embs)
        # else:
        #     self.h_0 = self.rel_gru(r_input, self.h_0)

        r_e_input = torch.zeros(rel_embs.size(), device=self.device)
        for r_type, r_to_e in zip(g.uniq_r, g.r_to_e):
            x = torch.index_select(ent_embs, 0, torch.tensor(r_to_e, device=self.device))
            x_mean = torch.mean(x, dim=0, keepdim=True)
            r_e_input[r_type] = x_mean
        r_input = torch.cat((rel_embs, r_e_input), dim=1)  # (num_rels*2, h_dim*2)
        if g_idx == 0:
            self.h_0 = self.rel_gru(r_input, rel_embs)
            self.h_0 = F.normalize(self.h_0)
        else:
            self.h_0 = self.rel_gru(r_input, self.h_0)
            self.h_0 = F.normalize(self.h_0)

        # return F.normalize(self.h_0)
        return self.h_0

class DRlocal_net(torch.nn.Module):
    def __init__(self, in_feat, out_feat, dropout, activation=None):
        super(DRlocal_net, self).__init__()
        self.out_feat = out_feat
        self.in_feat = in_feat
        self.act = activation
        self.dropout = nn.Dropout(dropout)

        self.connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
        nn.init.xavier_uniform_(self.connect_weight, gain=nn.init.calculate_gain('relu'))
        self.connect_bias = nn.Parameter(torch.Tensor(out_feat))
        nn.init.zeros_(self.connect_bias)

        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        self.rel_gru = nn.GRUCell(2* self.in_feat, self.out_feat)


    def forward(self, g, ent_embs, rel_embs, e_r_bias, g_idx):
        # self.rel_embs_0 = rel_embs
        self.ent_embs_0 = ent_embs
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = ent_embs[node_id]
        # g.ndata['e_r_bias'] = e_r_bias[node_id]
        self.propagate(g)
        e_r_bias_new = g.ndata['e_r_bias']
        #
        # skip_weight = F.sigmoid(torch.mm(e_r_bias, self.connect_weight) + self.connect_bias)
        # e_r_bias_new = skip_weight * e_r_bias_new + (1 - skip_weight) * e_r_bias
        # e_r_bias_new = self.act(e_r_bias_new)
        # e_r_bias_new = self.dropout(e_r_bias_new)

        g.ndata.pop('h')
        g.ndata.pop('e_r_bias')
        # return F.normalize(e_r_bias_new)

        # Gru
        e_r_bias_new = torch.cat([e_r_bias_new, e_r_bias], dim=1)
        if g_idx == 0:
            # self.bias_init = torch.zeros_like(e_r_bias_new)
            self.h_0 = self.rel_gru(e_r_bias_new, e_r_bias)
            self.h_0 = self.act(self.h_0)
            self.h_0 = self.dropout(self.h_0)
            self.h_0 = F.normalize(self.h_0)
        else:
            self.h_0 = self.rel_gru(e_r_bias_new, self.h_0)
            self.h_0 = self.act(self.h_0)
            self.h_0 = self.dropout(self.h_0)
            self.h_0 = F.normalize(self.h_0)
        return self.h_0



    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='e_r_bias'), self.apply_func)

    def msg_func(self, edges):
        # relation = self.rel_embs_0.index_select(0, edges.data['type']).view(-1, self.out_feat)  #
        node = edges.src['h'].view(-1, self.out_feat)  #
        # bias = edges.src['e_r_bias'].view(-1, self.out_feat)  #

        msg = node #
        # msg = torch.mul(bias, relation)
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'e_r_bias': nodes.data['e_r_bias'] * nodes.data['out_norm']}  #

class R_GCN(nn.Module):
    def __init__(self, in_feat, out_feat, dropout, activation=None):
        super(R_GCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = nn.Dropout(dropout)
        self.act = activation

        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))  #
        nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
        self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
        nn.init.zeros_(self.skip_connect_bias)  #

        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def msg_func(self, edges):
        relation = self.rel_embs.index_select(0, edges.data['type']).view(-1, self.out_feat)  #
        node = edges.src['h'].view(-1, self.out_feat)  #
        # msg = node + relation + e_r_bias #
        msg = node + relation #
        # msg = torch.mul(node, relation)

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['in_norm']}  #

    def forward(self, g, prev_ent_embs, rel_embs, e_r_bias=None):
        self.rel_embs = rel_embs
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = prev_ent_embs[node_id]  #
        # g.ndata['e_r_bias'] = e_r_bias[node_id]  #

        masked_index = torch.masked_select(
            torch.arange(0, g.number_of_nodes(), dtype=torch.long, device=g.device),  #
            (g.in_degrees(range(g.number_of_nodes())) > 0))  #

        loop_message = torch.mm(g.ndata['h'],
                                self.evolve_loop_weight)  # g.ndata['h']: node embedding (g_num_nodes, h_dim) (h_dim. h_dim)
        loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]

        self.propagate(g)  # calculate the neighbor message with neighbor entity and relation
        node_repr = g.ndata['h']  # node embedding
        # skip_weight = F.sigmoid(torch.mm(prev_ent_embs, self.skip_connect_weight) + self.skip_connect_bias)
        node_repr = node_repr + loop_message
        # node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_ent_embs
        node_repr = self.act(node_repr)  #
        node_repr = self.dropout(node_repr)
        g.ndata.pop('h')
        # g.ndata.pop('e_r_bias')
        return node_repr  # g.ndata['h']

