import torch.nn as nn
import torch
import numpy as np
from layers import *
from decoder import ConvTrans_decoder
import time


class ERDNet(nn.Module):
    def __init__(self, num_E, num_R, h_dim, n_layers, dropout, input_dropout, hidden_dropout, feature_map_dropout,
                 device, alpha):
        super(ERDNet, self).__init__()
        self.num_E = num_E
        self.num_R = num_R
        self.device = device
        self.h_dim = h_dim

        self.ent_embs = nn.Parameter(torch.Tensor(num_E, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.ent_embs)
        self.rel_embs = nn.Parameter(torch.Tensor(num_R * 2, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.rel_embs)
        self.e_r_bias = nn.Parameter(torch.Tensor(num_E, h_dim), requires_grad=True).float()
        torch.nn.init.zeros_(self.e_r_bias)

        self.DRR_layer = DRGlobal_net(num_E, h_dim, device)
        self.BiasGCN_layers = nn.ModuleList(
            [DRlocal_net(h_dim, h_dim, dropout, activation=F.rrelu) for i in range(n_layers)])

        self.RGCN_layers = nn.ModuleList(
            [R_GCN(h_dim, h_dim, dropout, activation=F.rrelu) for i in range(n_layers)])

        self.decoder = ConvTrans_decoder(num_E, h_dim, device, input_dropout=input_dropout,
                                         hidden_dropout=hidden_dropout,
                                         feature_map_dropout=feature_map_dropout, alpha=alpha)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.loss_nll = torch.nn.NLLLoss()

    def forward(self, hist_glist, use_bias=0):  # different window size
        ent_embs = F.normalize(self.ent_embs)
        rel_embs = F.normalize(self.rel_embs)
        e_r_bias = F.normalize(self.e_r_bias)
        # rel_embs = self.rel_embs
        # e_r_bias = self.e_r_bias

        ent_embs_list = []
        rel_embs_list = []
        e_r_bias_list = []

        for i, g in enumerate(hist_glist):
            # ---------------------
            tmp_ent_embs = ent_embs
            for layer in self.RGCN_layers:  #
                tmp_ent_embs = layer(g, tmp_ent_embs, rel_embs)
                tmp_ent_embs = F.normalize(tmp_ent_embs)
                break
            time_weight = F.sigmoid(torch.mm(ent_embs, self.time_gate_weight) + self.time_gate_bias)
            ent_embs = time_weight * tmp_ent_embs + (1 - time_weight) * ent_embs
            ent_embs = F.normalize(ent_embs)

            # ent_embs = tmp_ent_embs

            # rel_embs = self.DRR_layer(g, rel_embs, ent_embs, e_r_bias, self.num_R, i)

            if use_bias:
                for layer in self.BiasGCN_layers:
                    e_r_bias = layer(g, ent_embs, rel_embs, e_r_bias, i)
                    break
            else:
                e_r_bias = None

            ent_embs_list.append(ent_embs)
            rel_embs_list.append(rel_embs)
            e_r_bias_list.append(e_r_bias)
            # e_r_bias_list.append(None)

        # return ent_embs, rel_embs, e_r_bias
        return ent_embs_list, rel_embs_list, e_r_bias_list

    def loss_func(self, ghist, facts, facts_neg=None, history_tail_seq=None, one_hot_tail_seq=None, use_bias=0,
                  freq_info=0, neg_ratio=1):
        if ghist is not None:
            ent_embs_list, rel_embs_list, e_r_bias_list = self.forward(ghist, use_bias)
            score = self.decoder(facts,
                                     ent_embs_list, rel_embs_list, e_r_bias_list,
                                     history_tail_seq, one_hot_tail_seq,
                                     freq_info,
                                     self.ent_embs, self.rel_embs)
            loss = self.loss_nll(score, facts[:,2])
            return loss
        else:  # 预训练
            loss = self.decoder.pretrain_loss(facts, facts_neg,
                                              self.ent_embs, self.rel_embs,
                                              neg_ratio=neg_ratio)
            return loss

    def predict(self, ghist, num_R, facts, history_tail_seq=None, one_hot_tail_seq=None, freq_info=0, use_bias=0):
        with torch.no_grad():
            if ghist is None:
                scores = self.decoder.pretrain_predict(facts, self.ent_embs, self.rel_embs)
                return facts, scores

            ent_embs_list, rel_embs_list, e_r_bias_list = self.forward(ghist, use_bias)

            score = self.decoder(facts,
                                     ent_embs_list, rel_embs_list, e_r_bias_list,
                                     history_tail_seq, one_hot_tail_seq,
                                     freq_info,
                                     self.ent_embs, self.rel_embs)


            return facts, score
