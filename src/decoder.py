import math
import time

from torch.nn import functional as F
from torch import nn
import torch
import os
import numpy as np

path_dir = os.getcwd()


class ConvTrans_decoder(torch.nn.Module):
    def __init__(self, num_E, h_dim, device, sequence_len=10, alpha=0.5, d_k=64, d_v=64, n_heads=8,
                 d_ff=2048, n_layers_=1, input_dropout=0, hidden_dropout=0,
                 feature_map_dropout=0, channels=50, kernel_size=3):
        super(ConvTrans_decoder, self).__init__()
        self.num_E = num_E
        self.device = device
        self.alpha = alpha

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        # 2通道 (head, relation)
        self.bn0_pre = torch.nn.BatchNorm1d(2)
        self.bn1_pre = torch.nn.BatchNorm1d(channels)
        self.conv1_pre = torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(
            math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)

        # 3通道 (head, relation, DR_local)
        self.bn0 = torch.nn.BatchNorm1d(3)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv1 = torch.nn.Conv1d(3, channels, kernel_size, stride=1, padding=int(
            math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)

        self.bn2 = torch.nn.BatchNorm1d(h_dim)
        self.fc = torch.nn.Linear(h_dim * channels, h_dim)
        self.layers = nn.ModuleList([DecoderLayer(h_dim, d_k, d_v, n_heads, d_ff, device) for _ in range(n_layers_)])
        self.g_mlp = nn.Linear(h_dim, num_E)  # 用于随机生成实体概率


    def forward(self, facts, ent_embs, rel_embs, e_r_bias,
                history_tail_seq=None, one_hot_tail_seq=None,
                freq_info=0,
                basic_ent_embs=None, basic_rel_embs=None):
        # if basic_rel_embs is not None:
        #     E_embs = F.normalize(basic_ent_embs)
        #     E_embs = F.tanh(basic_ent_embs)
        #     e_s = E_embs[facts[:, 0]].unsqueeze(1)
        #     e_r = basic_rel_embs[facts[:, 1]].unsqueeze(1)
        #     stacked_inputs = torch.cat([e_s, e_r], 1)  # (batch_size, 3, h_dim)
        #     stacked_inputs = self.bn0_pre(stacked_inputs)
        #     x0 = self.inp_drop(stacked_inputs)
        #     x0 = self.conv1_pre(x0)
        #     x0 = self.bn1_pre(x0)
        #     x0 = F.relu(x0)
        #     x0 = self.feature_map_drop(x0)
        #     x0 = x0.view(len(facts), -1)  # (batch_size, h_dim*channels)
        #     x0 = self.fc(x0)  # (batch_size, h_dim)
        #     score_basic = torch.mm(x0, basic_ent_embs.transpose(1, 0)).to(self.device)  # 计算得分
        #     score_basic = F.softmax(score_basic, dim=1)


        score_list = []
        for i in range(len(ent_embs)):
            ent_emb_i = ent_embs[-1]
            # ent_emb_i = F.tanh(ent_emb_i)
            rel_emb_i = rel_embs[-1]
            e_r_bias_i = e_r_bias[-1]
        # if 1:
        #     ent_emb_i = ent_embs
        #     # ent_emb_i = F.normalize(ent_emb_i)
        #     ent_emb_i = F.tanh(ent_emb_i)
        #     rel_emb_i = rel_embs
        #     e_r_bias_i = e_r_bias

            e_s = ent_emb_i[facts[:, 0]].unsqueeze(1)
            e_r = rel_emb_i[facts[:, 1]].unsqueeze(1)

            if e_r_bias_i is None:
                stacked_inputs = torch.cat([e_s, e_r], 1)  # discard local dynamics of relations
                stacked_inputs = self.bn0_pre(stacked_inputs)
                x0 = self.inp_drop(stacked_inputs)
                x0 = self.conv1_pre(x0)
                x0 = self.bn1_pre(x0)
            else:
                bias = e_r_bias_i[facts[:, 0]].unsqueeze(1)

                # stacked_inputs = torch.cat([e_s, e_r + bias], 1)
                # stacked_inputs = self.bn0_pre(stacked_inputs)
                # x0 = self.inp_drop(stacked_inputs)
                # x0 = self.conv1_pre(x0)
                # x0 = self.bn1_pre(x0)

                stacked_inputs = torch.cat([e_s, e_r, bias], 1)  # (batch_size, 3, h_dim)
                stacked_inputs = self.bn0(stacked_inputs)
                x0 = self.inp_drop(stacked_inputs)
                x0 = self.conv1(x0)
                x0 = self.bn1(x0)

            x0 = F.relu(x0)
            x0 = self.feature_map_drop(x0)
            x0 = x0.view(len(facts), -1)  # (batch_size, h_dim*channels)
            x0 = self.fc(x0)  # (batch_size, h_dim)


            x = self.hidden_drop(x0)
            if len(facts) > 1:
                x = self.bn2(x)
            x = F.relu(x)
            score = torch.mm(x, ent_emb_i.transpose(1, 0)).to(self.device)
            # score = F.softmax(score, dim=1)
            # score_list.append(score)

            # history statistics information
            if freq_info:  # generation mode
                mask_hist = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float)*(-1000),
                                         device=self.device)
                # score1 = F.softmax(score)
                score1 = F.softmax(score+mask_hist)

                score2 = self.g_mlp(x)
                score2 = F.softmax(score2)
                # score2 = F.softmax(score2+mask_gene)

                score = (score1 * self.alpha + score2 * (1 - self.alpha))
                # score = score1 + score2
                score_list.append(score)
                # final_score = torch.log(final_score)
            else:
                score_list.append(F.softmax(score))
                # score_list.append(score)

            break


        scores_list = [x.unsqueeze(2) for x in score_list]
        score = torch.cat(scores_list, dim=-1)
        score = torch.sum(score, dim=-1)
        score.squeeze()
        # score = F.softmax(score)
        score = torch.log(score)


        return score

    def pretrain_loss(self, facts, facts_neg, E_embs, R_embs, neg_ratio=1):
        E_embs = F.normalize(E_embs)
        E_embs = F.tanh(E_embs)
        e_s = E_embs[facts[:, 0]].unsqueeze(1)
        e_r = R_embs[facts[:, 1]].unsqueeze(1)
        e_o = E_embs[facts[:, 2]].unsqueeze(1)
        e_o_neg = E_embs[facts_neg[:, 2]].unsqueeze(1)  # negative samples
        stacked_inputs = torch.cat([e_s, e_r], 1)  # (batch_size, 3, h_dim)

        stacked_inputs = self.bn0_pre(stacked_inputs)
        x0 = self.inp_drop(stacked_inputs)
        x0 = self.conv1_pre(x0)
        x0 = self.bn1_pre(x0)
        x0 = F.relu(x0)
        x0 = self.feature_map_drop(x0)
        x0 = x0.view(len(facts), -1)  # (batch_size, h_dim*channels)
        x0 = self.fc(x0)  # (batch_size, h_dim)
        e_o_neg = e_o_neg.squeeze()
        e_o = e_o.squeeze()
        loss = torch.zeros(1).to(self.device)
        loss.requires_grad_()
        gap = 6
        for i in range(facts.shape[0]):
            s = torch.dot(x0[i], e_o[i])
            for j in range(i * neg_ratio, i * neg_ratio + neg_ratio):
                s_neg = torch.dot(x0[i], e_o_neg[j])
                if s - s_neg < gap:
                    loss = loss + s_neg + gap - s

        loss = loss / (facts.shape[0] * neg_ratio)
        return loss

    def pretrain_predict(self, facts, E_embs, R_embs):
        E_embs = F.normalize(E_embs)
        E_embs = F.tanh(E_embs)
        e_s = E_embs[facts[:, 0]].unsqueeze(1)
        e_r = R_embs[facts[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e_s, e_r], 1)  # (batch_size, 3, h_dim)
        stacked_inputs = self.bn0_pre(stacked_inputs)
        x0 = self.inp_drop(stacked_inputs)
        x0 = self.conv1_pre(x0)
        x0 = self.bn1_pre(x0)
        x0 = F.relu(x0)
        x0 = self.feature_map_drop(x0)
        x0 = x0.view(len(facts), -1)  # (batch_size, h_dim*channels)
        x0 = self.fc(x0)  # (batch_size, h_dim)
        scores = torch.mm(x0, E_embs.transpose(1, 0)).to(self.device)
        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self, h_dim, d_k, d_v, n_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(h_dim, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(h_dim, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(h_dim, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, h_dim, bias=False)
        self.SDPAtt = ScaledDotProductAttention(d_k)
        self.h_dim = h_dim
        self.d_k = d_k
        self.d_v = d_v  # d_k = d_v = d_q
        self.n_heads = n_heads
        self.device = device

    def forward(self, input_Q, input_K, input_V):  # enc_query_inputs, enc_history_inputs, enc_history_inputs
        '''
        input_Q: (batch_size, 1, h_dim)
        input_K: (batch_size, k, h_dim)
        input_V: (batch_size, k, h_dim)
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: (batch_size, n_heads, 1, d_k)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: (batch_size, n_heads, 3, d_k)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: (batch_size, n_heads, 3, d_v)

        context = self.SDPAtt(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: (batch_size, 1, n_heads * d_v)
        output = self.fc(context)  # (batch_size, 1, h_dim)
        return nn.LayerNorm(self.h_dim).to(self.device)(output + residual)  # (batch_size, 1, h_dim)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        '''
        Q: (batch_size, n_heads, 1, d_k)
        K: (batch_size, n_heads, 3, d_k)
        V: (batch_size, n_heads, 3, d_v)
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : (batch_size, n_heads, 1, 3)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # (batch_size, n_heads, 1, d_v)
        return context  # (batch_size, n_heads, 1, d_v)


class FeedForwardNet(nn.Module):
    def __init__(self, h_dim, d_ff, device):
        super(FeedForwardNet, self).__init__()
        self.h_dim = h_dim
        self.fc = nn.Sequential(
            nn.Linear(h_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, h_dim, bias=False)
        )
        self.device = device

    def forward(self, inputs):
        '''
        inputs: (batch_size, 1, h_dim)
        '''
        residual = inputs
        output = self.fc(inputs)
        # nn.LayerNorm(x)会对输入的最后一维进行归一化, x需要和输入的最后一维一样大
        return nn.LayerNorm(self.h_dim).to(self.device)(output + residual)  # (batch_size, 1, h_dim_2)


class DecoderLayer(nn.Module):
    def __init__(self, h_dim, d_k, d_v, n_heads, d_ff, device):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(h_dim, d_k, d_v, n_heads, device)
        self.pos_ffn = FeedForwardNet(h_dim, d_ff, device)

    def forward(self, dec_query, dec_K, dec_V):
        '''
        dec_query_inputs: (batch_size, 1, h_dim)
        dec_history_iutputs: (batch_size, k, h_dim) 由dec_inputs得到，长度最长是k=10
        '''
        # enc_query_inputs生成Q矩阵, enc_history_inputs生成K, V矩阵
        dec_outputs = self.dec_self_attn(dec_query, dec_K, dec_V)  # (batch_size, 1, h_dim)
        dec_query_outputs = self.pos_ffn(dec_outputs)  # (batch_size, 1, h_dim)
        return dec_query_outputs  # dec_query_outputs: (batch_size, 1, h_dim) dec_history_outputs: (batch_size, 3, h_dim)
