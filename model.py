import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)




class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        scs.append(self.f_k(h_pl, c))

        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        # 1. GCN Layers
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)

        # 2. Classifier
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)

        self.act = nn.ReLU()

        self.risk_scale = nn.Parameter(torch.tensor(5.0))  
        self.risk_bias = nn.Parameter(torch.tensor(-2.5))

        self.disc = Discriminator(n_h, negsamp_round)
    
    def get_risk_weighted_peers(self, emb, partition, risk_scores, num_samples, idx_train, normal_idx):

        device = emb.device
        N = emb.size(0)

        node_weights = risk_scores[partition] + 1e-6 

        train_mask = torch.zeros(N, device=device) 

        train_mask[idx_train] = 1.0
        train_mask[normal_idx] = 0.0 
        

        node_weights = node_weights * train_mask

        sampled_indices = torch.multinomial(node_weights, num_samples, replacement=True)

        peer_emb = emb[sampled_indices]
        
        return peer_emb

    def forward(self, seq1, adj, partition, risk_scores, sample_abnormal_idx, normal_idx, idx_train, train_flag, args, sparse=False):
        
        h_1 = self.gcn1(seq1, adj, sparse)
        emb = self.gcn2(h_1, adj, sparse)  # (1, N, n_h)
        
        node_emb = emb[0] # (N, n_h)
        emb_con = None
        emb_combine = None
        
        emb_target = node_emb[sample_abnormal_idx] 
        
        emb_abnormal = emb_target.unsqueeze(0) 

        if train_flag:            
            num_targets = len(sample_abnormal_idx)
    
            emb_peers = self.get_risk_weighted_peers(node_emb, partition, risk_scores, num_targets, idx_train, normal_idx)  

            target_comm_ids = partition[sample_abnormal_idx]
            target_risks = risk_scores[target_comm_ids].unsqueeze(1) # [Batch, 1]

            inverse_risk = 1.0 - target_risks
            
            alpha = torch.sigmoid(inverse_risk * self.risk_scale + self.risk_bias)

            emb_con = (1 - alpha) * emb_target + alpha * emb_peers

            random_noise = torch.randn_like(emb_con) * args.mean
            # random_noise = torch.randn(emb_con.size()) * args.var + args.mean
            emb_con = emb_con + random_noise

            emb_normal_batch = node_emb[normal_idx]
            emb_combine = torch.cat((emb_normal_batch, emb_con), 0)
            emb_combine = emb_combine.unsqueeze(0)    # (1, N_batch, n_h)

            f_1 = self.act(self.fc1(emb_combine))
            f_2 = self.act(self.fc2(f_1))
            f_3 = self.fc3(f_2)
        
        else:
            f_1 = self.act(self.fc1(emb))
            f_2 = self.act(self.fc2(f_1))
            f_3 = self.fc3(f_2)

        return emb, emb_combine, f_3, emb_con, emb_abnormal


