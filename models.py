from embedding import *
from collections import OrderedDict
import torch
import torch.nn.functional as F
# from relational_path_gnn import RelationalPathGNN
class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        # self.relation_learner = nn.Linear(input_dim, output_dim)
        # parameter['few'], embed_size = 50, num_hidden1 = 250,
        # num_hidden2 = 100, out_size = 50, dropout_p = self.dropout_p
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

        # xavier初始化，也称为Glorot初始化，是一种在训练深度学习模型时用于初始化网络权重的策略。
        # 其核心思想是保持每一层输出的方差与输入的方差一致，以防止信号在深层网络中的爆炸或消失。
        # 如果方差过大，那么网络的层将会更难以学习；如果方差过小，那么该层的权重将会难以更新。

    def forward(self, inputs):
        size = inputs.shape
        # print(size)
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)   #按行计算平均值（沿着第1维）
#计算均值。这表示将每个样本的多个输出值取平均化，从而减少特征维度。
        # print(x.shape)
        return x.view(size[0], 1, 1, self.out_size)
        # return x.view(size[0], 1, 1,-1)

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score
class MarginRankingLoss(nn.MarginRankingLoss):
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super().__init__(margin=margin, reduction=reduction)

    def forward(self, p_score, n_score, y, reduction: str = None):
        reduction = reduction if reduction is not None else self.reduction
        return F.margin_ranking_loss(p_score, n_score, y, margin=self.margin, reduction=reduction)

class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']#5
        self.dropout_p = parameter['dropout_p']#0.5
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']#1
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)

        # self.r_path_gnn = RelationalPathGNN(g, dataset['ent2id'], len(dataset['rel2emb']), parameter)

        if parameter['dataset'] == 'primkg-assistant(disease)':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        self.embedding_learner = EmbeddingLearner()  #返回正样本得分和fen负样本得分
        self.loss_func = MarginRankingLoss(self.margin)
        self.extract_relations = self.extract_relations
        self.relation_learner = self.relation_learner.to(self.device)
        self.rel_q_sharing = dict()

    def extract_relations(task):
        support, support_negative, query, negative = task

        rels = set()

        # support 是 tuple，每个元素是一个三元组列表
        for support_set in support:
            if isinstance(support_set, list):
                for triple in support_set:
                    if isinstance(triple, list) and len(triple) == 3:
                        rels.add(triple[1])  # 添加关系部分

        return list(rels)
    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel='', support_weight_indices=None):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        # support, support_negative, query, negative = [self.r_path_gnn(t) for t in task]
#将输入的 task 中的支持集（support）、支持集负样本（support_negative）、vb
# 查询集（query）和查询集负样本（negative）通过 self.embedding 方法转换为嵌入表示。
#         support, support_negative = [self.embedding(t) for t in task]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        rel = self.relation_learner(support)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_relation,sup_neg_e2 = self.split_concat(support, support_negative)
                p_score, n_score = self.embedding_learner(rel_s, sup_neg_e2, sup_neg_relation, few)
                y = torch.ones_like(p_score).to(self.device)
                y = y.to(self.device)  # 确保 target 在正确的设备上
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                #----------------------------------------------------------
                # 🔐 只有在 rel.requires_grad=True 时才调用 retain_grad
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
            else:
                rel_q = rel
            self.rel_q_sharing[curr_rel] = rel_q
        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        que_neg_relation, que_neg_e2 = self.split_concat(query, negative)
        p_score, n_score = self.embedding_learner(rel_q, que_neg_e2, que_neg_relation, num_q)
        return p_score, n_score
