import random
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.curr_rel_idx = 0
        self.tasks = dataset[step + '_tasks']
        # self.rel2candidates = dataset['rel2candidates']
        self.e1candidates = dataset['e1candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        # self.e1_rele2 = dataset['e1_rele2']
        self.all_rels = sorted(list(self.tasks.keys()))  # 获取字典self.tasks中的所有键（key）
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']

        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)  # 从每个类别中提取few到所有类别样本数目
            self.curr_tri_idx = 0

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        # curr_cand = self.rel2candidates[curr_rel]
        curr_cand = self.e1candidates[curr_rel]
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            # 候选项：来源是self.rel2candidates，它提供了一个关系对应的实体2候选集合，用于构建负样本。
            # 任务：来源是self.tasks，它代表了所有与当前关系相关的正样本三元组。
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            # curr_cand = self.rel2candidates[curr_rel]
            curr_cand = self.e1candidates[curr_rel]  # 提取到当前头实体的候选尾实体集合
        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq)  # 几few-shot+几query，为当前任务的curr_tasks_idx索引
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        # ✅ 找出 support_triples 中 rel 为 indication 或 contraindication 的索引
        query_weight_indices = [
            i for i, (_, rel, _) in enumerate(query_triples)
            if rel in ["indication", "contraindication"]
        ]
        support_weight_indices = [
            i for i, (_, rel, _) in enumerate(support_triples)
            if rel in ["indication", "contraindication"]
        ]
        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)  # 尾实体
                if (negative not in self.e1rel_e2[str(e1) + rel]) \
                        and negative != e2:
                    break
                # if (negative not in self.e1_rele2[rel + e2]) \
                #         and negative != e1:
                #     break
            support_negative_triples.append([e1, rel, negative])
            # support_negative_triples.append([negative, rel, e2])
        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[str(e1) + rel]) \
                        and negative != e2:
                    break

            negative_triples.append([e1, rel, negative])
            # negative_triples.append([negative, rel, e2])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel,query_weight_indices,support_weight_indices

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel, query_weight_indices,support_weight_indices = zip(*next_batch_all)
        return [support, support_negative, query, negative], curr_rel, query_weight_indices,support_weight_indices

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[0]
        # curr_cand = self.rel2candidates[curr_rel]
        curr_cand = self.e1candidates[curr_rel]  # 提取到当前头实体的候选尾实体集合
        curr_task = self.tasks[curr_rel]  # 获取当前关系的任务集

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negativ
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                # if (negative not in self.e1_rele2[rel + e2]) \
                #         and negative != e1:
                #     break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])
            # if (negative not in self.e1_rele2[rel + e2]) \
            #         and negative != e1:
            #     negative_triples.append([negative, rel, e2])
        # print(len(support_triples), len(support_negative_triples), len(query_triple), len(negative_triples))
        # 训练时打印的mrr等值,遍历curr_cand中所有候选
        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]
        # print(support_triples, support_negative_triples, query_triple, negative_triples)
        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[0]
        # curr_cand = self.rel2candidates[curr_rel]
        curr_cand = self.e1candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                # if (negative not in self.e1_rele2[rel + e2]) \
                #         and negative != e1:
                #     break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])  # 为每一个支持三元组构造一个负样本

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[str(e1) + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])
            # if (negative not in self.e1_rele2[rel + e2]) \
            #         and negative != e1:
            #     negative_triples.append([negative, rel, e2])

        # print(len(support_triples), len(support_negative_triples), len(query_triple), len(negative_triples))
        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]

        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    # def next_support_only(self):
    #     """
    #     Return only one task: support set and support negatives for the specific (head, relation) pair.
    #     """
    #     # if self.support_sampled:
    #     #     return "EOT", "EOT"
    #
    #     # rel = "indication"  # e.g., "83914"
    #     e1 = "83914"  # e.g., "indication"
    #     curr_cand = self.e1candidates[e1]
    #     support_triples = self.tasks[e1]  # List[List[head, rel, tail]]
    #     candidates = self.e1candidates[e1]  # List of tail candidates
    #     support_negative_triples = []
    #
    #     shift = 0
    #     for triple in support_triples:
    #         e1, rel, e2 = triple
    #         while True:
    #             negative = curr_cand[shift]
    #             if (negative not in self.e1rel_e2[e1 + rel]) \
    #                     and negative != e2:
    #                 break
    #             else:
    #                 shift += 1
    #         support_negative_triples.append([e1, rel, negative])  # 为每一个支持三元组构造一个负样本
    #
    #     # wrap to keep shape [1, N, 3]
    #     support_triples = [support_triples]
    #     support_negatives = [support_negative_triples]
    #
    #     # self.support_sampled = True  # Only sample once
    #     return [support_triples, support_negatives], e1
