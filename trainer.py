
from models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from params import *
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import csv
import os
from tqdm import tqdm
# import dgl


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

class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters

        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        # epoch

        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']

        # kg = self.build_kg(dataset['ent2emb'], dataset['rel2emb'], max_=100)

        self.metaR = MetaR(dataset, parameter)
        self.metaR.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''


        # logging
        logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        # logging.basicConfig(filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

        # load state_dict and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()


    def build_kg(self, ent_emb, rel_emb, max_=100):
        print("Build KG...")
        src = []
        dst = []
        e_feat = []
        e_id = []
        with open(self.data_path + '/path_graph1.txt') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split('\t')
                src.append(self.ent2id[e1])
                dst.append(self.ent2id[e2])
                e_feat.append(rel_emb[self.rel2id[rel]])
                e_id.append(self.rel2id[rel])
                # Reverse
                # src.append(self.ent2id[e2])
                # dst.append(self.ent2id[e1])
                # e_feat.append(rel_emb[self.rel2id[rel + '_inv']])
                # e_id.append(self.rel2id[rel + '_inv'])

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        kg = dgl.graph((src, dst))
        kg.ndata['feat'] = torch.FloatTensor(ent_emb)
        kg.edata['feat'] = torch.FloatTensor(np.array(e_feat))
        kg.edata['eid'] = torch.LongTensor(np.array(e_id))
        return kg
    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    # def do_one_step(self, task, iseval=False, curr_rel=''):
    #     loss, p_score, n_score = 0, 0, 0
    #
    #     if not iseval:
    #         self.optimizer.zero_grad()
    #         p_score, n_score = self.metaR(task, iseval, curr_rel)
    #         relations_in_this_task = extract_relations(task)
    #         # if "indication" in relations_in_this_task or "contraindication" in relations_in_this_task:
    #
    #             # y = torch.Tensor([1]).to(self.device)
    #         y = torch.ones_like(p_score).to(self.device)
    #         loss = self.metaR.loss_func(p_score, n_score, y)
    #         loss.backward()
    #         self.optimizer.step()
    #         # else:
    #         #     # 忽略非目标关系，loss设为 0，仅作为占位值，不更新梯度
    #         #     loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
    #
    #     elif curr_rel != '':
    #         p_score, n_score = self.metaR(task, iseval, curr_rel)
    #         # y = torch.Tensor([1]).to(self.device)
    #         y = torch.ones_like(p_score).to(self.device)
    #         loss = self.metaR.loss_func(p_score, n_score, y)
    #     return loss, p_score, n_score
    def do_one_step(self, task, iseval=False, curr_rel='', query_weight_indices=None, support_weight_indices=None):
        loss, p_score, n_score = 0, 0, 0
            # relations_in_this_task = extract_relations(task)
            # # # 是否包含目标关系
            # has_target_relation = any(
            #     rel in ["indication", "contraindication"] for rel in relations_in_this_task
            # )
        if query_weight_indices is not None:
            bs, num_query = len(task[0]), len(task[2][0])  # batch_size, 每个任务的 query 数量
            weight_matrix = []
            for task_id in range(bs):
                indices = set(query_weight_indices[task_id])  # 该 task 中应保留的 query 索引
                row_weights = [
                    1.0 if i in indices else 0.08 for i in range(num_query)
                ]
                weight_matrix.append(row_weights)  # 每一行为一个 task 的 query 权重
            weight_tensor = torch.tensor(weight_matrix, dtype=torch.float32).to(self.device)  # shape: [bs, num_query]
        else:
            # fallback: 全部设为 1，shape [bs, num_query]
            bs, num_query = len(task[0]), len(task[2][0])  # batch_size, 每个任务的 query 数量
            weight_tensor = torch.ones((bs, num_query), dtype=torch.float32).to(self.device)
            #     👉 目标关系：启用 autograd
            # 无论是否是目标关系，前向传播都执行（用于 rel_q 共享、embedding 学习等）

        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.metaR(task, iseval=False, curr_rel=curr_rel,support_weight_indices=support_weight_indices)
            y = torch.ones_like(p_score).to(self.device)
            # loss = self.metaR.loss_func(p_score, n_score, y)
            # loss.backward()
            # self.optimizer.step()
            loss_per_sample = self.metaR.loss_func(p_score, n_score, y, reduction='none')
            weighted_loss = (loss_per_sample * weight_tensor).mean()
            weighted_loss.backward()
            self.optimizer.step()
            # else:
            #     # ❌ 非目标关系：禁用 autograd，避免显存泄露
            # with torch.no_grad():
            #     _ = self.metaR(task, iseval=False, curr_rel=curr_rel)
            #     loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        elif curr_rel != '':
            p_score, n_score = self.metaR(task, iseval=True, curr_rel=curr_rel, support_weight_indices=support_weight_indices)
            y = torch.ones_like(p_score).to(self.device)
            # loss = self.metaR.loss_func(p_score, n_score, y)
            loss_per_sample = self.metaR.loss_func(p_score, n_score, y, reduction='none')
            weighted_loss = (loss_per_sample * weight_tensor).mean()
        return weighted_loss, p_score, n_score

    def plot_curves(self, labels, scores):
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)

        # 创建保存路径
        plot_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # 绘制 ROC 曲线
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.3f})'.format(roc_auc_score(labels, scores)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'roc_curve.png'))
        plt.close()

        # 绘制 PR 曲线
        plt.figure()
        plt.plot(recall, precision, label='PR curve (AU-PR = {:.3f})'.format(average_precision_score(labels, scores)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'pr_curve.png'))
        plt.close()

    def plot_bin_performance(self, bin_metrics, bin_counts):
        bins = sorted(bin_metrics.keys())
        metrics = ['MRR', 'Hits@1', 'Hits@5', 'Hits@10', 'AUC', 'AU-PR']

        plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = [bin_metrics[b][metric] / bin_counts[b] if bin_counts[b] > 0 else 0 for b in bins]
            plt.plot(bins, values, label=metric, marker='o')

        plt.xlabel("Disease Node Degree (binned)")
        plt.ylabel("Metric Score")
        plt.title("Evaluation Metrics vs. Disease Degree")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("degree_performance.png")
        print("📊 保存性能-度数曲线图为 degree_performance.png")

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_performance_by_disease(self, head_entity_metrics, output_dir='Y:/YL/MetaR-master/log/disease_metrics_plots'):
        os.makedirs(output_dir, exist_ok=True)

        metrics_keys = ['MRR', 'Hits@1', 'Hits@5', 'Hits@10', 'AUC', 'AU-PR']
        diseases = list(head_entity_metrics.keys())

        for metric in metrics_keys:
            values = []
            for d in diseases:
                val = head_entity_metrics[d][metric]
                if isinstance(val, list):
                    if len(val) > 0:
                        val = np.mean(val)
                    else:
                        val = 0.0
                values.append(val)

            plt.figure(figsize=(max(10, len(diseases) * 0.4), 6))
            plt.bar(diseases, values, color='skyblue')
            plt.xticks(rotation=90, fontsize=8)
            plt.ylabel(metric)
            plt.title(f'{metric} per Disease')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(output_dir, f'{metric}_per_disease.png'))
            plt.close()

    # def save_head_entity_metrics_csv(head_entity_metrics, output_path="head_entity_metrics.csv"):
    #     fieldnames = ["Head_Entity", "MRR", "Hits@1", "Hits@5", "Hits@10", "AUC", "AU-PR", "Count"]
    #     with open(output_path, mode="w", newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for head, metrics in head_entity_metrics.items():
    #             writer.writerow({
    #                 "Head_Entity": head,
    #                 "MRR": round(metrics['MRR'] / metrics['count'], 4) if metrics['count'] > 0 else 0,
    #                 "Hits@1": round(metrics['Hits@1'] / metrics['count'], 4) if metrics['count'] > 0 else 0,
    #                 "Hits@5": round(metrics['Hits@5'] / metrics['count'], 4) if metrics['count'] > 0 else 0,
    #                 "Hits@10": round(metrics['Hits@10'] / metrics['count'], 4) if metrics['count'] > 0 else 0,
    #                 "AUC": metrics['AUC'],
    #                 "AU-PR": metrics['AU-PR'],
    #                 "Count": metrics['count']
    #             })
    #     print(f"✅ Saved per-head metrics to: {os.path.abspath(output_path)}")

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0
        real_epoch =0


        # training by epoch
        for e in range(self.epoch):
            # sample one batch from data_loader
            train_task, curr_rel, query_weight_indices, support_weight_indices = self.train_data_loader.next_batch()
            loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel, query_weight_indices=query_weight_indices, support_weight_indices=support_weight_indices)

            if e % self.print_epoch == 0:
                loss_num = loss.item()
                # if loss.item() > 0:
                self.write_training_log({'Loss': loss_num}, e)
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))

                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)

                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] >= best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        print('Finish')
    # def train(self):
    #      best_epoch = 0
    #      best_loss = float('inf')  # 用于保存最低loss
    #      real_epoch = 0
    #
    #      for e in range(self.epoch):
    #          # 采样一个任务
    #          train_task, curr_rel, query_weight_indices, support_weight_indices = self.train_data_loader.next_batch()
    #          loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel,
    #                                        query_weight_indices=query_weight_indices,
    #                                        support_weight_indices=support_weight_indices)
    #
    #          if e % self.print_epoch == 0:
    #              loss_num = loss.item()
    #              self.write_training_log({'Loss': loss_num}, e)
    #              print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
    # #
    #          # 保存中间模型
    #          # if e % self.checkpoint_epoch == 0 and e != 0:
    #          #     print(f'Epoch {e} finished, saving checkpoint...')
    #          #     self.save_checkpoint(e)
    #
    #          # 更新最优模型（基于最小loss）
    #          loss_num = loss.item()
    #          if loss_num < best_loss:
    #              best_loss = loss_num
    #              best_epoch = e
    #              print(f'\tNew best model at epoch {e} | Loss = {best_loss:.4f}')
    #              self.save_checkpoint(best_epoch)
    #      print('Training finished.')
    #      print(f'\tBest epoch is {best_epoch} | Loss = {best_loss:.4f}')
    #      self.save_best_state_dict(best_epoch)
    #      print('Finish')

    def eval(self, istest=False, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # 读取测试文件统计疾病节点度数
        if istest:
            # import json
            with open(data_dir['test_tasks'], 'r') as f:
                test_tasks = json.load(f)
            self.head_entity_degrees = {head: len(triples) for head, triples in test_tasks.items()}

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'AUC': 0, 'AU-PR': 0}
        ranks = []
        # 存储预测得分和标签用于绘图
        all_scores = []
        all_labels = []
        all_rankings = dict()
        # 每个疾病的性能记录（以 head 为单位）
        head_entity_metrics = defaultdict(
            lambda: {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'AUC': 0, 'AU-PR': 0, 'count': 0, 'scores': [], 'labels': []})

        t = 0
        temp = dict()

        # 按度数区间初始化评估容器
        degree_bins = list(range(5, 55, 1))  # 5~50, 每5一组
        bin_metrics = {deg: defaultdict(float) for deg in degree_bins}
        bin_counts = {deg: 0 for deg in degree_bins}
        # with open("Y:/primkg-assistant(disease)/e1candidates-processed-disease-all-other-200.json","r", encoding="utf-8") as f:
        #     e1candidates = json.load(f)

        # 获取 head entity 为 "83914" 的所有候选尾实体（药物）
        # candidates = e1candidates["83914"]

        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1
            head = eval_task[3][0][0][0]
            _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)
            x = torch.cat([n_score, p_score], 1).squeeze()
            y_true = torch.cat([torch.zeros_like(n_score), torch.ones_like(p_score)], dim=1).squeeze()
            all_scores.extend(x.detach().cpu().numpy().tolist())
            all_labels.extend(y_true.detach().cpu().numpy().tolist())
            self.rank_predict(data, x, ranks)
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()
            if istest and head is not None:
                head_entity_metrics[head]['MRR'] += data['MRR'] / t
                head_entity_metrics[head]['Hits@1'] += data['Hits@1'] / t
                head_entity_metrics[head]['Hits@5'] += data['Hits@5'] / t
                head_entity_metrics[head]['Hits@10'] += data['Hits@10'] / t
                head_entity_metrics[head]['scores'].extend(x.detach().cpu().numpy().tolist())
                head_entity_metrics[head]['labels'].extend(y_true.detach().cpu().numpy().tolist())
                head_entity_metrics[head]['count'] += 1
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        auc = roc_auc_score(all_labels, all_scores)
        aupr = average_precision_score(all_labels, all_scores)
        data['AUC'] = round(auc, 3)
        data['AU-PR'] = round(aupr, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\tAUC: {:.3f}\tAU-PR: {:.3f}".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1'], data['AUC'], data['AU-PR']))
        # 打印每个疾病（head）的指标
        if istest:
            print("\n\nPer-disease evaluation (head entity):")
            print(f"{'Head Entity':<20}{'MRR':<8}{'H@1':<8}{'H@5':<8}{'H@10':<8}{'AUC':<8}{'AU-PR':<8}")
            for head, metrics in head_entity_metrics.items():
                count = metrics['count'] if metrics['count'] > 0 else 1
                mrr = metrics['MRR'] / count
                h1 = metrics['Hits@1'] / count
                h5 = metrics['Hits@5'] / count
                h10 = metrics['Hits@10'] / count
                scores = metrics['scores']
                labels = metrics['labels']

                if len(scores) > 0 and len(set(labels)) > 1:
                    try:
                        auc = roc_auc_score(labels, scores)
                        aupr = average_precision_score(labels, scores)
                    except:
                        auc, aupr = 0.0, 0.0
                else:
                    auc, aupr = 0.0, 0.0
                degree = self.head_entity_degrees.get(head, 0)

                print(f"{head:<20}{degree:<8}{mrr:<8.3f}{h1:<8.3f}{h5:<8.3f}{h10:<8.3f}{auc:<8.3f}{aupr:<8.3f}")

        return data

    def eval_by_relation(self, istest=False, epoch=None):

        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'AUC': 0, 'AU-PR': 0}
        all_t = 0
        all_ranks = []

        for rel in data_loader.all_rels:
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                rel, len(data_loader.e1candidates[rel]), len(data_loader.tasks[rel][self.few:])))
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0
            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == 'EOT':
                    break
                t += 1

                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
                x = torch.cat([n_score, p_score], 1).squeeze()

                self.rank_predict(data, x, ranks)

                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
                sys.stdout.flush()

            print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

        return all_data




    # def eval_for_ranking(self, istest=True, epoch=None):
    #     self.metaR.eval()
    #     self.metaR.rel_q_sharing = dict()
    #     if istest:
    #         data_loader = self.test_data_loader
    #         eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
    #         _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
    #         x = torch.cat([n_score, p_score], 1).squeeze()
    #         true_tail = eval_task[0][0][2]      # 正样本尾实体
    #         neg_tails = [trip[2] for trip in negative_triples[0]]
    #         ranked = sorted(zip(all_tails, all_scores), key=lambda x: x[1], reverse=True)

    # def eval_for_ranking(self, istest=True, epoch=None):
    #     self.metaR.eval()
    #     self.metaR.rel_q_sharing = dict()
    #     # data_loader = self.test_data_loader
    #
    #     # Step 1: 获取 support set 与 head 实体
    #     (support_triples, support_negatives), head = self.test_data_loader.next_support_only()
    #
    #     # 解析 support 信息
    #     support_triples = support_triples[0]  # [ [h, r, t], ... ]
    #     support_negatives = support_negatives[0]  # [ [h, r, neg_t], ... ]
    #     rel = support_triples[0][0]
    #
    #     # Step 2: 获取候选尾实体（药物）
    #     candidates = self.test_data_loader.e1candidates[head]
    #
    #     # Step 3: 构建查询三元组列表 (head, rel, tail_candidate)
    #     query_triples = [[head, rel, tail] for tail in candidates]
    #
    #     # Step 4: 构造模型输入，调用 do_one_step 计算得分
    #     with torch.no_grad():
    #         # _, p_score, n_score = self.do_one_step((support_triples, support_negatives), iseval=True, curr_rel=rel)
    #         p_score, n_score = self.metaR((support_triples, support_negatives), iseval=True, curr_rel=rel,
    #                                       support_weight_indices=None)
    #     scores = torch.cat([n_score, p_score], 1).squeeze()
    #
    #     # Step 5: 将候选药物按得分排序输出
    #     scores = scores.detach().cpu().numpy()
    #     ranking = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    #
    #     print(f"\n🔍 疾病: {head}，关系: {rel}，候选药物数: {len(candidates)}")
    #     print("💊 Top 10 药物排名:")
    #     for i, (tail, score) in enumerate(ranking[:10], 1):
    #         print(f"{i:2d}. 药物ID: {tail:>6}，得分: {score:.4f}")
    #     return ranking

    # def eval_for_ranking(self, disease_id, relation, support_triples, candidates):
    # def eval_for_ranking(self, disease_id="83914", relation = "indication", support_triples = [["83914", "indication", "14246"],["83914", "indication", "14245"]], candidates =None):
    #     cand_file = "Y:/primkg-assistant(disease)/e1candidates-processed-disease-all-other-200.json"
    #
    #     # 加载 JSON 文件
    #     with open(cand_file, "r", encoding="utf-8") as f:
    #         e1candidates = json.load(f)
    #
    #     # 获取 head entity 为 "83914" 的所有候选尾实体（药物）
    #     candidates = e1candidates["83914"]
    #     e1rel_e2_path = "Y:/primkg-assistant(disease)/e1rel_e2-1.json"
    #     with open(e1rel_e2_path, "r", encoding="utf-8") as f:
    #         e1rel_e2 = json.load(f)
    #     self.metaR.eval()
    #     self.metaR.rel_q_sharing = dict()
    #
    #     # 构造 support set
    #     support_triples_tensor = [[triple for triple in support_triples]]
    #     support_negative_triples_tensor = [[]]
    #
    #     # 构造负样本：对每个 support triple 随便取一个候选药物作为负例
    #     shift = 0
    #     for triple in support_triples:
    #         e1, rel, e2 = triple
    #         while True:
    #             neg_tail = candidates[shift]
    #             if (neg_tail not in e1rel_e2[e1 + rel]) and (neg_tail != e2):
    #                 support_negative_triples_tensor[0].append([e1, rel, neg_tail])
    #                 break
    #             shift += 1
    #
    #     # 构造 query_triples（即：所有候选尾实体的三元组）
    #     query_triples = [[[disease_id, relation, tail] for tail in candidates]]
    #     negative_triples = [[]]  # 不需要负样本
    #
    #     # 打包成 MetaR 的输入格式
    #     eval_task = [support_triples_tensor, support_negative_triples_tensor, query_triples, negative_triples]
    #
    #     # 前向传播得分
    #     _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=relation)
    #     # scores = scores.squeeze().detach().cpu().numpy()
    #     scores = torch.cat([n_score, p_score], 1).squeeze()
    #     # 得分排序
    #     ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    #
    #     print("\n🎯 疾病实体：{} 的候选药物预测得分排名：\n".format(disease_id))
    #     print(f"{'Rank':<5}{'Drug (Tail ID)':<15}{'Score':<10}")
    #     for i, (tail, score) in enumerate(ranked_results[:30]):  # 前30个
    #         print(f"{i+1:<5}{tail:<15}{score:<10.4f}")
    #     return ranked_results
    import numpy as np
    from collections import defaultdict

    # def eval_single_disease_ranking(self, disease_id: str, relation: str = "indication"):
    #     dataset = dict()
    #     self.metaR.eval()
    #     self.metaR.rel_q_sharing = dict()
    #
    #     # 加载 candidate tails
    #     with open("Y:/primkg-assistant(disease)/e1candidates-processed-disease-all-other-200.json", "r",
    #               encoding="utf-8") as f:
    #         e1candidates = json.load(f)
    #     candidate_tails = e1candidates[disease_id]
    #     # ==== 加载必要数据 ====
    #     with open('Y:/primkg-assistant(disease)/ent2ids.json') as f:
    #         entity2id = json.load(f)
    #
    #     with open('Y:/primkg-assistant(disease)/rel2ids.json') as f:
    #         relation2id = json.load(f)
    #
    #     # 加载任务三元组
    #     with open("Y:/primkg-assistant(disease)/alcohol withdrawal delirium_indication_test_tasks.json", "r", encoding="utf-8") as f:
    #         test_tasks = json.load(f)
    #     assert disease_id in test_tasks, f"No query triples found for {disease_id}"
    #
    #     query_triples = test_tasks[disease_id]
    #     assert len(query_triples) > 0, "No queries to evaluate"
    #
    #     support = torch.tensor([[[
    #         entity2id[h], relation2id[r], entity2id[t]
    #     ] for h, r, t in query_triples]], dtype=torch.long).to(self.device)
    #
    #     support_negative = torch.tensor([[[0, 0, 0]]], dtype=torch.long).to(self.device)
    #
    #     score_dict = defaultdict(list)
    #     # print("loading ent2id ... ...")
    #     # dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    #     # print("loading rel2id ... ...")
    #     # dataset['rel2id'] = json.load(open(data_dir['rel2ids']))
    #     for h, r, _ in query_triples:
    #         query = torch.tensor([[[
    #             entity2id[h], relation2id[r], entity2id[tail]
    #         ] for tail in candidate_tails]], dtype=torch.long).to(self.device)
    #
    #         dummy_neg = torch.tensor([[[0, 0, 0]]], dtype=torch.long).to(self.device)
    #         task = (support, support_negative, query, dummy_neg)
    #
    #         with torch.no_grad():
    #             _, p_scores, n_scores, _ = self.metaR(task, iseval=True, curr_rel=r)
    #         scores = torch.cat([n_scores, p_scores], 1)
    #         scores = scores.squeeze(0).cpu().numpy()
    #         for tail, score in zip(candidate_tails, scores):
    #             score_dict[tail].append(score)
    #
    #     avg_scores = [(tail, np.mean(score_list)) for tail, score_list in score_dict.items()]
    #     avg_scores.sort(key=lambda x: x[1], reverse=True)
    #
    #     print(f"\n=== Ranking for Disease {disease_id} ({relation}) ===")
    #     print(f"{'Rank':<6}{'Tail Entity (Drug)':<20}{'Avg Score':<10}")
    #     for idx, (tail, avg_score) in enumerate(avg_scores, 1):
    #         print(f"{idx:<6}{tail:<20}{avg_score:<10.4f}")
    #
    #     return avg_scores

    def eval_single_disease_ranking(self, disease_id: str, relation: str = "indication"):
        from collections import defaultdict
        import json
        import numpy as np

        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        # 加载 candidate tails
        with open("/home/ubuntu/YL/primkg-assistant(disease)/28714_candidate.json", "r",
                  encoding="utf-8") as f:
            e1candidates = json.load(f)
        candidate_tails = e1candidates[disease_id]

        # 加载测试三元组
        with open("/home/ubuntu/YL/primkg-assistant(disease)/test_cystic fibrosis_indication_tasks.json", "r",
                  encoding="utf-8") as f:
            test_tasks = json.load(f)
        assert disease_id in test_tasks, f"No query triples found for {disease_id}"
        query_triples = test_tasks[disease_id]
        assert len(query_triples) > 0, "No queries to evaluate"

        # 用字符串三元组直接构造支持集
        support = [query_triples]  # shape: [1, num_support, 3]
        # support_negative = [[[]]]  # dummy，不参与训练
        # dummy_negative = [[[]]]  # query负样本
        support_negative = [[["0","indication", "0"]]]
        dummy_negative = [[["0","indication", "0"]]]
        # 统计每个候选尾实体的得分
        score_dict = defaultdict(list)

        for triple in query_triples:
            h, r, _ = triple
            query = [[
                [h, r, tail] for tail in candidate_tails
            ]]  # shape: [1, num_candidates, 3]

            task = (support, support_negative, query, dummy_negative)

            # with torch.no_grad():
            p_scores, n_scores = self.metaR(task, iseval=True, curr_rel=h, support_weight_indices=None)
            scores = torch.cat([n_scores, p_scores], 1)
            scores = scores.squeeze(0).detach().cpu().numpy()  # shape: [num_candidates]
            for tail, score in zip(candidate_tails, scores):
                score_dict[tail].append(score)

        # 平均打分
        avg_scores = [(tail, np.mean(score_list)) for tail, score_list in score_dict.items()]
        avg_scores.sort(key=lambda x: x[1], reverse=True)

        # 打印排名
        print(f"\n=== Ranking for Disease {disease_id} ({relation}) ===")
        print(f"{'Rank':<6}{'Tail Entity (Drug)':<20}{'Avg Score':<10}")
        for idx, (tail, avg_score) in enumerate(avg_scores, 1):
            print(f"{idx:<6}{tail:<20}{avg_score:<10.4f}")

        csv_file = f"{disease_id.replace(' ', '_')}_{relation}_ranking.csv"
        with open(csv_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Tail Entity (Drug)", "Avg Score"])
            for idx, (tail, avg_score) in enumerate(avg_scores, 1):
                writer.writerow([idx, tail, round(avg_score, 4)])

        print(f"\nRanking result exported to: {csv_file}")
        return avg_scores
