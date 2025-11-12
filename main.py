from trainer import *
from params import *

from data_loader import *
import json
CONDA_NO_PLUGINS=True


if __name__ == '__main__':
    params = get_params()


    print("---------Parameters---------")

    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = ''
    if params['data_form'] == 'In-Train':

         tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks'+tail]))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))

    print("loading e1candidates{} ... ...".format(tail))
    # dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates'+tail]))
    print(data_dir['e1candidates'])
    dataset['e1candidates'] = json.load(open(data_dir['e1candidates']))

    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2']))
    # dataset['e1_rele2'] = json.load(open(data_dir['e1_rele2'+tail]))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
    print("loading rel2id ... ...")
    dataset['rel2id'] = json.load(open(data_dir['rel2ids']))
    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        # dataset['ent2emb'] = np.load(data_dir['ent2vec'])
        dataset['ent2emb'] = np.loadtxt(params['data_path'] + '/entity2vec.TransE')
        dataset['rel2emb'] = np.loadtxt(params['data_path'] + '/relation2vec.TransE')

    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]
    # data_loaders = [train_data_loader, test_data_loader]


    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
        # trainer.eval_single_disease_ranking(disease_id='28714')
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.reload()
            trainer.eval(istest=True)
            # trainer.eval_single_disease_ranking(disease_id='28714')
            # trainer.eval_for_ranking()
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            # rankings = trainer.eval(istest=True)
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)



# from trainer import *
# from params import *
# from data_loader import *
# import json
# import copy
#
# if __name__ == '__main__':
#     base_params = get_params()
#     # ⚠️ 保存原始 data_dir
#     base_data_dir = copy.deepcopy(data_dir)
#
#     # 控制随机种子
#     if base_params['seed'] is not None:
#         SEED = base_params['seed']
#         torch.manual_seed(SEED)
#         torch.cuda.manual_seed(SEED)
#         torch.backends.cudnn.deterministic = True
#         np.random.seed(SEED)
#         random.seed(SEED)
#
#
#
#     # ===== 重复运行 4 次 =====
#     for run in range(1, 5):
#         print(f"\n================ Run {run} ================\n")
#
#         # 拷贝一份参数，避免覆盖
#         params = copy.deepcopy(base_params)
#         params['prefix'] = f"{base_params['prefix']}_run{run}"
#
#         # # 选择数据集
#         # for k, v in data_dir.items():
#         #     data_dir[k] = params['data_path']+v
#
#         # 每次运行前重新拼接路径
#         data_dir = copy.deepcopy(base_data_dir)
#         for k, v in data_dir.items():
#             # if not os.path.isabs(v):
#             data_dir[k] = os.path.join(params['data_path'], v)
#         # Debug 打印，方便检查路径是否正常
#         # print("DEBUG data_dir:")
#         # for k, v in data_dir.items():
#         #     print(f"  {k}: {v}")
#         # tail = ''
#         if params['data_form'] == 'In-Train':
#             tail = '_in_train'
#
#         dataset = dict()
#         # print("loading train_tasks{} ... ...".format(tail))
#         dataset['train_tasks'] = json.load(open(data_dir['train_tasks']))
#         print("loading test_tasks ... ...")
#         dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
#         print("loading dev_tasks ... ...")
#         dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
#
#         # print("loading e1candidates{} ... ...".format(tail))
#         dataset['e1candidates'] = json.load(open(data_dir['e1candidates']))
#
#         # print("loading e1rel_e2{} ... ...".format(tail))
#         dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2']))
#
#         print("loading ent2id ... ...")
#         dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
#         print("loading rel2id ... ...")
#         dataset['rel2id'] = json.load(open(data_dir['rel2ids']))
#         if params['data_form'] == 'Pre-Train':
#             print('loading embedding ... ...')
#             dataset['ent2emb'] = np.loadtxt(params['data_path'] + '/entity2vec.TransE')
#             dataset['rel2emb'] = np.loadtxt(params['data_path'] + '/relation2vec.TransE')
#
#         print("----------------------------")
#
#         # data_loader
#         train_data_loader = DataLoader(dataset, params, step='train')
#         dev_data_loader = DataLoader(dataset, params, step='dev')
#         test_data_loader = DataLoader(dataset, params, step='test')
#         data_loaders = [train_data_loader, dev_data_loader, test_data_loader]
#
#         trainer = Trainer(data_loaders, dataset, params)
#
#         if params['step'] == 'train':
#             trainer.train()
#             print("test")
#             print(params['prefix'])
#             trainer.reload()
#             trainer.eval(istest=True)
#
#             # trainer.eval_single_disease_ranking(disease_id='28714')
#
#         elif params['step'] == 'test':
#             print(params['prefix'])
#             if params['eval_by_rel']:
#                 trainer.eval_by_relation(istest=True)
#             else:
#                 trainer.reload()
#                 trainer.eval(istest=True)
#                 # trainer.eval_single_disease_ranking(disease_id='28714')
#
#         elif params['step'] == 'dev':
#             print(params['prefix'])
#             if params['eval_by_rel']:
#                 trainer.eval_by_relation(istest=False)
#             else:
#                 trainer.eval(istest=False)
