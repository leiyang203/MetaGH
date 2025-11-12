import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default=r"primkg-assistant(disease)", type=str)
    args.add_argument("-path", "--data_path", default=r"data/primkg-assistant(disease)", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-form", "--data_form", default="Discard", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=5, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    # , "AUC", "AU-PR"
    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.0010, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)

    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=True, type=bool)
    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="state/exp", type=str)
    #cystic-fibrosis_indication
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default=r"log", type=str)
    args.add_argument("-state_dir", "--state_dir", default=r"state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'primkg-assistant(disease)':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    params['device'] = torch.device('cuda:'+str(args.device))

    return params


data_dir = {
    # 'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/train_tasks4.json',
    # 'train_tasks': '/train_cystic fibrosis_tasks.json',
    'test_tasks': '/test_tasks4.json',
    # 'test_tasks': '/test_cystic fibrosis_indication_tasks.json',
    'dev_tasks': '/dev_tasks4.json',

    # 'rel2candidates_in_train': '/rel2candidates_in_train.json',
    # 'rel2candidates': '/rel2candidates.json',
    # 'e1candidates': '/e1candidates-processed.json',
    'e1candidates': '/e1candidates-processed-disease-all-other-200.json',
    # 'e1candidates': '/e1candidates.json',
    # 'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2-4.json',
    # 'e1rel_e2': '/e1rel_e2-rare-pro.json',
    'ent2ids': '/ent2ids-4.json',
    # 'ent2ids': '/entity2id.txt',
    'rel2ids': '/rel2ids.json',
    'ent2vec': '/ent2vec.npy',
}
