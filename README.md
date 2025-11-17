# MetaGH
This repository contains an implementation of MetaGH model based on the paper "MetaGH: Gradient-Guided Meta-Learning on Diseases for Few-Shot Drug Repurposing" by Lei Yang, Pingjian Ding, Lingyun Luo.

#Overview

We formulate drug repurposing as a few-shot link-prediction task on a large-scale biomedical knowledge graph that integrates heterogeneous data sources, and propose MetaGH, a novel meta-learning framework that updates head disease entity embeddings using gradient-based meta-information. The framework generates head representations via a learnable transformation of concatenated relation and tail entity embeddings, computes support set gradients to refine these representations, and uses the adapted embeddings for query set evaluation, enabling rapid generalization to novel diseases from minimal labeled examples. 


# Dataset
We use the PrimeKG dataset to evaluate the MetaGH model.
PrimeKG was originally proposed and constructed by Payal Chandak, Kexin Huang, and Marinka Zitnik from Harvard University.
The official dataset and documentation can be accessed from the following repository:
👉 https://github.com/mims-harvard/PrimeKG

To ensure experimental reproducibility, we directly utilize the original PrimeKG data provided by the authors.
Based on this dataset, we further perform custom data preprocessing and splitting to suit our meta-learning framework.
These processed subsets are derived from PrimeKG and follow the same entity and relation schema.

The processed datasets used in our experiments are provided as follows:
train_tasks.json

valid_tasks.json

test_tasks.json

ent2ids.json

rel2ids.json

e1candidate.json

All these files are generated based on PrimeKG and can be used directly to reproduce the results reported in our paper.


# Requirements
python                  3.9.18
torch                     2.0.1
tensorboardX              2.5.1


# Usage

To reproduce the few-shot drug repurposing experiments presented in this work, please follow the steps below.

1️⃣ Generate the Few-Shot Task Dataset

Before training the model, you need to construct the task-based datasets (support/query sets) used in the MetaGH framework.
Run the following command to generate all necessary files:
```bash
python data_process.py
```

This script automatically processes the original PrimeKG dataset and produces the few-shot task splits used in our experiments

2️⃣ Train and Evaluate the Model

After generating the datasets, execute the following command to start model training and evaluation:
```bash
python main.py
```

This script runs the MetaGH training pipeline, including:

Meta-learning based model optimization

Validation during training with early stopping

Final testing and metric computation (MRR, Hits@K, AUC, AU-PR)

Automatic logging of performance results

3️⃣ Output and Results

After successful execution, the following outputs will be available:

Model checkpoints under the checkpoints/ directory

Evaluation logs and metrics in the logs/ directory

# result

On disease-centric prediction tasks, MetaGH substantially outperforms state-of-the-art baselines across all few-shot settings: in 1-shot scenarios averaged over 10 independent runs, MetaGH achieves improvements of 0.051 in MRR, 0.102 in Hits@10, and 0.010 in AUC compared to the best baseline, while in 5-shot scenarios, the model demonstrates improvements of 0.036 in MRR and 0.030 in Hits@5.

