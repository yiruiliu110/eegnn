# Edge Enhanced Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)



## Introduction

Training deep graph neural networks (GNNs) poses a challenging task, as the performance of GNNs may suffer from the number of hidden message-passing layers. The literature has focused on the proposals of over-smoothing and under-reaching to explain the performance deterioration of deep GNNs. In this paper, we propose a new explanation for such deteriorated performance phenomenon, mis-simplification, that is, mistakenly simplifying graphs by preventing self-loops and forcing edges to be unweighted. We show that such simplifying can reduce the potential of message-passing layers to capture the structural information of graphs. In view of this, we propose a new framework, edge enhanced graph neural network (EEGNN). EEGNN uses the structural information extracted from the proposed Dirichlet mixture Poisson graph model, a Bayesian nonparametric model for graphs, to improve the performance of various deep message-passing GNNs. Experiments over different datasets show that our method achieves considerable performance increase compared to baselines
## Requirements
#### Installation with Conda
```bash
conda create -n deep_gcn_benchmark
conda activate deep_gcn_benchmark
pip install -r requirement.txt
```

## Train Conventional Deep GNN models 

To train a deep SGC model `<model>` on dataset `<dataset>` as your baseline, run:

```bash
python main.py --compare_model=1 --cuda_num=0 --SGC --dataset=TEXAS --num_layers=64
# <model>   in  [APPNP,  GCNII,  SGC]
# <dataset> in  [Cora, Citeseer, Pubmed, TEXAS, WISCONSIN, CORNELL]
```

#### Train Edge Enhanced Deep GNN models

To train a deep SGC model `<model>` on dataset `<dataset>` as your baseline, run:

```bash
python main.py --compare_model=1 --cuda_num=0 --SGC_new --dataset=TEXAS --num_layers=64
# <model>   in  [APPNP_new,  GCNII_new,  SGC_new]
# <dataset> in  [Cora, Citeseer, Pubmed, TEXAS, WISCONSIN, CORNELL]
```