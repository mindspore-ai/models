# GFlowCausal

## Paper

GFlowCausal: Generative Flow Networks for Causal Discovery

## Introduction

Causal discovery aims to uncover causal structure among a set of variables. Score-based is one primary causal discovery class, which focuses on searching for the best Directed Acyclic Graph (DAG) based on a predefined score function. However, most of them are not applicable on a large scale due to the limited searchability. Inspired by the active learning in generative flow networks, we propose GFlowCausal to convert the graph search problem to a generation problem, in which direct edges are added gradually. GFlowCausal aims to learn the best policy to generate high-reward DAGs by sequential actions with probabilities proportional to predefined rewards. We propose a plug-and-play module based on transitive closure to ensure efficiently sampling. Theoretical analysis shows that this module could guarantee acyclicity properties effectively and the consistency between final states and fully-connected graphs. We conduct extensive experiments on both synthetic and real datasets, and results show the proposed approach to be superior and also performs well in a large-scale setting.

## Framework

![](http://image.huawei.com/tiny-lts/v1/images/89a9a4466d4e91f9ed683be59fa83673_1019x583.png)
Our code will be uploaded here soon after the company review.

## main function

main_ms.py

## Datasets

ER、SF

## Environment

language：python 3.7.0

framework：MindSpore 1.9.0

## Directory

```test
.
└─GFlowCausal
  |
  ├─castle
  | ├─metrics
  |   ├─evaluation.py
  | ├─datasets
  |   ├─simulator.py
  |
  ├─network
  | ├─model_ms.py          # ms network model
  |
  ├─README.md
  ├─requirements.txt
  ├─env.py                 # environment
  ├─lossnetwork.py         # mindspore loss function
  ├─args.py
  ├─main_ms.py
  └─utils.py
```
