# CFlowNets

## Paper

CFlowNets: Continuous Control with Generative Flow Networks, ICLR 2023

## Citing CFlowNets

@inproceedings{li2023cflownet,
  title={CFlowNets: Continuous Control with Generative Flow Networks},
  author={Yinchuan Li and Shuang Luo and Haozhi Wang and Jianye Hao},
  booktitle{ICLR},
  year={2023}
}

## Introduction

Generative flow networks (GFlowNets), as an emerging technique, can be used as an alternative to reinforcement learning for exploratory control tasks. GFlowNets aims to sample actions with a probability proportional to the reward, similar to sampling different candidates in an active learning fashion. However, existing GFlowNets cannot adapt to continuous control tasks because GFlowNets need to form a DAG and compute the flow matching loss by traversing the inflows and outflows of each node in the trajectory. In this paper, we propose generative continuous flow networks (CFlowNets) that can be applied to continuous control tasks. First, we present the theoretical formulation of CFlowNets. Then, a training framework for CFlowNets is proposed, including the action selection process, the flow approximation algorithm, and the continuous flow matching loss function. Afterward, we theoretically prove the error bound of the flow approximation. The error decreases rapidly as the number of flow samples increases. Finally, experimental results on continuous control tasks demonstrate the performance advantages of CFlowNets compared to many reinforcement learning methods, especially regarding exploration ability.

## Framework

 ![图片说明](http://image.huawei.com/tiny-lts/v1/images/ef863fab51087a22e6b21e01fe3a23ba_1366x557.png)![图片说明](http://image.huawei.com/tiny-lts/v1/images/ef863fab51087a22e6b21e01fe3a23ba_1366x557.png)

## main function

cfn_grid_final.py

## Environment

languaue：python 3.7.0
framework：MindSpore 1.9.0

## Directory

```test
.
└─CFlowNets
  |
  ├─cfn_grid_final.py      # main
  ├─config.py
  ├─lossnetwork.py         # mindspore loss function
  ├─point_env.py           # rl environment
  |
  ├─README.md
  ├─requirements.txt
  |
  ├─repaly_buffer.py
  ├─transaction.py         # network model
  └─utils.py
```
