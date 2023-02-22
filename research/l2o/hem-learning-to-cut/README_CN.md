# 目录

[TOC]

## HEM 概述

分层序列模型于2022 年提出，旨在利用强化学习和分层序列模型学习策略替换混合整数规划求解器中的割平面选择模块，提升求解器效率和求解质量。

## 模型架构

分层序列模型分为两层模型，上层模型学习选择割平面的数量，下层模型根据上层模型给定的数量，学习选择一个相应大小的有序子集。网络训练的目标函数由我们推导的分层策略梯度得到。

## 数据集

实验基于华为真实的排产规划问题，该数据集会在不久的将来经过脱敏处理后开源，敬请期待。

## 环境要求

* 硬件: 具备GPU 和CPU 的机器
* 深度学习框架: MindSpore
* Python 依赖: Python 3.7, tqdm, gtimer
* 求解器依赖: SCIP 8.0.0

## 快速入门

在环境和数据集准备完成后，执行以下代码则可以开始训练，并定期测试训练模型性能
python reinforce_algorithm.py --config_file configs/cbg_66_v3_config_dual_bound_minspore.json --sel_cuts_percent 0.2
--reward_type solving_time

## 脚本说明

* configs/*.json # 参数配置文件
* reinforce_algorithm.py # 启动串行训练实验入口
* pointer_net.py # 网络结构模型
* environments.py # 将求解器封装为强化学习环境

## 模型描述

### 性能

在上述华为真实排产规划数据集上，给定求解时间限制为600s，默认SCIP 求解器求解时间为296.12秒，嵌入我们的学习模型后求解时间降低为241.77秒。

## 随机情况说明

代码随机性来源有两个，其一是求解器内部算法的随机性，可由设置scip_seed 参数固定；其二是python 中的随机模块和MindSpore
中的随机模块导致，可由设置seed 参数统一设定。

## ModelZoo 主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
