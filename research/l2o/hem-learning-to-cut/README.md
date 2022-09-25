# Contents

[TOC]

## HEM overview

Hierarchical sequence model was proposed in 2022 to replace the cut-plane selection module of mixed-integer programming
solver with reinforcement learning and hierarchical sequence model learning strategies, so as to improve the efficiency
and solution quality of the solver.

## Model architecture

The hierarchical sequence model is divided into two layers. The upper layer model learns to select the number of cut
planes, and the lower layer model learns to select an ordered subset of the corresponding size according to the given
number of the upper layer model. The objective function of network training is obtained from the hierarchical policy
gradient we derive.

## Dataset

The experiment is based on Huawei's real production planning problem. The data set will be open source in the near
future after desensitization. Please look forward to it.

## Environmental requirements

- Hardware: indicates a GPU and CPU equipped machine

- Deep learning framework: MindSpore

- Python rely on

- Python 3.7

- tqdm

- gtimer

- Solver dependencies

- SCIP 8.0.0

## Quick Start

After the environment and dataset are ready, execute the following code to begin training and periodically test the
training model performance

Python C usce_algorithm. py --config_file configs/cbg_66_v3_config_dual_bound_minspore. Json --sel_cuts_percent 0.2
--reward_type solving_time

## Script description

- configs/*. Json # Indicates the parameter configuration file

- c usce_algorithm. py # Start a serial training experimental inlet

- pointer_net.py # network structure model

-Environments. Py # packages the solver as a reinforcement learning environment

## Model description

### Performance

On the above Huawei real production planning dataset, the given solution time limit is 600s, the default SCIP solver
solution time is 296.12 seconds, and the solution time is reduced to 241.77 seconds after embedding our learning model.

## Random fact Sheet

There are two sources of code randomness. One is the randomness of the algorithm inside the solver, which can be fixed
by setting the scip_seed parameter. The second is the random module in Python and the random module in MindSpore, which
can be uniformly set by setting the seed parameter.

## ModelZoo home page

Please browse the website home page (<https://gitee.com/mindspore/models>).
