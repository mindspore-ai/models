# 目录

- [目录](#目录)
- [DDPG描述](#DDPG描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
- [模型描述](#模型描述)
    - [训练性能](#训练性能)
    - [推理流程](#推理流程)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

# DDPG 描述

DDPG(deep deterministic policy gradient),深度确定性策略梯度算法。

DDPG 本质上是AC框架的一种强化学习算法，结合了基于policy的policy gradient和基于action value的DQN，可以通过off—policy的方法，单步更新policy，预测出确定性的策略，进而实现total reward最大化

在DDPG出现之前，在强化学习领域遇到连续动作的问题，一般会将连续动作离散化，再进行强化学习。DDPG的出现，使得**连续动作**的直接预测问题成为可能。

<font color='cornflowerblue'>论文：</font> Timothy P. Lillicrap,Jonathan J. Hunt,Alexander Pritzel,Nicolas Heess,Tom Erez,Yuval Tassa,David Silver,Daan Wierstra. Continuous control with deep reinforcement learning.[J]. CoRR,2015,abs/1509.02971:

# 模型架构

DDPG 拥有四个网络，其中有两个Actor网络和两个Critic网络，分别为：actor网络、critic网络，actor_target网络和critic_target网络。其中两个Actor网络结构相同，两个Critic网络结构相同。

# 数据集

强化学习本质上不是一种监督学习，不需要传统意义上的数据集，只需要借助仿真物理环境，所以数据部分皆为gym库下的环境。本项目使用'Pendulum-v0'环境作为数据来源。

# 环境要求

- 硬件

    - Ascend910

- 框架

    - [MindSpore](https://www.mindspore.cn)

- 如需查看详情，请参见如下资源：

    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

- 第三方库

    - [Gym](https://gym.openai.com/)

        - 安装gym

            ```python
            pip install gym
            ```

        - 查看gym可用环境

            ```python
            from gym import envs
            env_spaces = envs.registry.all()
            env_ids = [env_space.id for env_space in env_spaces]
            print(env_ids)
            ```

        - 除了以上环境外，还可选择基于Gym的付费环境Mujoco，或采用自定义环境。详细操作参考Gym官网

# 快速入门

通过官方网站安装MindSpore后，您可按照以下步骤进行训练和推理

```python
python train.py
# 待train执行完成后，生成checkpoint文件后可执行推理脚本：
python verify.py
```

# 脚本说明

- 脚本及样例代码

```python
ddpg
├── README_CN.md               # 说明文档
├── default_paras.yaml      # 超参文件
├── ascend_310_infer
│   ├── inc
│   │   ├── utils.h
│   ├── src
│   │   ├── utils.cc
│   │   ├── main.cc         # Ascend 310 cpp模型文件
│   ├── build.sh            # 编译脚本
│   ├── CMakeLists.txt      # C++过程文件
├── output                  # C++过程文件
├── scripts
│   ├── run_910_train.sh    # 训练shell脚本
│   ├── run_910_verify.sh   # ddpg Ascend 310推理shell脚本
│   ├── run_infer_310 .sh   # ddpg Ascend 310 推理shell脚本
├── src
│   ├── agent.py            # ddpg 模型脚本
│   ├── ac_net.py           # ddpg 基础网络
│   ├── loss_net.py         # ddpg 损失网络
│   ├── config.py           # ddpg 加载超参脚本
├── train.py                # ddpg 训练脚本
├── verify.py               # ddpg Ascend 910推理脚本
├── verify_310.py           # ddpg Ascend 310推理脚本
├── export.py               # ddpg 模型导出脚本
```

- 脚本参数

```python
GAMMA = 0.99             #  Q衰减系数
TAU = 0.001              #  软更新权重
BATCH_SIZE = 32          #  训练样本大小
MEMORY_CAPICITY = 10000  #  经验池大小
LR_ACTOR = 1e-3          #  Actor网络学习率
LR_CRITIC = 1e-4         #  Critic网络学习率
CRITIC_DECAY = 0.01      #  Critic网络L2正则化系数
EPISODES = 5000          #  训练轮数
EP_STEPS = 200           #  训练单轮步数
EP_TEST = 5              #  测试轮数
STEP_TEST = 100          #  测试单轮步数
```

- 训练过程

    ```python
    python:
        python train.py
    shell:
    cd scripts
    bash run_910_train.sh [device_id]
    #示例
    bash run_910_train.sh 3
    ```

- 推理过程

    ```python
    python：
        python verify.py
    shell:
    cd scripts
    bash run_910_verify.sh [device_id]
    #示例
    bash run_910_verify.sh 3
    ```

- 导出过程

    ```python
    python export.py
    ```

# 模型描述

## 训练性能

| 参数           | DDPG                                                         |
| -------------- | ------------------------------------------------------------ |
| 资源           | Ascend 910; CPU 2.60GHz,192cores;Memory,755G                 |
| 上传日期       | 2021.12.11                                                   |
| MindSpore 版本 | 1.5.0                                                        |
| 训练参数       | Gamma:0.99, EPISODES:5000, EP_STEPS:200, MEMORY_CAPACITY:10000,   BATCH_SIZE:32, EP_TEST:5, STEP_TEST:100, TAU:0.001, LR_ACTOR:1e-3, LR_CRITIC: 1e-4, CRITIC_DECAY: 0.01 |
| 优化器         | Adam                                                         |
| 损失函数       | MSEloss                                                      |
| 输出           | reward                                                       |
| 速度           | 0.06s/step                                                   |
| 参数(M)        | 0.13                                                         |
| 脚本           | train.py                                                     |

## 推理流程

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html)。下面是Ascend 910操作步骤示例：

- Ascend 910处理器环境操作说明：

    ```python
    # 设置上下文，单卡环境无需设置
    context.set_context(device_id=0)

    #定义推理环境
    env = gym.make('Pendulum-v0')
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    #定义模型
    ddpg_agent = agent.Agent(a_dim, s_dim)

    # 加载预训练模型参数
    load_checkpoint("actor_net.ckpt",net=verify_agent.actor_net)
    load_checkpoint("actor_target.ckpt",net=verify_agent.actor_target)
    load_checkpoint("critic_net.ckpt",net=verify_agent.critic_net)
    load_checkpoint("critic_target.ckpt",net=verify_agent.critic_target)

    #最后仅需调用gym环境计算reward即可
    ```

- Ascend 910运行推理脚本

    ```python
    # 执行python文件
    python verify.py
    # 执行shell文件
    cd scripts
    bash run_910_verify.sh
    ```

- GPU、CPU下运行同上

- Ascend 310 运行推理脚本

    ```python
    # Ascend310推理需导出Ascend910下训练结束后的模型文件和模型参数文件
    python export.py
    ```

    ```shell
    # 先编译C++文件，再执行Python推理
    bash run_infer_310.sh [MINDIR_PATH] [OUTPUT_DIR] [DEVICE_ID]
    # example
    cd scripts
    bash run_infer_310.sh ../test.mindir ../output 0
    ```

# 随机情况说明

在推理过程和训练过程中，我们都使用到gym环境下的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
