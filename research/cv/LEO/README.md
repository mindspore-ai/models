# 目录

<!-- TOC -->

- [LEO描述](#leo描述)
    - [概述](#概述)
    - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估精度](#评估精度)
    - [使用流程](#使用流程)
        - [推理](#推理)
        - [继续训练预训练模型](#继续训练预训练模型)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [Resource说明](#Resource说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# LEO描述

## 概述

LEO(Meta-Learning with Latent Embedding Optimization)由Rusu et.Al等人提出，是一种基于参数优化的元学习模型，该文章于2018年7月发布在arXiv上，并在ICLR 2019上展示。LEO基于Chelsea Finn等人提出的MAML（Model-agnostic meta-learning）算法建立，在小样本学习的情况下，MAML在高维参数空间中的计算梯度使得模型泛化变得困难，为解决此问题，LEO最重要的改进是引入了一个低维的隐空间（Latent Space），在隐层表示上执行基于梯度的元学习来对模型进行优化更新。

## 论文

[论文1]
[Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pp. 1126–1135, 2017.](https://arxiv.org/pdf/1703.03400v3.pdf)

[论文2]
[Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. In International Conference on Learning Representations,2019. 1, 6, 7](https://github.com/deepmind/leo)

[参考工程代码]
[https://github.com/timchen0618/pytorch-leo](https://github.com/timchen0618/pytorch-leo)

# 模型架构

LEO由以下几个模块组成，分类器，编码器，关系网络和编码器，各模块均采用三层MLP。如下为5-way 1-shot 的架构细节。

| Part of the model | Architecture           | Hidden layer size  | Shape of the output |
| ----------------- | ---------------------- | ------------------ | ------------------- |
| Inference model   | 3-layer MLP with ReLU  | 40                 | (12,5,1)            |
| Encoder           | 3-layer MLP with ReLU  | 16                 | (12,5,16)           |
| Relation network  | 3-layer MLP with ReLU  | 32                 | (12,2×16)           |
| Decoder           | 3-layer MLP with ReLU  | 32                 | (12,2×1761)         |

# 数据集

## 使用的数据集

- miniImageNet
    - 数据集大小：大小一共2.86GB，包含100个类，每类有600个样本，每张图片的规格为 84 × 84
    - 训练集：包含64个类
    - 验证集：包含16个类
    - 测试集：包含20个类
- 数据格式：pkl

- tieredImageNet
    - 数据集大小：大小一共12.9GB，包含34个大类，608个小类，共779165个样本，每张图片的规格为 84 × 84
    - 训练集：包含20个大类，351个小类
    - 验证集：包含6个大类，97个小类
    - 测试集：包含8个类，160小类
- 数据格式：pkl

## 数据预训练

训练一个28层的wide residual network（WRN-28-10），将图片数据进行预训练并保存为640维特征向量的形式，可通过[http://storage.googleapis.com/leo-embeddings/embeddings.zip](http://storage.googleapis.com/leo-embeddings/embeddings.zip) 直接下载embeddings文件

## 数据集架构

```bash
├─ datasets/
   ├─ miniImageNet/
   │  ├─ center/
   │  │  ├─ test_embeddings.pkl
   │  │  ├─ train_embeddings.pkl
   │  │  └─ val_embeddings.pkl
   │  └─ multiview/
   │     ├─ test_embeddings.pkl
   │     ├─ train_embeddings.pkl
   │     └─ val_embeddings.pkl
   └─ tieredImageNet/
      └─ center/
         ├─ test_embeddings.pkl
         ├─ train_embeddings.pkl
         └─ val_embeddings.pkl
```

# 环境要求

- 硬件（GPU or Ascend）
    - 使用GPU处理器来搭建硬件环境。
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- GPU处理器环境运行

  ```bash
  # 运行训练示例
  bash scripts/run_train_gpu.sh [DEVICE_NUM] [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [SAVE_PATH]
  # 例如：
  bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1
  # 运行分布式训练示例
  bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/8P_mini_1
  # 例如：
  bash scripts/run_eval_gpu.sh [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [CKPT_FILE]
  # 运行评估示例
  bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1/xxx.ckpt

  ```

- Ascend处理器环境运行

  ```bash
  # 运行训练示例
  bash scripts/run_train_gpu.sh [DEVICE_ID] [DEVICE_TARGET] [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [SAVE_PATH]
  # 例如：
  bash scripts/run_train_ascend.sh 6 Ascend /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpts/1P_mini_5
  # 运行分布式训练示例
  bash scripts/run_distribution_ascend.sh [RANK_TABLE_FILE] [DEVICE_TARGET] [DATA_PATH] [DATA_NAME] [NUM_TR_EXAMPLES_PER_CLASS] [SAVE_PATH]
  # 例如：
  bash scripts/run_distribution_ascend.sh ./hccl_8p_01234567_127.0.0.1.json Ascend /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpts/8P_mini_5
  # 运行评估示例
  bash scripts/run_eval_gpu.sh [DEVICE_ID] [DATA_PATH] [CKPT_FILE]
  # 例如
  bash scripts/run_eval_ascend.sh 4 Ascend /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/1P_mini_5/xxx.ckpt
  ```

以上为第一个实验示例，其余三个实验请参考训练部分。

# 脚本说明

## 脚本及样例代码

```bash
├─ LEO
   ├─ README.md                   # LEO相关说明
   ├─ train.py                    # 训练脚本
   ├─ eval.py                     # 评估脚本
   ├─ scripts
   │  ├─ run_distribution_ascend.sh          # 启动8卡Ascend训练
   │  ├─ run_eval_ascend.sh           # ascend启动评估
   │  ├─ run_eval_gpu.sh              # gpu启动评估
   │  ├─ run_train_ascend.sh          # ascend启动训练
   │  └─ run_train_gpu.sh             # gpu启动训练
   ├─ src
   │  ├─ data.py                  # 数据处理
   │  ├─ model.py                 # LEO模型
   │  ├─ outerloop.py             # 外循环训练代码
   │  └─ trainonestepcell.py      # 单词循环训练代码
   ├─ config
   │  ├─ LEO-N5-K1_miniImageNet_config.yaml       # miniImageNet 1-shot配置
   │  ├─ LEO-N5-K5_miniImageNet_config.yaml       # miniImageNet 5-shot配置
   │  ├─ LEO-N5-K1_tieredImageNet_config.yaml     # tieredImageNet 1-shot配置
   │  └─ LEO-N5-K5_tieredImageNet_config.yaml     # tieredImageNet 5-shot配置
   ├─ model_utils
   │  └─ config.py                # 读取配置
   └─ embeddings               # 数据集特征向量
```

## 脚本参数

在default_config.yaml中可以配置训练参数和评估参数。

初始配置

  ```python
  enable_modelarts: False
  data_url: ""
  train_url: ""
  checkpoint_url: ""
  device_target: "GPU"
  device_num: 1
  data_path: "/home/mindspore/dataset/embeddings/"
  save_path: "./checkpoint"
  ckpt_file: ""
  enable_profiling: False
  ```

配置内循环模型参数

  ```python
  dataset_name: "miniImageNet"
  embedding_crop: "center"
  train_on_val: False
  inner_unroll_length: 5  
  finetuning_unroll_length: 5
  num_latents: 64
  inner_lr_init: 1.0
  finetuning_lr_init: 0.001
  dropout_rate: 0.3   #超参
  kl_weight: 0.001    #超参
  encoder_penalty_weight: 1E-9   #超参
  l2_penalty_weight: 0.0001      #超参
  orthogonality_penalty_weight: 303.0   #超参
  ```

配置外循环

  ```python
  num_classes: 5
  num_tr_examples_per_class: 1
  num_val_examples_per_class: 15
  metatrain_batch_size: 12
  metavalid_batch_size: 200
  metatest_batch_size: 200
  num_steps_limit: int(1e5)
  outer_lr: 0.004      #超参
  gradient_threshold: 0.1
  gradient_norm_threshold: 0.1
  total_steps: 200000
  ```

更多配置细节请参考config文件夹，**启动训练之前请根据不同的实验设置上述超参数。**

## 训练过程

- 四个实验设置不同的超参

| 超参                           | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
| ------------------------------ |---------------------|---------------------|-----------------------| --------------------- |
| `dropout`                      | 0.3                 | 0.3                 | 0.2                   | 0.3                   |
| `kl_weight`                    | 0.001               | 0.001               | 0                     | 0.001                 |
| `encoder_penalty_weight`       | 1E-9                | 2.66E-7             | 5.7E-1                | 5.7E-6                |
| `l2_penalty_weight`            | 0.0001              | 8.5E-6              | 5.10E-6               | 3.6E-10               |
| `orthogonality_penalty_weight` | 303.0               | 0.00152             | 4.88E-1               | 0.188                 |
| `outer_lr`                     | 0.005               | 0.005               | 0.005                 | 0.0025                |

### 训练

- 配置好上述参数后，GPU环境运行

  ```bash
  bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1
  bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/1P_mini_5
  bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/1P_tiered_1
  bash scripts/run_train_gpu.sh 1 /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/1P_tiered_5
  ```

- 配置好上述参数后，AScend环境运行

  ```bash
  bash scripts/run_train_ascend.sh 6 Ascend /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpts/1P_mini_1
  bash scripts/run_train_ascend.sh 6 Ascend /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpts/1P_mini_5
  bash scripts/run_train_ascend.sh 6 Ascend /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/1P_tiered_1
  bash scripts/run_train_ascend.sh 6 Ascend /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/1P_tiered_5
  ```

  训练将在后台运行，您可以通过`1P_miniImageNet_1_train.log`等日志文件查看训练过程。
  训练结束后，您可在 ` ./ckpt/1P_mini_1` 等checkpoint文件夹下找到检查点文件。

### 分布式训练

- 配置好上述参数后，GPU环境运行

  当上述脚本设置DEVICE_NUM > 1时将自动启动分布式训练，例如使用如下代码启用8卡分布式训练

  ```bash
  bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/8P_mini_1
  bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/8P_mini_5
  bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/8P_tiered_1
  bash scripts/run_train_gpu.sh 8 /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/8P_tiered_5
  ```

- 配置好上述参数后，Ascend环境运行

  ```bash
  bash scripts/run_distribution_ascend.sh ./hccl_8p_01234567_127.0.0.1.json Ascend /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpts/8P_mini_1
  bash scripts/run_distribution_ascend.sh ./hccl_8p_01234567_127.0.0.1.json Ascend /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpts/8P_mini_5
  bash scripts/run_distribution_ascend.sh ./hccl_8p_01234567_127.0.0.1.json Ascend /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpts/8P_tired_1
  bash scripts/run_distribution_ascend.sh ./hccl_8p_01234567_127.0.0.1.json Ascend /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpts/8P_tired_5
  ```

  与单卡训练一样，可以在`8P_miniImageNet_1_train.log`文件查看训练过程，并在默认`./ckpt/8P_mini_1`等checkpoint文件夹下找到检查点文件。

## 评估过程

### 评估

- **评估前请确认设置了相应训练时相同的超参数**

- GPU环境运行

  ```bash
  bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1/xxx.ckpt
  bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/1P_mini_5/xxx.ckpt
  bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/1P_tiered_1/xxx.ckpt
  bash scripts/run_eval_gpu.sh /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/1P_tiered_5/xxx.ckpt
  ```

- Ascend环境运行

  ```bash
  bash scripts/run_eval_ascend.sh 0 Ascend /home/mindspore/dataset/embeddings/ miniImageNet 1 ./ckpt/1P_mini_1/xxx.ckpt
  bash scripts/run_eval_ascend.sh 0 Ascend /home/mindspore/dataset/embeddings/ miniImageNet 5 ./ckpt/1P_mini_5/xxx.ckpt
  bash scripts/run_eval_ascend.sh 0 Ascend /home/mindspore/dataset/embeddings/ tieredImageNet 1 ./ckpt/1P_tiered_1/xxx.ckpt
  bash scripts/run_eval_ascend.sh 0 Ascend /home/mindspore/dataset/embeddings/ tieredImageNet 5 ./ckpt/1P_tiered_5/xxx.ckpt
  ```

  评估将在后台运行，您可以通过`1P_miniImageNet_1_eval.log`等日志文件查看评估过程。

# 模型描述

## 性能

### 训练性能

- 训练参数

| 参数          | LEO                                                         | Ascend                                        |
| -------------| ----------------------------------------------------------- |-----------------------------------------------|
| 资源          | NVIDIA GeForce RTX 3090；CUDA核心 10496个；显存 24GB | Ascend 910; CPU 24cores; 显存 256G; OS Euler2.8 |
| 上传日期       | 2022-03-27                                             | 2022-06-12                                    |
| MindSpore版本 | 1.7.0                                                      | 1.5.0                                         |
| 数据集        | miniImageNet                                                 | miniImageNet                                  |
| 优化器        | Adam                                                         | Adam                                          |
| 损失函数       | Cross Entropy Loss                                           | Cross Entropy Loss                            |
| 输出          | 准确率                                                        | 准确率                                           |
| 损失          | GANLoss,L1Loss,localLoss,DTLoss                             | GANLoss,L1Loss,localLoss,DTLoss               |
| 微调检查点     | 672KB (.ckpt文件)                                     | 672KB (.ckpt文件)                               |

- GPU评估性能

| 实验 | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
| ----- | ------------------- | ------------------- | --------------------- | --------------------- |
| 单卡(速度，总时长) | 90毫秒/步；413分钟 | 90毫秒/步；411分钟 | 130毫秒/步；522分钟 | 150毫秒/步；531分钟 |
| 多卡(速度，总时长) | xx毫秒/步；xx分钟   | xx毫秒/步；xx分钟 | xx毫秒/步；xx分钟 | xx毫秒/步；xx分钟     |

### 评估精度

- 评估参数

| 参数          | LEO                                                         | Ascend                                        |
| ------------ | ----------------------------------------------------------- |-----------------------------------------------|
| 资源          | NVIDIA GeForce RTX 3090；CUDA核心 10496个；显存 24GB | Ascend 910; CPU 24cores; 显存 256G; OS Euler2.8 |
| 上传日期       | 2022-03-27                                              | 2022-06-12                                    |
| MindSpore版本 | 1.7.0                                                      |1.5.0                                         |
| 数据集        | miniImageNet                                                 | miniImageNet                                  |
| 输出          | 准确率                                                        | 准确率                                           |

- 评估精度

| 代码   | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
| ----- | ------------------- | ------------------- | --------------------- | --------------------- |
| 参考工程（实测） | 58.46 ± 0.08%   | 75.59 ± 0.12%    | 66.47 ± 0.05%      | 80.80 ± 0.09%  |
| 此模型 | 59.59 ± 0.16% | 75.60 ± 0.10% | 66.44 ± 0.17%   | 80.94 ± 0.12% |

# 随机情况说明

使用了train.py中的随机种子。

# Resource说明

网络训练效果依赖初始化，为了达到更高的精度标准，此处提供一个可达标精度初始化文件[leo_ms_init.ckpt](https://download.mindspore.cn/thirdparty/)，下载后放入resource文件夹下。同样也可以在pytorch上初始化后，使用convert.py文件进行转换。

```bash
python convert.py  --toch_init [torch_init.pth]
```

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。