# PFNN

## [目录](#目录)

- [描述](#描述)

- [模型架构](#模型架构)

- [数据集](#数据集)

- [环境要求](#环境要求)

- [快速开始](#快速开始)

- [脚本说明](#脚本说明)

- [模型性能](#模型性能)

- [ModelZoo主页](https://gitee.com/mindspore/models)

## [描述](#目录)

PFNN (Penalty-free neural network)方法是一种基于神经网络的微分方程求解方法，适用于求解复杂区域上的二阶微分方程。该方法克服了已有类似方法在处理问题光滑性约束和边界约束上的缺陷，具有更高的精度，效率和稳定性。

[论文](https://www.sciencedirect.com/science/article/pii/S0021999120308597)：H. Sheng, C. Yang, PFNN: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries, Journal of Computational Physics 428 (2021) 110085.

## [模型架构](#目录)

PFNN采用神经网络逼近微分方程的解。不同于大多数只采用单个网络构造解空间的神经网络方法，PFNN采用两个网络分别逼近本质边界和区域其它部分上的真解。为消除两个网络之间的影响，一个由样条函数所构造的length factor函数被引入以分隔两个网络。为进一步降低问题对于解的光滑性需求，PFNN利用Ritz变分原理将问题转化为弱形式，消除损失函数中的高阶微分算子，从而降低最小化损失函数的困难，有利于提高方法的精度。

## [数据集](#目录)

PFNN根据方程信息和计算区域信息生成训练集和测试集。

- 训练集：分为内部集和边界集，分别在计算区域内部和边界上采样得到。
    - 内部集：在计算区域内部采样3600个点，并计算控制方程右端项在这些点上的值作为标签。
    - 边界集：在Dirichlet边界和Neumann边界上分别采样60和180个点，并计算边界方程右端项在这些点上的值作为标签。

- 测试集：在整个计算区域上采样10201个点，并计算真解在这些点上的值作为标签。

    注：该数据集在各向异性的扩散方程场景中使用。数据将在pfnn/Data/Data.py中处理

## [环境要求](#目录)

- 硬件（GPU/CPU）
- 框架
    - [Mindspore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [Mindspore教程](#https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [Mindspore Python API](#https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

### 训练过程

```shell
bash run_standalone_train_gpu.sh
python train.py --problem [PROBLEM] --g_epochs [G_EPOCHS] --f_epochs [F_EPOCHS] --g_lr [G_LR] --f_lr [F_LR] --device [DEVICE]
```

### 评估过程

```shell
bash run_standalone_eval_gpu.sh
python eval.sh --problem [PROBLEM] --device [DEVICE]
```

## [脚本说明](#目录)

### 文件描述

```shell
├── data
│   ├── data.py                     #根据方程生成数据
│   ├── dataset.py                  #生成数据集
│   └── __init__.py
├── eval.py                         #测试函数
├── README_CN.md
├── requirement.txt
├── scripts
│   ├── run_standalone_eval_gpu.sh  #测试执行脚本
│   └── run_standalone_train_gpu.sh #训练执行脚本
├── src
│   ├── callback.py
│   ├── __init__.py
│   └── model.py                    #网络模型
└── train.py                        #训练代码
```

## [模型性能](#目录)

| 参数          | GPU                                                          |
| ------------- | ------------------------------------------------------------ |
| 资源          | GPU(Tesla V100 PCI-E), Memory 16G                            |
| 更新时间      | 2021.10.25                                                   |
| Mindspore版本 | 1.5.0-rc1                                                    |
| 数据集        | 根据方程信息和计算区域信息生成                               |
| 问题          | Anisotropic Diffusion                                        |
| 输出          | 精度                                                         |
| 优化器        | Adam                                                         |
| Speed         | 12.2ms/step                                                   |
| 精度          | 0.00043                                                      |
| 训练耗时      | 55s                                                          |
| Scripts       | [Link](https://gitee.com/mindspore/models/tree/master/research/hpc/pfnn) |

## [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)
