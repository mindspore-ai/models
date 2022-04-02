# 目录

- [目录](#目录)
- [IBN-Net概述](#IBN-Net概述)
- [IBN-Net示例](#IBN-Net示例)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [预训练模型](#预训练模型)
        - [训练过程](#训练过程)
            - [训练](#训练)
            - [分布式训练](#分布式训练)
        - [评估过程](#评估过程)
            - [评估](#评估)
        - [导出mindir模型](#导出mindir模型)
        - [推理过程](#推理过程)
            - [用法](#用法)
            - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# IBN-Net概述

卷积神经网络（CNNs）在许多计算机视觉问题上取得了巨大的成功。与现有的设计CNN架构的工作不同，论文提出了一种新的卷积架构IBN-Net，它可以提高单个域中单个任务的性能，这显著提高了CNN在一个领域（如城市景观）的建模能力以及在另一个领域（如GTA5）的泛化能力，而无需微调。IBN-Net将InstanceNorm（IN）和BatchNorm（BN）作为构建块进行了集成，并可以封装到许多高级的深度网络中以提高其性能。这项工作有三个关键贡献。（1） 通过深入研究IN和BN，我们发现IN学习对外观变化不变的特征，例如颜色、样式和虚拟/现实，而BN对于保存内容相关信息是必不可少的。（2） IBN-Net可以应用于许多高级的深层体系结构，如DenseNet、ResNet、ResNeXt和SENet，并在不增加计算量的情况下不断地提高它们的性能。（3） 当将训练好的网络应用到新的领域时，例如从GTA5到城市景观，IBN网络作为领域适应方法实现了类似的改进，即使不使用来自目标领域的数据。

[论文](https://arxiv.org/abs/1807.09441)： Pan X ,  Ping L ,  Shi J , et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net[C]// European Conference on Computer Vision. Springer, Cham, 2018.

# IBN-Net示例

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)
训练集：1,281,167张图片+标签
验证集：50,000张图片+标签
测试集：100,000张图片

# 环境要求

- 硬件：Ascend/GPU
    - 使用Ascend/GPU处理器来搭建硬件环境。

- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 分布式训练运行示例
bash scripts/run_distribute_train.sh /path/dataset /path/evalset pretrained_model.ckpt rank_size

# 单机训练运行示例
bash scripts/run_standalone_train.sh /path/dataset /path/evalset pretrained_model.ckpt device_id

# 运行评估示例
bash scripts/run_eval.sh
```

## 脚本说明

## 脚本和样例代码

```path
└── IBNNet  
 ├── README.md                           // IBNNet相关描述
 ├── ascend310_infer                     //310推理
  ├── inc
   ├── utils.h
  ├── src
   ├── main.cc
   ├── utils.cc
  ├── build.sh
  └── CMakeLists.txt
 ├── scripts
  ├── run_310_infer.sh               // 用于310推理的shell脚本
  ├── run_distribute_train.sh        // 用于分布式训练的shell脚本
  ├── run_distribute_train_gpu.sh    // 用于GPU分布式训练的shell脚本
  ├── run_standalone_train.sh        // 用于单机训练的shell脚本
  ├── run_standalone_train.sh        // 用于GPU单机训练的shell脚本
  ├── run_eval.sh                    // 用于评估的shell脚本
  └── run_eval.sh                    // 用于GPU评估的shell脚本
 ├── src
  ├── loss.py                         //损失函数
  ├── lr_generator.py                 //生成学习率
  ├── config.py                       // 参数配置
  ├── dataset.py                      // 创建数据集
  ├── resnet_ibn.py                   // IBNNet架构
 ├── utils
  ├── pth2ckpt.py                       //转换pth文件为ckpt文件
 ├── export.py
 ├── eval.py                             // 测试脚本
 ├── train.py                            // 训练脚本
 ├── preprocess.py                       // 310推理数据预处理
 ├── preprocess.py                       // 310推理数据后处理

```

## 脚本参数

```python
train.py和config.py中主要参数如下：

-- use_modelarts：是否使用modelarts平台训练。可选值为True、False。
-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- device_num：使用train.sh进行分布式训练时使用的设备数。
-- train_url：checkpoint的输出路径。
-- data_url：训练集路径。
-- ckpt_url：checkpoint路径。
-- eval_url：验证集路径。

```

## 预训练模型

可以使用utils/pth2ckpt.py将预训练的pth文件转换为ckpt文件。
pth预训练模型文件获取路径如下：[预训练模型](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth)

## 训练过程

### 训练

- 在Ascend环境训练

```shell
bash scripts/run_standalone_train.sh /path/dataset /path/evalset pretrained_model.ckpt device_id
```

- 在GPU环境训练

```shell
bash scripts/run_standalone_train_gpu.sh /path/dataset /path/evalset pretrained_model.ckpt
```

### 分布式训练

- 在Ascend环境训练

```shell
bash scripts/run_distribute_train.sh /path/dataset /path/evalset pretrained_model.ckpt rank_size
```

上述shell脚本将在后台运行分布训练。可以通过`device[X]/test_*.log`文件查看结果。
采用以下方式达到损失值：

```log
epoch: 12 step: 2502, loss is 1.7709649
epoch time: 331584.555 ms, per step time: 132.528 ms
epoch: 12 step: 2502, loss is 1.2770984
epoch time: 331503.971 ms, per step time: 132.496 ms
...
epoch: 82 step: 2502, loss is 0.98658705
epoch time: 331877.856 ms, per step time: 132.645 ms
epoch: 82 step: 2502, loss is 0.82476664
epoch time: 331689.239 ms, per step time: 132.570 ms

```

- 在GPU环境训练

```shell
bash scripts/run_distribute_train_gpu.sh /path/dataset /path/evalset pretrained_model.ckpt rank_size
```

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet数据集

```bash
bash scripts/run_eval.sh path/evalset path/ckpt
```

上述命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```bash
{'Accuracy': 0.7785483870967742}
```

- 在GPU环境运行时评估ImageNet数据集

```bash
bash scripts/run_eval_gpu.sh path/evalset path/ckpt
```

上述命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```bash
============== Accuracy:{'top_5_accuracy': 0.93684, 'top_1_accuracy': 0.7743} ==============
```

## 导出mindir模型

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中进行选择。

# 推理过程

## 用法

在执行推理之前，需要通过export.py导出mindir文件。

```bash
# Ascend310 推理
bash run_310_infer.sh [MINDIR_PATH] [DATASET_PATH]
```

`MINDIR_PATH` 为mindir文件路径，`DATASET_PATH` 表示数据集路径。

### 结果

推理结果保存在当前路径，可在acc.log中看到最终精度结果。

# 模型描述

## 性能

### 训练性能

| 参数          | IBN-Net                                         |
| ------------- | ----------------------------------------------- |
| 模型版本      | resnet50_ibn_a                                  |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G |
| 上传日期      | 2021-03-30                                     |
| MindSpore版本 | 1.1.1-c76-tr5                          |
| 数据集        | ImageNet2012                                       |
| 训练参数      | lr=0.1; gamma=0.1                      |
| 优化器        | SGD                                             |
| 损失函数      | SoftmaxCrossEntropyExpand                       |
| 输出          | 概率                                            |
| 损失          | 0.6                                            |
| 速度 | 1卡：127毫秒/步；8卡：153毫秒/步 |
| 总时间 | 1卡：65小时；8卡：10小时 |
| 参数(M) | 46.15 |
| 微调检查点 | 293M （.ckpt file） |
| 脚本 | [脚本路径](https://gitee.com/mindspore/models/tree/master/research/cv/ibnnet) |

### 评估性能

| 参数          | IBN-Net            |
| ------------- | ------------------ |
| 模型版本      | resnet50_ibn_a     |
| 资源          | Ascend 910         |
| 上传日期      | 2021/03/30        |
| MindSpore版本 | 1.1.1-c76-tr5      |
| 数据集        | ImageNet2012          |
| 输出          | 概率               |
| 准确性        | 1卡：77.45%; 8卡：77.45% |

# 随机情况说明

在dataset.py中，我们设置了“create_dataset_ImageNet”函数内的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。

