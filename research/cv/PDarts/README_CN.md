# 目录

<!-- TOC -->

- [目录](#目录)
- [PDarts描述](#PDarts描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
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
        - [导出MindIR](#导出MindIR)
        - [在Ascend310执行推理](#在Ascend310执行推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练准确率结果](#训练准确率结果)
        - [训练性能结果](#训练性能结果)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# PDarts描述

通过在cifar10数据集上进行搜索，从而得到合适的模型架构；本代码实现的是论文中在cifar10数据上搜索而来的模型架构，并用于训练cifar10格式数据集的PDarts模型；通过600个epoch（batchsize=128）的训练，最终在cifar10的验证集上acc top1达到97.1%，acc top5达到99.93%。有关该模型更详细的描述，可查阅[此论文](https://arxiv.org/pdf/1904.12760.pdf)。本代码是[MindSpore](https://www.mindspore.cn/)上的一个实现。

本代码中还包含用于启动在Ascend910平台训练、评估和Ascend310平台推理例程的脚本。

# 模型架构

![pipeline](https://github.com/chenxin061/pdarts/raw/master/pipeline2.jpg)

本图片来源于模型论文

# 数据集

使用的数据集： [cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)  数据集的默认配置如下：

- 训练数据集预处理：
    - 图像的输入尺寸：32\*32
    - 随机裁剪：RandomCrop(size=(32,32),padding=(4,4,4,4))
    - 图像翻转度数：5度，双三次插值
    - 根据平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理：
    - 原始图片尺寸输入，并做与训练过程中一致的归一化操作

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/enable_mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

  ```bash
  # 训练示例

  bash ./scripts/run_standalone_train_ascend /data/cifar-10-binary ./output
  bash ./scripts/run_distribution_train_ascend ../rank_table.json /data/cifar-10-binary ../output
  bash ./scripts/run_distribution_train_gpu.sh 8 0,1,2,3,4,5,6,7 ../cifar-10-binary/ ./output/

  # 评估示例
  bash ./scripts/run_standalone_eval_ascend.sh /data/cifar-10-binary/val ./output/model_checkpoint.ckpt
  bash ./scripts/run_standalone_eval_gpu.sh ../cifar-10-binary/val/ model_checkpoint.ckpt
  ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                                      // 所有模型的说明
    ├── PDarts
        ├── README.md                                  // PDarts相关说明
        ├── scripts
        │   ├── run_standalone_eval_ascend.sh          // Ascend评估shell脚本
        │   ├── run_standalone_eval_gpu.sh             // GPU评估shell脚本
        │   ├── run_export.sh                          // 导出模型shell脚本
        │   ├── run_standalone_train_ascend.sh         // Ascend单卡训练shell脚本
        │   ├── run_standalone_train_gpu.sh            // GPU单卡训练shell脚本
        │   ├── run_distribution_train_ascend.sh       // Ascend 8卡训练shell脚本
        │   ├── run_distribution_train_gpu.sh          // GPU 8卡训练shell脚本
        │   ├── run_infer_310.sh                       // Ascend310环境推理shell脚本
        ├── src
        │   ├── call_backs                             // 训练过程中的回调方法
        │   ├── dataset                                // 数据读取
        │   ├── genotypes                              // 模型中所使用的一些结构类型名称等
        │   ├── loss                                   // 自定义loss
        │   ├── model                                  // PDarts模型架构
        │   ├── my_utils                               // 封装的工具方法
        │   ├── operations                             // 封装的组合算子
        ├── train.py                                   // 训练脚本
        ├── eval.py                                    // 评估脚本
        ├── export.py                                  // 模型导出脚本
        ├── preprocess.py                              // Ascend310推理的数据预处理脚本，会在run_infer_310.sh中被调用
        ├── postprocess.py                             // Ascend310推理的结果后处理脚本，会在run_infer_310.sh中被调用
        ├── ascend310_infer                            // 该文件夹内部为在Ascend310环境上部署推理的C++实现代码
```

## 脚本参数

可通过`train.py`脚本中的参数修改训练行为。`train.py`脚本中的参数如下：

```bash
  --device_target         设备类型，支持Ascend、GPU
  --local_data_root       数据拷贝的缓存目录（主要针对在modelarts上运行时使用）
  --data_url              数据路径
  --train_url             训练结果输出路径
  --batch_size            默认值128（经测试，batch_size不同会影响最终的精度）
  --load_weight           加载预训练权重的路径
  --no_top                是否加载头部fc全连接层的权重
  --learning_rate         初始学习率，默认0.025
  --momentum              梯度下降时的动量值，默认0.9
  --weight_decay          L2权重衰减，默认3e-4
  --epochs                训练的epoch数量，默认值600
  --init_channels        模型架构中初始化channels的数量，默认36
  --layers               模型架构中layer的总数，默认20
  --auxiliary            是否使用辅助塔，默认为True
  --auxiliary_weight      auxiliary loss的权重比例，当auxiliary为True时有效
  --drop_path_prob        dropout的比例
  --arch                  模型架构，默认值为'PDARTS'
  --amp_level             混合精度级别，Ascend910环境建议使用O3，GPU环境建议用O2
  --optimizer            训练用的优化器，默认使用Momentum
  --cutout_length        数据的裁剪长度，默认为16
```

## 训练过程

### 训练

- Ascend910处理器、GPU环境运行

  ```bash
  单卡Ascend910
  bash ./scripts/run_standalone_train_ascend /data/cifar-10-binary ./output
  8卡Ascend910
  bash ./scripts/run_distribution_train_ascend ../rank_table.json /data/cifar-10-binary ../output
  注：单卡Ascend910训练启动脚本一共有2个参数，8卡训练脚本有3个参数，分别为[rank_table配置文件(8卡训练脚本需要使用)] [cifar10数据集路径] [训练输出路径]

  单卡GPU
  bash ./scripts/run_standalone_train_gpu.sh ./cifar-10-binary/ ./output/
  8卡GPU
  bash ./scripts/run_distribution_train_gpu.sh 8 0,1,2,3,4,5,6,7 ../cifar-10-binary/ ./output/
  注：单卡GPU训练启动脚本一共有2个参数，8卡训练脚本有4个参数，分别为[DEVICE_NUM(8卡环境需要设置为8)][VISIABLE_DEVICES(0,1,2,3,4,5,6,7，即每个GPU分配的id，中间用逗号隔开)] [cifar10数据集路径] [训练输出路径]
  ```

  cifar10数据集的要求格式

  ```text
  数据集刚下载下来时格式可能不符合本代码的要求，需要手动修改只如下结构：
  ├── cifar-10-binary
      ├── train
          ├── data_batch_1.bin
          ├── data_batch_2.bin
          ├── data_batch_3.bin
          ├── data_batch_4.bin
          ├── data_batch_5.bin
      ├── val
          ├── test_batch.bin
  ```

  训练脚本的log如下（训练过程中边训练，边在验证集上评估）：

  ```text
  epoch: 577 step: 390, loss is 0.029428542
  epoch time: 73852.709 ms, per step time: 189.366 ms
  ==========val metrics:{'top_1_accuracy': 0.9707532051282052, 'top_5_accuracy': 0.999198717948718, 'loss': 0.16085085990981987} use times:5253.3118724823ms=========================
  =================save checkpoint....====================
  ==============save checkpoint finished===================
  The best accuracy is 0.9707532051282052
  epoch: 578 step: 390, loss is 0.07095037
  epoch time: 73861.921 ms, per step time: 189.390 ms
  ==========val metrics:{'top_1_accuracy': 0.9703525641025641, 'top_5_accuracy': 0.999198717948718, 'loss': 0.15911210587439248} use times:5181.187629699707ms=========================
  The best accuracy is 0.9707532051282052
  epoch: 579 step: 390, loss is 0.044823572
  epoch time: 73873.990 ms, per step time: 189.420 ms
  ==========val metrics:{'top_1_accuracy': 0.9710536858974359, 'top_5_accuracy': 0.9993990384615384, 'loss': 0.1574480276172742} use times:5301.135063171387ms=========================
  =================save checkpoint....====================
  ==============save checkpoint finished===================
  ```

### 评估

- Ascend处理器环境

  运行以下命令进行评估。

  ```bash
  bash ./scripts/run_standalone_eval_ascend.sh /data/cifar-10-binary/val ./output/model_checkpoint.ckpt
  注：评估脚本参数一共为两个，分别是[验证集路径] [ckpt文件路径]；
      数据集格式与上面训练过程相同，并选择cifar-10-binary/val部分进行评估；
  ```

  单卡训练最终精度acc top1为97.1%，acc top5为99.93%

  8卡训练最终精度acc top1为97.01%，acc top5为99.91%

- GPU环境

  运行以下命令进行评估。

  ```bash
  bash ./scripts/run_standalone_eval_gpu.sh ../cifar-10-binary/val/ model_checkpoint.ckpt
  注：评估脚本参数一共为两个，分别是[验证集路径] [ckpt文件路径]；
      数据集格式与上面训练过程相同，并选择cifar-10-binary/val部分进行评估；
  ```

  8卡训练最终精度acc top1为97.205%，acc top5为99.939%

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出MindIR

```bash
bash ./scripts/run_export.sh model_checkpoint.ckpt MINDIR
注：该脚本有两个参数，分别为[ckpt文件路径] [导出文件格式]
```

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`run_export.sh`脚本导出。以下展示了使用minir模型执行推理的示例。
目前310环境仅支持batch_Size为1的推理，所以模型导出和推理中batchsize都默认设置成了1。

```bash
# Ascend310 inference
bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATASET_PATH]
```

- `MINDIR_PATH` mindir文件路径
- `DATASET_PATH` 推理数据集路径，如上面训练和评估中的cifar-10-binary/val

# 模型描述

## 性能

### 训练准确率结果

| 模型 | PDarts | PDarts |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本 | PDarts-Ascend | PDarts-GPU |
| 资源 | Ascend 910 | V100 |
| 上传日期 | 2021/6/9 | 2021/12/28 |
| MindSpore版本 | 1.2.0 Ascend | 1.5.0 GPU |
| 数据集 | cifar10 | cifar10 |
| 轮次 | 600 | 600 |
| 输出 | 概率 | 概率 |
| 损失 | 0.1574 | 0.1241 |
| 总时间 | 单卡：约15小时         8卡：约3.2小时 | 8卡：约7小时 |
| 训练精度 | 单卡：Top1：97.1%； Top5：99.93%         8卡：Top1：97.01%； Top5：99.91% | 8卡：Top1：97.205%； Top5：99.939% |

### 训练性能结果

| 模型 | PDarts | PDarts |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本 | PDarts-Ascend | PDarts-GPU |
| 资源 | Ascend 910 | V100 |
| 上传日期 | 2021/6/9 | 2021/12/28 |
| MindSpore版本 | 1.2.0 Ascend | 1.5.0 GPU |
| 数据集 | cifar10 |cifar10|
| batch_size | 单卡：128    8卡：32 | 8卡：32 |
| 输出 | 概率 |概率|
| 速度 | 单卡：189.4ms/step        8卡：65.5ms/step | 8卡：约180ms/step |

# 随机情况说明

dataset.py创建训练数据集时shuffle参数设置为True，评估数据时设置shuffle为False

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。  

