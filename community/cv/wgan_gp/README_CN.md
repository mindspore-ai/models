# 目录

<!-- TOC -->

- [目录](#目录)
- [WGAN-GP描述](#wgan-gp描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [单机训练](#单机训练)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# WGAN-GP描述

WGAN-GP(Wasserstein GAN-Gradient Penalty)是一种包含DCGAN结构判别器与生成器的生成对抗网络，它在WGAN基础上用梯度惩罚替代了梯度剪裁，在损失函数引入了判别器输出相对输入的二阶导数，作为规范判别器损失模的函数，解决了WGAN随机不收敛与生成样本质量差的问题。

[论文](https://arxiv.org/pdf/1704.00028v3.pdf)：Improved Training of Wasserstein GANs

# 模型架构

WGAN-GP网络包含两部分，生成器网络和判别器网络。判别器网络采用卷积DCGAN的架构，即多层二维卷积相连。生成器网络采用卷积DCGAN生成器结构。输入数据包括真实图片数据和噪声数据，数据集Cifar10的真实图片resize到32*32，噪声数据随机生成。

# 数据集

## [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：175M, 60000张10分类彩色图像
    - 训练集：146M，共50000张图像。
    - 注：对于生成对抗网络，推理部分是传入噪声数据生成图片，故无需使用测试集数据。
- 数据格式：二进制文件

## [LSUN-Bedrooms](<http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip>)

- 数据集大小：42.8G
    - 训练集：42.8G，共3033044张图像。
    - 注：对于生成对抗网络，推理部分是传入噪声数据生成图片，故无需使用测试集数据。

- 数据格式：原始数据格式为lmdb格式，需要使用LSUN官网格式转换脚本把lmdb数据export所有图片，并将Bedrooms这一类图片放到同一文件夹下。
  - 注：LSUN数据集官网的数据格式转换脚本地址：(<https://github.com/fyu/lsun>)

# 环境要求

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 运行单机训练示例：
bash run_standalone_train_ascend.sh [DATAROOT] [DEVICE_ID]
# bash run_standalone_train_gpu.sh [DATAROOT] [DEVICE_ID]

# 运行评估示例
bash run_standalone_eval_ascend.sh [DEVICE_ID] [CKPT_FILE_PATH] [OUTPUT_DIR]
# bash run_standalone_eval_gpu.sh [DEVICE_ID] [CKPT_FILE_PATH] [OUTPUT_DIR]
```

# 脚本说明

## 脚本及样例代码

  ```md
├── WGAN-GP
    ├── README.md                  // WGAN-GP相关说明
    ├── scripts
    │   ├── run_standalone_train_ascend.sh           // 单机到Ascend处理器的shell脚本
    │   ├── run_standalone_eval_ascend.sh            // Ascend评估的shell脚本
    │   ├── run_standalone_train_gpu.sh           // 单卡GPU的shell脚本
    │   ├── run_standalone_eval_gpu.sh            // 单卡GPU评估的shell脚本
    ├── src
    │   ├── dataset.py             // 创建数据集及数据预处理
    │   ├── model.py               // WGAN-GP生成器与判别器创建
    │   ├── dcgan_model.py         // 基于DCGAN的WGAN-GP生成器与判别器定义
    │   ├── resgan_model.py        // 基于ResNet的WGAN-GP生成器与判别器定义
    │   ├── args.py                // 参数配置文件
    │   ├── cell.py                // 模型单步训练文件
    ├── requirements.txt           // 所需第三方库
    ├── README.md                  // 模型相关说明
    ├── train.py                   // 训练脚本
    ├── eval.py                    // 评估脚本
  ```

## 脚本参数

在args.py中可以同时配置训练参数、评估参数及模型导出参数。

  ```python
  # common_config
  'device_target': 'Ascend' or 'GPU', # 运行设备
  'device_id': 0, # 用于训练或评估数据集的设备ID

  # train_config
  'dataset': 'cifar10' or 'lsun' # 使用数据集
  'dataroot': None, # 数据集路径，必须输入，不能为空
  'workers': 8, # 数据加载线程数
  'batchSize': 64, # 批处理大小
  'imageSize': 32, # 图片尺寸大小，cifar10数据集为32，lsun数据集为64
  'DIM': 128, # GAN网络隐藏层大小
  'niter': 1200, # 网络训练的epoch数
  'save_iterations': 1000, # 保存模型文件的生成器迭代次数
  'lrD': 0.0001, # 判别器初始学习率
  'lrG': 0.0001, # 生成器初始学习率
  'beta1': 0.5, # Adam优化器beta1参数
  'beta2': 0.9, # Adam优化器beta2参数
  'netG': '', # 恢复训练的生成器的ckpt文件路径
  'netD': '', # 恢复训练的判别器的ckpt文件路径
  'Diters': 5, # 每训练一次生成器需要训练判别器的次数
  'experiment': None, # 保存模型和生成图片的路径，若不指定，则使用默认路径

  # eval_config
  'ckpt_file': None, # 训练时保存的生成器的权重文件.ckpt的路径，必须指定
  'output_dir': None, # 生成图片的输出路径，必须指定
  ```

更多配置细节请参考脚本`args.py`。

## 训练过程

### 单机训练

- Ascend或GPU环境运行

```bash
# Ascend
bash run_standalone_train_ascend.sh [DATAROOT] [DEVICE_ID]

# GPU
bash run_standalone_train_gpu.sh [DATAROOT] [DEVICE_ID]
```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在存储的文件夹（默认是./samples）下找到生成的图片、检查点文件和.json文件。采用以下方式得到损失值：

```bash
[0/1200][230/937][23] Loss_D: -379.555344 Loss_G: -33.761238
[0/1200][235/937][24] Loss_D: -214.557617 Loss_G: -23.762344
```

  以上脚本默认使用cifar10数据集进行训练，如需在lsun数据集上训练，可使用以下方式

```bash
# Ascend
python train.py --dataset lsun --dataroot [DATAROOT] --device_id [DEVICE_ID] --imageSize 64 --niter 5 --device_target Ascend > ./train.log 2>&1 &

# GPU
python train.py --dataset lsun --dataroot [DATAROOT] --device_id [DEVICE_ID] --imageSize 64 --niter 5 --device_target GPU > ./train.log 2>&1 &
```

## 推理过程

### 推理

- 在Ascend或GPU环境下评估

  在运行以下命令之前，请检查用于推理的检查点和json文件路径，并设置输出图片的路径。

```bash
# Ascend
bash run_standalone_eval_ascend.sh [DEVICE_ID] [CKPT_FILE_PATH] [OUTPUT_DIR]

# GPU
bash run_standalone_eval_gpu.sh [DEVICE_ID] [CKPT_FILE_PATH] [OUTPUT_DIR]
```

  上述python命令将在后台运行，您可以通过eval/eval.log文件查看日志信息，在输出图片的路径下查看生成的图片。

  以上脚本默认使用cifar10数据集进行训练，如需在lsun数据集上训练，可使用以下方式

  ```bash
# Ascend
python eval.py   --dataset lsun --imageSize 64 --device_id $1 --ckpt_file $CKPT_FILE_PATH --output_dir $OUTPUT_DIR --device_target Ascend > ./eval.log 2>&1 &

# GPU
python eval.py   --dataset lsun --imageSize 64 --device_id $1 --ckpt_file $CKPT_FILE_PATH --output_dir $OUTPUT_DIR --device_target GPU > ./eval.log 2>&1 &
  ```

# 模型描述

## 性能

### 训练性能

| 参数          | Ascend                                        | GPU                                           |
| ------------- | --------------------------------------------- | --------------------------------------------- |
| 资源          | Ascend 910 ；CPU 2.60GHz，192核；内存：755G   | GPU 3090; CPU 2.60GHz ，64核，内存：256GB；   |
| 上传日期      | 2022-08-01                                    | 2022-11-12                                    |
| MindSpore版本 | 1.8.0                                         | 1.9.0                                         |
| 数据集        | CIFAR-10                                      | CIFAR-10                                      |
| 训练参数      | max_epoch=1200, batch_size=64, lr_init=0.0001 | max_epoch=1200, batch_size=64, lr_init=0.0001 |
| 优化器        | Adam                                          | Adam                                          |
| 损失函数      | 自定义损失函数                                | 自定义损失函数                                |
| 输出          | 生成的图片                                    | 生成的图片                                    |
| 速度          | 单卡：0.06秒/步                               | 单卡：0.05秒/步                               |

生成图片效果如下：

![GenSample1](imgs/fake_samples_200000.png "生成的图片样本")

# 随机情况说明

在train.py中，我们设置了随机种子。

# 代码说明

基于mindspore的另一个实现版本在: [**WGAN_PG**](https://gitee.com/mindspore/models/tree/master/community/cv/wgan_gp)

# reference

**[improved_wgan_training](https://github.com/igul222/improved_wgan_training)**

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
