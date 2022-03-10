# 目录

<!-- TOC -->

- [目录](#目录)
    - [RedNet30描述](#rednet30描述)
    - [模型结构](#模型结构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [训练过程](#训练过程)
        - [训练参数](#训练参数)
        - [训练启动](#训练启动)
    - [评估过程](#评估过程)
        - [评估参数](#评估参数)
        - [评估启动](#评估启动)
    - [转换过程](#转换过程)
    - [推理过程](#推理过程)
        - [在昇腾310上推理](#在昇腾310上推理)
    - [模型描述](#模型描述)
        - [训练性能结果](#训练性能结果)

<!-- /TOC -->

## RedNet30描述

RedNet30是一个使用encoder-decoder处理图像降噪任务的模型， 本项目是图像去躁模型RedNet30在mindspore上的复现。

论文: Mao X J ,  Shen C ,  Yang Y B . [Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections[J]](https://arxiv.org/pdf/1603.09056v2.pdf).  2016.

## 模型结构

网络由15层的conv block和15层的deconv block组成，其中下采样中的每一层是conv加relU，上采样过程中的每一层是deconv加relu。

## 数据集

训练集：[BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
测试集：[BSD200](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

其中，训练集是由[BSD300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) 训练集和验证集合成得到的300张彩色图像，验证集是BSD300的训练集的200张彩色图像。

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)
- 安装requirements.txt中的python包。
- 生成config json文件用于8卡训练。

## 脚本说明

```shell
.
└── rednet30
    ├─ README_CN.md
    ├── ascend310_infer
        ├──src                                    # 实现Ascend-310推理源代码
        ├──inc                                    # 实现Ascend-310推理源代码
        ├──build.sh                               # 构建Ascend-310推理程序的shell脚本
        └─CMakeLists.txt                          # 构建Ascend-310推理程序的CMakeLists
    ├─ scripts
        ├─run_standalone_train.sh                 # Ascend环境下的单卡训练脚本
        ├─run_standalone_train_gpu.sh             # GPU环境下的单卡训练脚本
        ├─run_distribute_train.sh                 # Ascend环境下的八卡并行训练脚本
        ├─run_distribute_train_gpu.sh             # GPU环境下的八卡并行训练脚本
        ├─run_eval.sh                             # Ascend环境下的评估脚本
        ├─run_infer_310.sh                        # Ascend-310推理shell脚本
        └─run_eval_gpu.sh                         # GPU环境下的评估脚本
    ├── src
        ├── dataset.py                            # 数据读取
        ├── get_input_data_310.py                 # 获取310推理噪声图片
        ├── get_input_data.py                     # 获取噪声图片
        └── model.py                              # 模型定义
    ├── export.py                                 # 导出MINDIR文件
    ├── preprocess.py                             # Ascend-310推理的数据准备脚本
    ├── postprocess.py                            # Ascend-310推理的数据后处理脚本
    ├── eval.py                                   # 评估脚本
    └── train.py                                  # 训练脚本
```

## 训练过程

可通过`train.py`脚本中的参数修改训练行为。`train.py`脚本中的参数如下：

### 训练参数

```bash
--dataset_path ./data/BSD300          # 训练数据路径
--platform 'GPU'                      # 训练设备
--is_distributed False                # 分布式训练
--patch_size 50                       # 输入数据大小
--batch_size 16                       # 批次大小
--num_epochs 1000                     # 训练轮次
--lr 0.0001                           # 学习率
--seed 1                              # 随机种子
--ckpt_save_max 5                     # ckpt最大保存数量
--init_loss_scale 65536.              # 初始loss scale
```

### 启动

您可以使用python或shell脚本进行训练。

```shell
# 训练示例
- running on Ascend with default parameters

  python:
      Ascend单卡训练示例：python train.py --dataset_path [DATA_DIR] --platform Ascend
      # example: python train.py --dataset_path ./data/BSD300 --platform Ascend

  shell:
      Ascend八卡并行训练: bash scripts/run_distribute_train.sh [DATA_DIR] [RANK_TABLE_FILE]
      # example: bash scripts/run_distribute_train.sh ./data/BSD300 ./rank_table_8p.json

      Ascend单卡训练示例: bash scripts/run_standalone_train.sh [DATA_DIR]
      # example: bash scripts/run_standalone_train.sh ./data/BSD300

- running on GPU with gpu default parameters

  python:
      Ascend单卡训练示例：python train.py --dataset_path [DATA_DIR] --platform GPU
      # example: python train.py --dataset_path ./data/BSD300 --platform GPU

  shell:
      GPU八卡并行训练: bash scripts/run_distribute_train_gpu.sh [DATA_DIR]
      # example: bash scripts/run_distribute_train_gpu.sh ./data/BSD300

      GPU单卡训练示例: bash scripts/run_standalone_train_gpu.sh [DATA_DIR]
      # example: bash scripts/run_standalone_train_gpu.sh ./data/BSD300
```

  分布式训练需要提前创建JSON格式的HCCL配置文件。

  运行分布式任务时需要用到RANK_TABLE_FILE指定的rank_table.json。您可以使用hccl_tools生成该文件，详见[链接](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) 。

## 评估过程

### 评估参数

```bash
--dataset_path ./data/BSD300                # 测试数据路径
--ckpt_path ./ckpt/RedNet30-1000_18.ckpt    # 测试ckpt文件路径
--platform 'GPU'                            # 训练设备
```

### 评估启动

您可以使用python或shell脚本进行评估。

```shell
# 评估前需要生成噪声图像，生成方法如下
  python ./src/get_input_data.py --dataset_path [DATA_DIR] --output_path [NOISE_IMAGE_DIR]

# Ascend评估示例
  python:
      python eval.py --dataset_path [DATA_DIR] --noise_path [NOISE_IMAGE_DIR] --ckpt_path [PATH_CHECKPOINT] --platform [PLATFORM]
      # example: python eval.py --dataset_path ./data/BSD200 --noise_path ./data/BSD200_jpeg_quality10 --ckpt_path ./train/ckpt/ckpt_0/RedNet30_0-1000_18.ckpt --platform 'Ascend'

  shell:
      bash scripts/run_eval.sh [DATA_DIR] [NOISE_IMAGE_DIR] [PATH_CHECKPOINT] [PLATFORM]
      # example: bash scripts/run_eval.sh ./data/BSD200 ./data/BSD200_jpeg_quality10 ./train/ckpt/ckpt_0/RedNet30_0-1000_18.ckpt Ascend

# GPU评估示例
  python:
      python eval.py --dataset_path [DATA_DIR] --noise_path [NOISE_IMAGE_DIR] --ckpt_path [PATH_CHECKPOINT] --platform [PLATFORM]
      # example: python eval.py --dataset_path ./data/BSD200 --noise_path ./data/BSD200_jpeg_quality10 --ckpt_path ./train/ckpt/ckpt_0/RedNet30_0-1000_18.ckpt --platform 'GPU'

  shell:
      bash scripts/run_eval_gpu.sh [PLATFORM] [DATA_DIR] [NOISE_IMAGE_DIR] [PATH_CHECKPOINT] [PLATFORM]
      # example: bash scripts/run_eval_gpu.sh ./data/BSD200 ./data/BSD200_jpeg_quality10 ./train/ckpt/ckpt_0/RedNet30_0-1000_18.ckpt GPU
```

## 转换过程

### 转换

如果您想推断Ascend 310上的网络，则应将模型转换为MINDIR：

```python
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

必须设置ckpt_file参数。
`FILE_FORMAT`取值为["AIR", "MINDIR"]。

## 推理过程

### 在昇腾310上推理

```python
#使用脚本./script/run_infer_310.sh进行推理，最后在run_infer.log文件中查看结果；
bash ./script/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [SAVE_BIN_PATH] [SAVE_OUTPUT_PATH] [DEVICE_ID]
vim run_infer.log
```

## 模型描述

### 训练性能结果

| 参数                        | Ascend                             | GPU                        | Ascend                             |
| --------------------------  | --------------------------------- | -------------------------- |-------------------------- |
| 模型名称                    | RedNet30                           | RedNet30                   | RedNet30                             |
| 运行环境                    | Ascend 910                         | RTX 3090                   | Ascend 310                             |
| 上传时间                    | 2022-03-06                         | 2022-03-06                 | 2022-03-06                             |
| MindSpore 版本              | 1.5.2                              | 1.5.2                      | 1.5.2                            |
| 数据集                      | BSD                                | BSD                         | BSD                             |
| 优化器                      | Adam                               | Adam                        | Adam                             |
| 损失函数                    | MSELoss                            | MSELoss                     | MSELoss                             |
| 精确度 (1p)                 | PSNR[27.51], SSIM[0.7946]          | PSNR[27.35], SSIM[0.7886]   | PSNR[28.67], SSIM[0.8614]                             |
| 训练总时间 (1p)             | 15m11s                             | 19m23s                      | -                             |
| 评估总时间                  | 38s                                | 17s                         | -                             |
| 参数量 (M)                  | 11.8M                              | 11.8M                       | 11.8M                              |



