# 目录

<!-- TOC -->

- [目录](#目录)
- [DeepSort描述](#DeepSort描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
    - [推理过程](#推理过程)
        - [用法](#用法)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练表现](#训练表现)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## DeepSort描述

DeepSort是2017年提出的多目标跟踪算方法。该网络在MOT16获得冠军，不仅提升了精度，而且速度比之前快20倍。

[论文](https://arxiv.org/abs/1602.00763)： Nicolai Wojke, Alex Bewley, Dietrich Paulus. "SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC". *Presented at ICIP 2016*.

## 模型架构

DeepSort由一个特征提取器、一个卡尔曼滤波和一个匈牙利算法组成。特征提取器用于提取框中人物特征信息，卡尔曼滤波根据上一帧信息预测当前帧人物位置，匈牙利算法用于匹配预测信息与检测到的人物位置信息。

## 数据集

使用的数据集：[MOT16](<https://motchallenge.net/data/MOT16.zip>)、[Market-1501](<https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view>)

MOT16:

- 数据集大小：1.9G，共14个视频帧序列
    - test：7个视频序列帧
    - train：7个序列帧
- 数据格式(一个train视频帧序列)：
    - det:视频序列中人物坐标以及置信度等信息
    - gt:视频跟踪标签信息
    - img1:视频中所有帧序列
    - 注意：由于作者提供的视频帧序列检测到的坐标信息和置信度信息不一样，所以在跟踪时使用作者提供的信息，作者提供的[npy](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)文件。

Market-1501:

- 使用：
    - 使用目的：训练DeepSort特征提取器
    - 使用方法： 先使用prepare.py处理数据

## 环境要求

- 硬件（Ascend/ModelArts）
    - 准备Ascend或ModelArts处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 在 GPU 上运行

```bash
# 从脚本中的路径中提取检测信息
python process-npy.py
# 通过脚本中的路径预处理 Market-1501
python prepare.py
# 在单个 GPU 上训练 DeepSort 特征提取器
bash run_standalone_train_gpu.sh DATA_PATH
# 生成特征信息
python generater-detection.py --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
# 生成结果
python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
# 使用产生 MOT16 挑战的指标 https://github.com/cheind/py-motmetrics
python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
```

- 在 Ascend 上运行

```bash
# 从脚本中的路径中提取检测信息
python process-npy.py
# 通过脚本中的路径预处理 Market-1501
python prepare.py
# 在 Ascend 上训练 DeepSort 特征提取器
python src/deep/train.py --run_modelarts=False --run_distribute=True --data_url="" --train_url=""
# 或者
bash run_distribute_train_ascend.sh [DATA_PATH] [CKPT_PATH]
# 生成特征信息
python generater_detection.py --run_modelarts=False --run_distribute=True --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
# 生成结果
python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
# 使用产生 MOT16 挑战的指标 https://github.com/cheind/py-motmetrics
python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```text
├── DeepSort
    ├── ascend310_infer
    ├── infer
    ├── modelarts
    ├── scripts #scripts for training
    │   ├──run_standalone_train_gpu.sh
    │   ├──run_distributed_train_ascend.sh
    │   ├──run_infer_310.sh
    │   ├──docker_start.sh
    ├── src
    │   │   ├── application_util
    │   │   │   ├──image_viewer.py
    │   │   │   ├──preprocessing.py
    │   │   │   ├──visualization.py
    │   │   ├──deep #features extractor code
    │   │   │   ├──feature_extractor.py
    │   │   │   ├──config.py
    │   │   │   ├──market1501_standalone_gpu.yaml #parameters for 1P GPU training
    │   │   │   ├──original_model.py
    │   │   │   ├──train.py
    │   │   ├──sort
    │   │   │   ├──detection.py
    │   │   │   ├──iou_matching.py
    │   │   │   ├──kalman_filter.py
    │   │   │   ├──linear_assignment.py
    │   │   │   ├──nn_matching.py
    │   │   │   ├──track.py
    │   │   │   ├──tracker.py
    ├── deep_sort_app.py #auxiliary module
    ├── Dockerfile
    ├── evaluate_motchallenge.py #script for generating tracking result
    ├── export.py
    ├── generate_videos.py
    ├── generater-detection.py #script for generating features information
    ├── preprocess.py #auxiliary module
    ├── prepare.py #script to prepare market-1501 dataset
    ├── process-npy.py #script to extract and prepare MOT detections
    ├── show_results.py
    ├── pipeline.sh #example of calling all scripts in a sequence
    ├── README.md
```

### 脚本参数

```text
generater_detection.py evaluate_motchallenge.py:

--data_url: path to dataset (MOT / Market)
--train_url: output path
--ckpt_url: path to checkpoint
--model_name: name of the checkpoint
--det_url: path to detection files
--detection_url:  path to features files
```

### 训练过程

#### 训练

- Ascend 处理器环境运行

  ```bash
  bash scripts/run_distributed_train_ascend.sh train_code_path RANK_TABLE_FILE DATA_PATH
  ```

  经过训练后，损失值如下：

  ```bash
  # grep "loss is " log
  epoch: 1 step: 3984, loss is 6.4320717
  epoch: 1 step: 3984, loss is 6.414733
  epoch: 1 step: 3984, loss is 6.4306755
  epoch: 1 step: 3984, loss is 6.4387856
  epoch: 1 step: 3984, loss is 6.463995
  ...
  epoch: 2 step: 3984, loss is 6.436552
  epoch: 2 step: 3984, loss is 6.408932
  epoch: 2 step: 3984, loss is 6.4517527
  epoch: 2 step: 3984, loss is 6.448922
  epoch: 2 step: 3984, loss is 6.4611588
  ...
  ```

- GPU 处理器环境运行

  ```bash
  #standalone
  bash run_standalone_train_gpu.sh DATA_PATH
  ```

  经过训练后，损失值如下：

  ```bash
  epoch: 1 step: 809, loss is 4.4773345
  epoch time: 14821.373 ms, per step time: 18.321 ms
  epoch: 2 step: 809, loss is 3.3706033
  epoch time: 9110.971 ms, per step time: 11.262 ms
  epoch: 3 step: 809, loss is 3.000544
  epoch time: 9131.733 ms, per step time: 11.288 ms
  epoch: 4 step: 809, loss is 1.196707
  epoch time: 8973.570 ms, per step time: 11.092 ms
  epoch: 5 step: 809, loss is 1.0504937
  epoch time: 9051.383 ms, per step time: 11.188 ms
  epoch: 6 step: 809, loss is 0.7604818
  epoch time: 9384.670 ms, per step time: 11.600 ms
  ...
  ```

  模型检查点保存在当前目录下。

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GPU 处理器环境运行

```bash
  python generater-detection.py --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name=""
  python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
  python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
```

- Ascend 处理器环境运行

```bash
  python generater-detection.py --data_url="" --train_url="" --det_url="" --ckpt_url="" --model_name="" --device="Ascend"
  python evaluate_motchallenge.py --data_url="" --train_url="" --detection_url=""
  python -m motmetrics.apps.eval_motchallenge <groundtruths> <tests>
```

-
  测试数据集的准确率如下 (GPU)

| Seq | MOTA | MOTP| MT | ML| IDs | FM | FP | FN |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -----------------------------------------------------------
| MOT16-02 | 29.0% | 0.207 | 12 | 10| 167 | 247 | 4212 | 8285 |
| MOT16-04 | 58.7% | 0.168| 42 | 15| 58 | 254 | 6268 | 13328 |
| MOT16-05 | 51.9% | 0.215| 31 | 27| 62 | 112 | 643 | 2577 |
| MOT16-09 | 64.4% | 0.162| 13 | 1| 42 | 57 | 313 | 1519 |
| MOT16-10 | 48.7% | 0.228| 24 | 1| 220 | 301 | 3183 | 2922 |
| MOT16-11 | 65.3% | 0.153| 29 | 9| 57 | 95 | 927 | 2195 |
| MOT16-13 | 44.3% | 0.237| 62 | 6| 328 | 332 | 3784 | 2264 |
| overall | 51.7% | 0.190| 211 | 69| 934 | 1398 | 19330 | 33090 |

-
  测试数据集的准确率如下 (Ascend)

| Seq | MOTA | MOTP| MT | ML| IDs | FM | FP | FN |
| -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -----------------------------------------------------------
| MOT16-02 | 29.0% | 0.207 | 11 | 11| 159 | 226 | 4151 | 8346 |
| MOT16-04 | 58.6% | 0.167| 42 | 14| 62 | 242 | 6269 | 13374 |
| MOT16-05 | 51.7% | 0.213| 31 | 27| 68 | 109 | 630 | 2595 |
| MOT16-09 | 64.3% | 0.162| 12 | 1| 39 | 58 | 309 | 1537 |
| MOT16-10 | 49.2% | 0.228| 25 | 1| 201 | 307 | 3089 | 2915 |
| MOT16-11 | 65.9% | 0.152| 29 | 9| 54 | 99 | 907 | 2162 |
| MOT16-13 | 45.0% | 0.237| 61 | 7| 269 | 335 | 3709 | 2251 |
| overall | 51.9% | 0.189| 211 | 70| 852 | 1376 | 19094 | 33190 |

## [导出mindir模型](#contents)

```shell
python export.py --device_id [DEVICE_ID] --ckpt_file [CKPT_PATH]
```

## [推理过程](#contents)

### 用法

执行推断之前，minirir文件必须由export.py导出。输入文件必须为bin格式

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

### 结果

推理结果文件保存在当前路径中，将文件作为输入，输入到eval_motchallenge.py中，然后输出result文件，输入到测评工具中即可得到精度结果。

## 模型描述

### 性能

#### 训练表现

| 参数 | | |
| -------------------------- | -----------------------------------------------------------|-------------------------------------------------------|
| 资源 | GPU Tesla V100-PCIE 32G| Ascend 910 CPU 2.60GHz, 192 cores：755G |
| 上传日期 | 2022-01-08 | 2021-08-12 |
| MindSpore版本 | 1.6.0 |  1.2.0 |
| 数据集 | MOT16 Market-1501 | MOT16 Market-1501 |
| 训练参数 | epoch=24, batch_size=16, lr=0.01 | epoch=100, step=191, batch_size=8, lr=0.1 |
| 优化器 | SGD |
| 损失函数 | SoftmaxCrossEntropyWithLogits | SoftmaxCrossEntropyWithLogits |
| 损失 | 0.04 | 0.03 |
| 速度 | 12.8 ms/step | 12.4 ms/step |
| 总时间 | 3 min | 10 min |
| 微调检查点 | 23.4 Mb | 40 Mb |

#### 评估性能

| 参数          || |
| ------------------- | --------------------------- | --- |
| 资源 | GPU Tesla V100-PCIE 32G| Ascend 910 CPU 2.60GHz, 192 cores：755G |
| MindSpore版本 | 1.6.0 |  1.2.0 |
| 数据集 | MOT16 Market-1501 | MOT16 Market-1501 |
| MOTA/MOTP | 51.7%/0.190                | 51.9%/0.189 |
| 微调检查点 | 23.4 Mb | 40 Mb |

## 随机情况说明

train.py中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。