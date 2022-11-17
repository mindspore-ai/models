# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [ECAPA-TDNN描述](#ECAPA-TDNN描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [生成训练和测试数据](#生成训练和测试数据)
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
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# ECAPA-TDNN描述

ECAPA-TDNN是2020年提出的深度网络，在voxceleb测试集中获得了目前最好的结果。迅速在声纹识别领域产生了较大影响。相比传统的tdnn网络，ECAPA-TDNN增加了SE-block + Res2Block + Attentive Stat Pooling。并增加了channel数，在模型增大的同时，性能也得到了极大的提升。
[论文](https://arxiv.org/abs/2005.07143)：Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck. Interspeech 2020.

# 模型架构

ECAPA-TDNN由多个SE-Res2Block模块串联起来，可以更加深入。SE-Res2Block的基本卷积单元使用和传统tdnn模块相同的1d卷积和dilation参数。模块一般包括**1×1卷积**、**3×1卷积**和**SE-block**以及**res2net**结构。

# 数据集

## 使用的数据集：[voxceleb](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/>)

- 数据集大小：7,000+人，超过一百万条语音，总时长2000+
    - [Voxceleb1](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html>) 10万+ 语音，1251人
        - 训练集：1211人
        - 测试集：40人
    - [Voxceleb2](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>) 包含100万+语音，6112人
        - 训练集：5994人
        - 测试集：118人
- 数据格式：voxceleb1为wav，voxceleb2为m4a
      - 注：训练数据需要全部转成wav格式
- 准备数据：voxceleb2语音文件为m4a格式，需要转成wav格式才能加入训练。请按照以下流程准备数据：
    - 下载voxceleb1和voxceleb2数据。
    - 转换vox2数据格式，脚本可参考：https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830
    - 把voxceleb1和voxceleb2的训练集目录的所有语音文件放在 wav目录下构成训练数据集。路径参考 voxceleb12/wav/id*/*.wav
    - 训练数据集目录结构如下：

      ``` bash
      voxceleb12
      ├── meta  # 需自行默认创建
      └── wav
          ├── id10001
          │   ├── 1zcIwhmdeo4
          │   ├── 7gWzIy6yIIk
          │   └── ...
          ├── id10002
          │   ├── 0_laIeN-Q44
          │   ├── 6WO410QOeuo
          │   └── ...
          ├── ...
      ```

    - 使用voxceleb1作为测试数据集，拷贝[测试使用的trials文件](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt) 到voxceleb1/veri_test2.txt
    - 测试集目录结构如下：

      ``` bash
      voxceleb1
      ├── veri_test2.txt
      └── wav
          ├── id10001
          │   ├── 1zcIwhmdeo4
          │   ├── 7gWzIy6yIIk
          │   └── ...
          ├── id10002
          │   ├── 0_laIeN-Q44
          │   ├── 6WO410QOeuo
          │   └── ...
          ├── ...
      ```

## 生成训练和测试数据

由于mindspore目前不支持在线提特征，所以我们要先对训练语音做数据增广并提取出fbank特征，作为mindspore训练的输入数据。此处生成脚本源自speechbrain库，做了一些简化修改

- 首先运行脚本 data_prepare.sh, 即可生成训练用的数据，五倍增广需要消耗数小时，空间占用大约1.3T，要达到目标精度需要50倍增广，需要占用大约13T空间。

  ``` bash
  bash data_prepare.sh
  ```

- 再运行运行以下脚本， 生成npy格式文件，加速数据读取

  ``` bash
  python3 merge_data.py hparams/prepare_train.yaml
  ```

- 注：参数配置参考快速入门

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - python3及其它依赖安装包
        - 安装完python3后，执行命令 `pip3 install -r requirements.txt``
        - 其中speechbrain库主要用来做数据增广及从wav语音中提取fbank特征。
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

配置完环境后，您可以按照如下步骤进行训练和评估：

  ```text
  # 在prepare_train.yaml中修改数据路径
  data_folder: /home/abc000/data/voxceleb12  # 训练数据集路径
  feat_folder: /home/abc000/data/feat_train/ # 训练特征存储路径

  # 在prepare_eval.yaml中修改数据路径
  data_folder: /home/abc000/data/voxceleb1/ # 测试数据集路径
  feat_eval_folder: /home/abc000/data/feat_eval/ # 测试集特征存储路径
  feat_norm_folder:  /home/abc000/data/feat_norm/ # 做norm的数据集特征存储路径

  # 在edapa-tdnn_config.yaml文件中修改数据路径
  train_data_path: /home/abc000/data/feat_train/
  ```

  ```bash
  # 运行训练示例
  python3 train.py > train.log 2>&1 &

  # 使用脚本的单卡训练
  bash run_standalone_train_ascend.sh DEVICE_ID

  # 使用脚本的分布式训练
  bash run_distribute_train_ascend.sh RANK_TABLE_FILE

  # 运行推理示例
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

# 脚本说明

## 脚本及样例代码

```bash
    ModelZoo_ECAPA-TDNN
    ├── ascend310_infer                              # 310 推理代码
    ├── data_prepare.sh                              # 准备mindspore训练和测试数据
    ├── ecapa-tdnn_config.yaml                       # 训练和测试等相关参数
    ├── eval_data_prepare.py                         # 准备测试集的脚本
    ├── eval.py                                      # 测试脚本
    ├── export.py                                    # 转出310模型脚本
    ├── hparams                                      # 准备mindspore训练和测试数据所需要的参数
    ├── README_CN.md                                 # ecapa-tdnn相关说明
    ├── README.md                                    # ecapa-tdnn相关说明
    ├── requirements.txt                             # python依赖包
    ├── scripts                                      # 训练和测试相关的shell脚本
    ├── src                                          # mindspore模型相关代码
    ├── train_data_prepare.py                        # 准备训练集的脚本
    ├── merge_data.py                                # 把零散数据文件合并为少量大文件，加速数据读取
    └── train.py                                     # 训练脚本

```

## 脚本参数

在hparams/prepare_train.yaml中可以配置训练特征提取参数

```text
  output_folder: ./augmented/                             # 中间结果存储路径
  save_folder: !ref <output_folder>/save/                 # 中间结果存储路径
  feat_folder: /home/abc000/data/feat_train/              # 训练特征存储路径
  data_folder: /home/abc000/data/voxceleb12               # 训练数据集路径
  train_annotation: !ref <save_folder>/train.csv          # 预先生成的csv文件， 若没有会重新生成
  valid_annotation: !ref <save_folder>/dev.csv            # 预先生成的csv文件， 若没有会重新生成
```

在hparams/prepare_eval.yaml中可以配置测试特征提取参数

```text
  output_folder: ./augmented_eval/                        # 中间结果存储路径
  feat_eval_folder: /home/abc000/data/feat_eval/          # 测试集特征存储路径
  feat_norm_folder:  /home/abc000/data/feat_norm/         # 做norm的训练集特征存储路径
  data_folder: /home/abc000/data/voxceleb1/               # 使用voxceleb1数据集做测试
  save_folder: !ref <output_folder>/save/
```

在edapa-tdnn_config.py中可以同时配置训练参数和评估参数。

- 配置ECAPA-TDNN和数据集

  ```text
    inChannels: 80                                                  # 输入层特征通道数，即fbank特征的维度
    channels: 1024                                                  # 中间层特征图的通道数
    base_lrate: 0.000001                                            # cyclic LR学习策略的基础学习率
    max_lrate: 0.0001                                               # cyclic LR学习策略的最大学习率
    momentum: 0.95                                                  # 优化器参数
    weightDecay: 0.000002                                           # 优化器参数
    num_epochs: 3                                                   # 训练周期数
    minibatch_size: 192                                             # batch size
    emb_size: 192                                                   # embedding 维度
    step_size: 65000                                                # cyclic LR学习策略达到最大学习率的步数
    CLASS_NUM: 7205                                                 # voxceleb1&2 的说话人个数
    pre_trained: False                                              # 是否载入预训练模型
    train_data_path: "/home/abc000/data/feat_train/"                # 训练数据目录
    keep_checkpoint_max: 30                                         # 最大保存模型数
    checkpoint_path: "/ckpt/train_ecapa_vox2_full-2_664204.ckpt"    # 预训练模型目录
    ckpt_save_dir: "./ckpt/"  # 模型训练的输出目录
    # eval
    eval_data_path: "/home/abc000/data/feat_eval/"                  # 测试语音目录
    veri_file_path: "veri_test_bleeched.txt"                        # 测试对列表
    model_path: "ckpt/train_ecapa_vox2_full-2_664204.ckpt"          # 测试模型目录
    "score_norm": "s-norm"                                          # 是否做norm
    train_norm_path: "/data/dataset/feat_norm/"                     # 用来做norm的数据

  ```

更多配置细节请参考脚本`edapa-tdnn_config.yaml`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python3 train.py > train.log 2>&1 &  
  或者 ./scripts/run_standalone_train_ascend.sh
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss: " train.log
  2022-02-13 13:58:33.898547, epoch:0/15, iter-719000/731560, aver loss:1.5836, cur loss:1.1664, acc_aver:0.7349
  2022-02-13 14:08:44.639722, epoch:0/15, iter-720000/731560, aver loss:1.5797, cur loss:1.1057, acc_aver:0.7363
    ...
  ```

  模型检查点保存在指定目录下。

### 分布式训练

  ```bash
  bash scripts/run_distribute_train_ascend.sh
  ```

  上述shell脚本将在后台运行分布训练。您可以通过log文件查看结果。损失值如下所示：

  ```bash
  # grep "loss: " train.log
  2022-02-13 13:58:33.898547, epoch:0/15, iter-719000/731560, aver loss:1.5836, cur loss:1.1664, acc_aver:0.7349
  2022-02-13 14:08:44.639722, epoch:0/15, iter-720000/731560, aver loss:1.5797, cur loss:1.1057, acc_aver:0.7363
  ...
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估voxceleb1数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/ECAPA-TDNN/train_ECAPA-TDNN_vox12-125_390.ckpt”。
  由于测试集的语音长度不固定，所以mindspore推理速度较慢，需要数小时才能完成计算

  ```bash
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "eer" eval.log
  eer with norm:0.0082
  ```

  我们还可以通过设置ecapa-tdnn_config.yaml配置文件中的cut_wav参数来得到固定3s语音的eer精度，此结果与310推理结果一致。

  ```text
  wav_cut: true                                              # 是否把测试语音截断至3s（与训练语音长度一致），默认为false，即完整语音
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件，配置文件为edapa-tdnn_config.yaml.
需要修改的配置项为 exp_ckpt_file.

```bash
python3 export.py
```

## 推理过程

### 推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用voxceleb1数据集进行推理

  首先设置ecapa-tdnn_config.yaml配置文件中的veri_file_path参数，veri_test_bleeched.txt文件存放在之前生成的测试集feat_eval目录下。由于310只支持固定长度推理，所以我们的310推理代码中会把测试语音截断至3s，整体结果要差于完整语音的推理结果，完整语音的推理结果可参考上文910评估结果。推理的结果保存在当前目录下，在infer.log日志文件中可以找到类似以下的结果。

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  # example: bash run_infer_310.sh /path/ecapatdnn.mindir /path/feat_eval 0
  cat acc.log | grep eer
  eer sub mean: 0.0248
  ```

# 模型描述

## 性能

### 训练性能

| 参数                       | Ascend
| -------------------------- | -----------------------------------------------------------
| 模型版本                   | ECAPA-TDNN
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8
| 上传日期              | 2022-02-26
| MindSpore版本          | 1.5.1
| 数据集                    | voxceleb1&voxceleb2
| 训练参数        | epoch=2, steps=733560*epoch, batch_size = 192, min_lr=0.000001, max_lr=0.0001
| 优化器                  | Adam
| 损失函数              | AAM-Softmax交叉熵
| 输出                    | 概率
| 损失                       | 1.3
| 速度                      | 单卡：565步/秒(fps:339)
| 总时长                 | 单卡：264小时
| 参数(M)             | 13.0
| 微调检查点 | 254M (.ckpt文件)
| 推理模型        |  76.60M(.mindir文件)

### 评估性能

#### voxceleb1上评估ECAPA-TDNN

| 参数          | Ascend
| ------------------- | ---------------------------
| 模型版本       | ECAPA-TDNN
| 资源            |  Ascend 910；系统 Euler2.8
| 上传日期       | 2021-07-05
| MindSpore 版本   | 1.3.0
| 数据集             | voxceleb1-eval, 4715条语音
| batch_size          | 1
| 输出             | 概率
| 准确性            | 单卡: EER=0.82%;  
| 推理模型 | 76.60M(.mindir文件)        |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
