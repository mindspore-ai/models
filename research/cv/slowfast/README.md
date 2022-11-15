# 目录

<!-- TOC -->

- [目录](#目录)
- [slowfast描述](#slowfast描述)
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
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [训练slowfast](#训练slowfast)
        - [评估性能](#评估性能)
            - [评估slowfast](#评估slowfast)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# slowfast描述

slowfast是由Facebook AI研究团队提出的一种新颖的方法来分析视频片段的内容，可以在两个应用最广的视频理解基准测试中获得了当前最好的结果：Kinetics-400和AVA。slowfast在两个数据集上都达到了迄今为止最好的结果，在Kinetics-400上它超过最好top-1得分5.1% (79.0% vs 73.9%) ，超过最好的top-5得分2.7% (93.6% vs 90.9%)。在 Kinetics-600 数据集上它也达到了最好的结果。
在AVA测试中，slowfast研究人员首先使用的版本，是一个较快速R-CNN目标识别算法和现成的行人检测器的整合，利用这个行人检测器获取感兴趣区域。研究人员随后对slowfast网络进行了预训练，最后在ROI上运行网络。结果是28.3 mAP (median average precision) ，比之前的最好结果21.9 mAP有大幅改进。

[论文](https://arxiv.org/abs/1812.03982)：Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He.Submitted on 10 Dec 2018 (v1), last revised 29 Oct 2019 (this version, v3).

# 模型架构

对同一个视频片段应用两个平行的卷积神经网络（CNN）—— 一个慢（Slow）通道，一个快（Fast）通道，最后通过侧向连接将不同层的特征进行融合。

![架构](picture/PatchCore.png)

# 数据集

使用的数据集：[ava2.2](<https://research.google.com/ava/>)

- 数据集大小：477G, 299个视频
    - 训练集：393G，235个视频
    - 测试集：84G，64个视频
- 数据格式：二进制文件
    - 注：数据在datasets/ava_dataset.py中处理。

- 获取数据集：
    - 下载视频
    由于ava2.2与ava2.1视频文件完全相同，我们可以直接使用下面的ava_file_names_trainval_v2.1.txt。

    ```text
    #!/usr/bin/env bash
    set -e
    DATA_DIR="../../../data/ava/videos"
    ANNO_DIR="../../../data/ava/annotations"
    if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating";
    mkdir -p ${DATA_DIR}
    fi
    wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt -P ${ANNO_DIR}
    cat ${ANNO_DIR}/ava_file_names_trainval_v2.1.txt |
    while read vid;
        do wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}" -P ${DATA_DIR}; done
    echo "Downloading finished."
    ```

    - 下载标注文件:[ava_annotations](<https://dl.fbaipublicfiles.com/pyslowfast/annotation/ava/ava_annotations.tar>)
    - 切分视频文件

    ```text
    IN_DATA_DIR="../../../data/ava/videos"
    OUT_DATA_DIR="../../../data/ava/videos_15min"
    if [[ ! -d "${OUT_DATA_DIR}" ]]; then
    echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
    mkdir -p ${OUT_DATA_DIR}
    fi

    for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
    do
    out_name="${OUT_DATA_DIR}/${video##*/}"
    if [ ! -f "${out_name}" ]; then
        ffmpeg -ss 900 -t 901 -i "${video}" -strict experimental "${out_name}"
    fi
    done
    ```

    - 提取RGB帧

    ```text
    IN_DATA_DIR="../../../data/ava/videos_15min"
    OUT_DATA_DIR="../../../data/ava/frames"
    if [[ ! -d "${OUT_DATA_DIR}" ]]; then
    echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
    mkdir -p ${OUT_DATA_DIR}
    fi

    for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
    do
    video_name=${video##*/}

    if [[ $video_name = *".webm" ]]; then
        video_name=${video_name::-5}
    else
        video_name=${video_name::-4}
    fi

    out_video_dir=${OUT_DATA_DIR}/${video_name}/
    mkdir -p "${out_video_dir}"

    out_name="${out_video_dir}/${video_name}_%06d.jpg"

    ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
    done
    ```

    最后将标注文件ava_annotations和上面的帧文件frames放在ava目录下，形成数据集。

    ```text
    ├── ava
    │   ├── ava_annotations
    │   ├── frames
    ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/enable_mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - Ascend：[MindSpore1.5.2](https://www.mindspore.cn/install/en)
    - GPU：[MindSpore1.7.0](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 三方lib，请参见以下文件：
    - pip-requirements.txt

# 快速入门

通过官方网站安装MindSpore后，运行启动命令之前请将相关启动脚本中的路径改为对应机器上的路径，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```text
  # 运行训练示例
  bash scripts/run_standalone_train_ascend.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt

  # 运行分布式训练示例
  bash scripts/run_distribute_train_ascend.sh RANK_TABLE_FILE configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt

  # 运行推理示例
  bash scripts/run_standalone_eval_ascend.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava checkpoint_epoch_00020_best248.pyth.ckpt 1

  # 310离线推理
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
  ```

  对于Ascend分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU处理器环境运行

  ```text
  # 运行训练示例
  bash scripts/run_standalone_train_gpu.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt

  # 运行分布式训练示例
  bash scripts/run_distribute_train_gpu.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt

  # 运行推理示例
  bash scripts/run_standalone_eval_gpu.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt
  ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                           // 所有模型相关说明
    ├── slowfast
        ├── README.md                       // slowfast相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_310_infer.sh      // 310离线推理的shell脚本
        │   ├──run_distribute_train.sh      // Ascend分布式训练的shell脚本
        │   ├──run_distribute_train_gpu.sh  // GPU分布式训练的shell脚本
        │   ├──run_export.sh                // checkpoint文件导出的shell脚本
        │   ├──run_standalone_eval.sh       // Ascend推理的shell脚本
        │   ├──run_standalone_train.sh      // Ascend单卡训练的shell脚本
        │   ├──run_standalone_eval_gpu.sh   // GPU推理的shell脚本
        │   ├──run_standalone_train_gpu.sh  // GPU单卡训练的shell脚本
        ├── src
        │   ├── datasets  // ava数据集处理
        │   ├── models
        │   │   ├──head_helper.py               // ResNe(X)t Head部分处理
        │   │   ├──optimizer.py                 // 优化器
        │   │   ├──resnet_helper.py             // Video models
        │   │   ├──stem_helper.py               // ResNe(X)t 3D stem helper
        │   │   ├──video_model_builder.py       // slowfast模型定义
        │   ├── config
        │   │   ├── custom_config.py            // 模型参数自定义配置
        │   │   ├── defaults.py                 // 模型参数默认配置项
        ├── train.py                        // 训练脚本
        ├── train_modelarts.py              // modelarts训练脚本
        ├── eval.py                         // 评估脚本
        ├── preprogress.py       // 310推理前处理脚本
        ├── postprogress.py       // 310推理后处理脚本
        ├── export.py                       // 将checkpoint文件导出到air/mindir
        ├── pip-requirements.txt                       // lib依赖文件
```

## 脚本参数

在defaults.py中可以同时配置训练参数和评估参数。

- 配置slowfast和ava数据集。

  ```python
  _C.AVA.FRAME_DIR = "/mnt/fair-flash3-east/ava_trainval_frames.img/"    # 视频帧文件路径
  _C.AVA.FRAME_LIST_DIR = ("/mnt/vol/gfsai-flash3-east/ai-group/users/xxx/ava/frame_list/")        # 视频帧列表
  _C.AVA.ANNOTATION_DIR = ("/mnt/vol/gfsai-flash3-east/ai-group/users/xxx/ava/frame_list/")        # 视频帧标注
  _C.AVA.TRAIN_LISTS = ["train.csv"]           # 训练集
  _C.AVA.TEST_LISTS = ["val.csv"] # 推理集
  _C.SOLVER.BASE_LR = 0.1 # 初始学习率
  _C.TRAIN.BATCH_SIZE = 64         # 训练批次大小
  ```

更多配置细节请参考脚本`src/config/defaults.py`。

## 训练过程

### 获取预训练模型

- 下载地址
    <https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl>

- 转换ckpt

  将下载的pkl文件放入项目的根目录下，执行：

  ```python
  python convert_ckpt_caffe2ms.py
  ```

  当前目录下会生成mindspore的ckpt，SLOWFAST_8x8_R50.pkl.ckpt作为训练参数中的CHECKPOINT_FILE_PATH

### 单卡训练

- Ascend处理器环境运行

  ```text
  bash scripts/run_standalone_train_ascend.sh CFG DATA_DIR CHECKPOINT_FILE_PATH
  ```

  如

  ```text
  bash scripts/run_standalone_train_ascend.sh RANK_TABLE_FILE configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt
  ```

  上述python命令将在后台运行，您可以通过log_standalone_ascend文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```text
  # grep "loss is " train.log
  epoch:1 step:390, loss is 1.4842823
  epcoh:2 step:390, loss is 1.0897788
  ...
  ```

- GPU处理器环境运行

  ```text
  bash scripts/run_standalone_train_gpu.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt
  ```

  上述python命令将在后台运行，您可以通过log_standalone_gpu文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```text
  # grep "loss is " train.log
  epoch:1 step:390, loss is 0.0990763
  epcoh:2 step:390, loss is 0.0603111
  ...
  ```

  模型检查点保存在当前目录下。

### 分布式训练

- Ascend处理器环境运行

  ```text
  bash scripts/run_distribute_train ~/hccl_8p_01234567_127.0.0.1.json
  ```

  上述shell脚本将在后台运行分布训练。您可以通过log_distributed_ascend文件查看结果。采用以下方式达到损失值：

  ```text
  # grep "result:" log_distributed_ascend
  train_parallel0/log:epoch:1 step:48, loss is 1.4302931
  train_parallel0/log:epcoh:2 step:48, loss is 1.4023874
  ...
  train_parallel1/log:epoch:1 step:48, loss is 1.3458025
  train_parallel1/log:epcoh:2 step:48, loss is 1.3729336
  ...
  ...
  ```

- GPU处理器环境运行

  ```text
  bash scripts/run_distribute_train_gpu.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt
  ```

  上述shell脚本将在后台运行分布训练。您可以通过log_distributed_gpu文件查看结果。采用以下方式达到损失值：

  ```text
  # grep "result:" log_distributed_gpu
  train_parallel0/log:epoch:1 step:48, loss is 0.2674269
  train_parallel0/log:epcoh:2 step:48, loss is 0.0610401
  ...
  train_parallel1/log:epoch:1 step:48, loss is 0.2730093
  train_parallel1/log:epcoh:2 step:48, loss is 0.0648247
  ...
  ...
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件，ava的配置文件为SLOWFAST_32x2_R50_SHORT.yaml.
其中slowfast-20_3056.ckpt是训练的结果ckpt，在train_parallel7/checkpoints/下。

  ```text
  bash scripts/run_export.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml slowfast-20_3056.ckpt
  ```

导出成功后，当前文件夹会生成slowfast.mindir文件。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

- 在昇腾910上使用ava数据集进行推理

  在运行推理之前我们需要先导出模型。
  在执行下面的命令之前，我们需要先修改ava的配置文件。修改的项包括AVA.FRAME_DIR、AVA.FRAME_LIST_DIR、AVA.ANNOTATION_DIR和TRAIN.CHECKPOINT_FILE_PATH。

  推理的结果保存在当前目录下，在log_eval_ascend日志文件中可以找到类似以下的结果。

  ```text
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/turn (e.g., a screwdriver)': 0.0031881659969238293,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/walk': 0.7207324941463648,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (a person)': 0.6626902737325869,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (e.g., TV)': 0.10220154817817734,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/work on a computer': 0.028072906328370745,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/write': 0.0774830044468495,
  'PascalBoxes_Precision/mAP@0.5IOU': 0.2173776249697695}
  [04/04 14:49:23][INFO] ava_eval_helper.py: 169: AVA eval done in 698.487868 seconds.
  [04/04 14:49:23][INFO] logging.py:  84: json_stats: {"map": 0.21738, "mode": "test"}
  ```

  ```text
  bash scripts/run_standalone_eval_ascend.sh CFG DATA_DIR CHECKPOINT_FILE_PATH DEVICE_ID
  ```

  示例

  ```text
  bash scripts/run_standalone_eval_ascend.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava checkpoint_epoch_00020_best248.pyth.ckpt 1
  ```

- 在GPU上使用ava数据集进行推理

  推理的结果保存在当前目录下，在log_eval_gpu日志文件中可以找到类似以下的结果。

  ```text
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/turn (e.g., a screwdriver)': 0.0031881659969238293,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/walk': 0.7207324941463648,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (a person)': 0.6626902737325869,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (e.g., TV)': 0.10220154817817734,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/work on a computer': 0.028072906328370745,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/write': 0.0774830044468495,
  'PascalBoxes_Precision/mAP@0.5IOU': 0.2173776249697695}
  [04/04 14:49:23][INFO] ava_eval_helper.py: 169: AVA eval done in 698.487868 seconds.
  [04/04 14:49:23][INFO] logging.py:  84: json_stats: {"map": 0.21738, "mode": "test"}
  ```

  ```text
  bash scripts/run_standalone_eval_gpu.sh configs/AVA/SLOWFAST_32x2_R50_SHORT.yaml data/ava SLOWFAST_8x8_R50.pkl.ckpt
  ```

- 在昇腾310上使用ava数据集进行推理
  在推理之前我们需要先通过上述导出步骤导出mindir模型。
  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。
  执行以下命令即可，其中MINDIR_PATH为mindir路径，DATASET为标注文件所在文件夹，NEED_PREPROCESS取值y与n分别表示是否做前处理，
  DEVICE_TARGET为Ascend，DEVICE_ID为卡号。

  ```text
  bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
  ```

  示例

  ```text
  bash run_310_infer.sh /home/zhanglei/slowfast/slowfast.mindir /home/ava n Ascend 0
  ```

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```text
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/touch (an object)': 0.2837172218931491,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/turn (e.g., a screwdriver)': 0.003158146229600418,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/walk': 0.7204151405570671,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (a person)': 0.6627780165255811,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/watch (e.g., TV)': 0.10216062831187808,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/work on a computer': 0.027904712618973603,
  'PascalBoxes_PerformanceByCategory/AP@0.5IOU/write': 0.07717198408534143,
  'PascalBoxes_Precision/mAP@0.5IOU': 0.21723233494686453}
  ```

# 模型描述

## 性能

### 训练性能

#### 训练slowfast

- 使用Ascend

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | Kunpeng-920
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 803G；               |
| MindSpore版本          | 1.5.2                                                       |
| 数据集                    | ava2.2                                                |
| 训练参数        | lr=0.15,fp=32,mmt=0.9,nesterov=false,roiend=1               |
| 优化器                  | Momentum                                                    |
| 损失函数              | BCELoss二分类交叉熵                                       |
| 速度                      | 8卡：476毫秒/步                        |
| 总时长                 | 8卡：8.1小时                                             |

- 使用GPU

| 参数                 | GPU                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | Nvidia
| 资源                   | Nvidia-GeForce RTX 3090；CPU 2.90GHz，64核；内存 251G；               |
| MindSpore版本          | 1.7.0                                                       |
| 数据集                    | AVA2.2                                                |
| 训练参数        | lr=0.15,fp=32,mmt=0.9,nesterov=false,roiend=1               |
| 优化器                  | Momentum                                                    |
| 损失函数              | BCELoss二分类交叉熵                                       |
| 速度                      | 8卡：1500毫秒/步                        |
| 总时长                 | 8卡：30.6小时                                             |

### 评估性能

#### 评估slowfast

- 使用Ascend

| 参数          | Ascend                      |
| ------------------- | --------------------------- |
| 模型版本       | Kunpeng-920               |
| 资源            |  Ascend 910；               |
| MindSpore版本   | 1.5.2                       |
| 数据集             | ava2.2                |
| batch_size          | 8                         |
| 输出             | 概率                 |
| 准确性            | 8卡: 21.73%                |

- 使用GPU

| 参数          | GPU                      |
| ------------------- | --------------------------- |
| 模型版本       | Nvidia               |
| 资源            |  Nvidia-GeForce RTX 3090；               |
| MindSpore版本   | 1.7.0                       |
| 数据集             | AVA2.2                |
| batch_size          | 16                         |
| 输出             | 概率                 |
| 准确性            | 8卡: 21.73%                |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
