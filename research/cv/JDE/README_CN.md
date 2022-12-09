## 目录

- [目录](#目录)
- [JDE描述](#jde描述)
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
- [导出过程](#导出过程)
- [Ascend推理过程](#ascend推理过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [验证性能](#验证性能)
- [官方主页](#官方主页)

## [JDE描述](#contents)

JDE模型旨在提高多目标追踪（MOT）系统的性能。

相比SDE（Separate Detection and Embedding）模型和Two-stage模型，JDE采用了共享的神经网络模型实现检测框的目标检测和外观嵌入，将边界框和特征集输入单独的 re-ID 模型中以进行外观特征提取。

JDE方法几乎可以达到实时检测，并且准确率可以与SDE相关方法媲美。

## [模型架构](#contents)

文章采用了特征金字塔网络（FPN）的架构。FPN通过多尺度及逆行预测，从而改善目标规模变化很大的行人检测。输入视频帧首先通过一个骨干网络进行前向传播，获得三种尺度下的特征图，分别是1/32 ，1/16和1/8下采样率的比例。然后，对最小尺度的特征图（也就是语义上最强的特征）进行向上采样，并通过跳过连接将其与第二小尺度的特征图融合，其他尺度也是如此。最后，将预测头添加到所有三个比例尺的融合特征图上。预测头由多层叠加的预测层组成，输出大小为(6𝐴 + 𝐷) × 𝐻 × 𝑊的稠密预测图，其中𝐴为分配给该标度的anchor模板的数量，𝐷确定嵌入的维数。

## [数据集](#contents)

通过将六个关于行人检测，多目标跟踪和行人搜索的公开可用数据集组合在一起，构建了大规模的训练集。这些数据集可分为两种类型，仅包含边界框注释的数据集，以及同时具有边界框和身份注释的数据集。第一类包括ETH数据集和CityPersons数据集。第二类包括CalTech（CT）数据集，MOT-16（M16）数据集，CUHK-SYSU（CS）数据集和PRW 数据集。收集所有这些数据集的训练子集以形成联合训练集，并排除ETH 数据集中与MOT-16测试集重叠的视频以进行公平评估。数据集相关描述在 [DATASET_ZOO.md](DATASET_ZOO.md)。

数据集大小：134G，一种类型（行人）。

注意：`--dataset_root`是所有数据集的入口点，包括训练集和验证集。

数据集的组织形式如下：

```text
.
└─dataset_root/
  ├─Caltech/
  ├─Cityscapes/
  ├─CUHKSYSU/
  ├─ETHZ/
  ├─MOT16/
  ├─MOT17/
  └─PRW/
```

训练集数据统计信息：

| Dataset | ETH |  CP |  CT | M16 |  CS | PRW | Total |
| :------:|:---:|:---:|:---:|:---:|:---:|:---:|:-----:|
| # img   |2K   |3K   |27K  |53K  |11K  |6K   |54K    |
| # box   |17K  |21K  |46K  |112K |55K  |18K  |270K   |
| # ID    |-    |-    |0.6K |0.5K |7K   |0.5K |8.7K   |

## [环境要求](#contents)

- 硬件（GPU/Ascend）
    - 使用GPU/Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [快速入门](#contents)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：
在训练之前，需要根据 `requirements.txt` 执行`pip install -r requirements.txt`安装需要的第三方库。

> 安装过程中出现错误，通过 `pip install --upgrade pip`更新 pip并再次尝试 。
> 选择手动安装软件包，`pip install {package from requirements.txt}`。

注意：PyTorch使用仅仅为了转换ckpt文件。

首先需要下载backbone预训练模型，[download](https://drive.google.com/file/d/1keZwVIfcWmxfTiswzOKUwkUz2xjvTvfm/view) 然后，使用如下命令转换PyTorch预训练模型为MindSpore预训练模型：

```bash
# 进入模型的根目录，运行如下代码
python -m src.convert_checkpoint --ckpt_url [PATH_TO_PYTORCH_CHECKPOINT]
```

- PATH_TO_PYTORCH_CHECKPOINT - PyTorch预训练模型路径

转换预训练模型和安装环境需要的第三方库之后，可以运行训练脚本

注意：根据运行环境修改default_config.yaml文件中is_distributed参数

- GPU处理器环境运行

为了在GPU处理器环境运行，请将配置文件default_config.yaml中的device_target从Ascend改为GPU

```bash
# Run standalone training example
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]

# Run distribute training example
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- Ascend处理器环境运行

```bash
# 运行单机训练示例
bash scripts/run_standalone_train.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]

# 运行分布式训练示例，
bash scripts/run_distribute_train.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT] [RANK_TABLE_FILE]
```

- DEVICE_ID - 设备号
- DEVICE_NUM -设备数量
- LOGS_CKPT_DIR - 训练结果，打印日志，ckpt文件存储的路径
- CKPT_URL - 预训练darknet53骨干
- DATASET_ROOT - 数据集根路径 (包含 [DATASET_ZOO.md](DATASET_ZOO.md)描述的全部数据)
- RANK_TABLE_FILE - JSON格式的hccl配置文件

## [脚本说明](#contents)

### [脚本及样例代码](#contents)

```text
.
└─JDE
  ├─data
  │ └─prepare_mot17.py                 # MOT17数据准备
  ├─model_utils
  │ ├─ccmcpe.json                      # 相关数据集路径(定义了相关的数据集路径结构)
  │ |─config.py                        # 参数配置
  | ├─devide_adapter.py                # 设备适配脚本
  | ├─local_adapter.py                 # 本地设备适配脚本
  | └─moxing_adapter.py                # 分布式设备适配脚本
  ├─scripts
  │ ├─run_eval_gpu.sh                  # GPU评估的shell脚本
  | ├─run_eval.sh                  # Ascend评估的shell脚本
  | ├─run_infer_310.sh                 # Ascend310推理的shell脚本
  │ ├─run_distribute_train_gpu.sh      # 分布式到GPU的shell脚本
  │ ├─run_distribute_train.sh      # 分布式到Ascend的shell脚本
  │ └─run_standalone_train_gpu.sh      # 单卡到GPU的shell脚本
  │ └─run_standalone_train.sh      # 单卡到Ascend的shell脚本
  ├─src
  │ ├─__init__.py
  │ ├─convert_checkpoint.py            # 骨干网络检查点转换脚本 (torch to mindspore)
  │ ├─darknet.py                       # 网络骨干
  │ ├─dataset.py                       # 创建数据集
  │ ├─evaluation.py                    # 评估指标
  │ ├─io.py                            # MOT评估工具脚本
  │ ├─initializer.py                   # 模型参数初始化
  │ ├─kalman_filter.py                 # kalman过滤脚本
  │ ├─log.py                           # 日志脚本
  │ ├─model.py                         # 模型脚本
  │ ├─timer.py                         # 时间脚本
  │ ├─utils.py                         # 工具脚本
  │ └─visualization.py                 # 推理可视化脚本
  ├─tracker
  │ ├─__init__.py
  │ ├─basetrack.py                     # tracking基础类
  │ ├─matching.py                      # matching for tracking 脚本
  │ └─multitracker.py                  # tracker init脚本
  ├─DATASET_ZOO.md                     # 数据集描述
  ├─ascend310_infer                    # 实现310推理源代码
  ├─README.md
  ├─default_config.yaml                # 默认配置
  ├─eval.py                            # 评估脚本
  ├─eval_detect.py                     # 评估检测脚本
  ├─export.py                          # 将检查点文件导出到air/mindir
  ├─preprocess.py                      # 310推理前处理脚本
  ├─postprocess.py                     # 310推理后处理脚本
  ├─infer.py                           # 推理脚本
  ├─requirements.txt
  └─train.py                           # 训练脚本
```

### [脚本参数](#contents)

```text
在config.py中可以同时配置训练参数和评估参数。

--config_path             默认参数配置文件（default_config.yaml）的路径
--data_cfg_url            数据集结构配置文件（.json）的路径
--momentum                动量
--decay                   权重衰减值
--lr                      初始学习率
--epochs                  总计训练epoch数
--batch_size              训练批次大小
--num_classes             目标类别数量
--k_max                   每一次映射的最大预测数（用于优化全连接层嵌入计算）
--img_size                输入图像的大小
--track_buffer            跟踪缓冲大小
--keep_checkpoint_max     最多保存检查点文件的数量
--backbone_input_shape    骨干层的输入过滤器
--backbone_shape          骨干层的输入过滤器
--backbone_layers         骨干层的输出过滤器
--out_channel             检测通道数
--embedding_dim           嵌入通道数
--iou_thres               交并比阈值
--conf_thres              置信度阈值
--nms_thres               非极大值抑制阈值
--min_box_area            最小框面积
--anchor_scales           12个预定义的锚框-3个特征图每一个有4个不同特征映射
--col_names_train         训练数据集的列名
--col_names_val           验证数据集的列名
--is_distributed          是否为分布式训练
--dataset_root            数据集根路径
--device_target           设备类型
--device_id               设备号
--device_start            设备起始号
--ckpt_url                检查点文件
--logs_dir                保存检查点文件，日志的路径
--input_video             输入视频的路径
--output_format           期望的输出格式
--output_root             期望的输出路径
--save_images             保存跟踪结果（图像）
--save_videos             保存跟踪结果（视频）
--file_format:            "MINDIR"
--infer310:               是否进行310推理
```

### [训练过程](#contents)

#### 训练

注意：所有的训练需要使用darknet53预训练模型，根据运行环境修改default_config.yaml文件中is_distributed参数。

- Ascend处理器环境运行

```bash
bash scripts/run_standalone_train.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- GPU处理器环境运行

```bash
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - 设备号
- LOGS_CKPT_DIR - 训练结果，打印日志，ckpt文件存储的路径
- CKPT_URL - 预训练darknet53骨干
- DATASET_ROOT - 数据集根的路径 (包含 [DATASET_ZOO.md](DATASET_ZOO.md)描述的全部数据)

上述python命令将在后台运行，您可以通过standalone_train.log文件查看结果。
训练结束后，您可在logs_dir找到检查点文件。

#### 分布式训练

- Ascend处理器环境运行

```bash
bash scripts/run_distribute_train.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT] [RANK_TABLE_FILE]
```

```text
epoch: 50 step: 1672, loss is -15.7901325
epoch: 50 step: 1672, loss is -17.905373epoch: 50 step: 1672, loss is -18.41980
epoch: 50 step: 1672, loss is -19.16711
epoch: 50 step: 1672, loss is -17.312708
epoch time: 710695.457 ms, per step time: 425.057 ms
epoch time: 710700.617 ms, per step time: 425.060 ms
epoch time: 710695.830 ms, per step time: 425.057 msepoch time: 710700.808 ms, per step time: 425.060 ms
epoch time: 710702.623 ms, per step time: 425.061 ms
epoch time: 710703.826 ms, per step time: 425.062 ms
epoch time: 711144.133 ms, per step time: 425.325 ms
train success
train success
```

- GPU处理器环境运行

```bash
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_NUM - 设备数量
- LOGS_CKPT_DIR - 训练结果，打印日志，ckpt文件存储的路径
- CKPT_URL - 预训练darknet53骨干
- DATASET_ROOT - 数据集根的路径 (包含 [DATASET_ZOO.md](DATASET_ZOO.md)描述的全部数据)
- RANK_TABLE_FILE - JSON格式的hccl配置文件

上述python命令将在后台运行，这里是训练日志的样例：

```text
epoch: 30 step: 1612, loss is -4.7679796
epoch: 30 step: 1612, loss is -5.816874
epoch: 30 step: 1612, loss is -5.302864
epoch: 30 step: 1612, loss is -5.775913
epoch: 30 step: 1612, loss is -4.9537477
epoch: 30 step: 1612, loss is -4.3535285
epoch: 30 step: 1612, loss is -5.0773625
epoch: 30 step: 1612, loss is -4.2019467
epoch time: 2023042.925 ms, per step time: 1209.954 ms
epoch time: 2023069.500 ms, per step time: 1209.970 ms
epoch time: 2023097.331 ms, per step time: 1209.986 ms
epoch time: 2023038.221 ms, per step time: 1209.951 ms
epoch time: 2023098.113 ms, per step time: 1209.987 ms
epoch time: 2023093.300 ms, per step time: 1209.984 ms
epoch time: 2023078.631 ms, per step time: 1209.975 ms
epoch time: 2017509.966 ms, per step time: 1206.645 ms
train success
train success
```

### [评估过程](#contents)

#### 评估

使用MOT16进行评估 (训练过程中不使用).

使用以下命令进行评估

```bash
bash scripts/run_eval[_gpu].sh [DEVICE_ID] [CKPT_URL] [DATASET_ROOT]
```

- DEVICE_ID - 设备号
- CKPT_URL - 训练的JDE模型路径
- DATASET_ROOT - 数据集根的路径 (包含 [DATASET_ZOO.md](DATASET_ZOO.md)描述的全部数据)

> 注意： DATASET_ROOT目录需要包含MOT16子文件夹。
上述python命令将在后台运行，您可以通过eval.log文件查看结果。

- Ascend处理器环境训练后评估结果

```text
DATE-DATE-DATE TIME:TIME:TIME [INFO]: Time elapsed: 323.49 seconds, FPS: 16.39
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN  IDs    FM  MOTA  MOTP IDt IDa IDm
MOT16-02 54.4% 59.6% 50.0% 74.4% 88.7%  54  25  24  5 1698  4557  345   530 63.0% 0.211 149  83   8
MOT16-04 72.4% 79.7% 66.3% 79.2% 95.3%  83  39  33 11 1860  9884  191   424 74.9% 0.210  72  59   3
MOT16-05 70.1% 75.3% 65.6% 79.6% 91.3% 125  64  52  9  518  1394  129   208 70.1% 0.216  80  54  26
MOT16-09 57.7% 63.8% 52.7% 77.6% 94.0%  25  16   8  1  262  1175  130   161 70.2% 0.193  68  34   3
MOT16-10 62.7% 65.7% 60.0% 81.5% 89.2%  54  30  24  0 1219  2284  313   506 69.0% 0.229 136  72   5
MOT16-11 70.7% 72.7% 68.8% 88.8% 93.7%  69  46  21  2  544  1031   82   144 81.9% 0.184  28  33   4
MOT16-13 70.7% 77.2% 65.2% 78.5% 92.9% 107  63  38  6  685  2462  236   539 70.5% 0.219 115  70  32
OVERALL  67.2% 72.9% 62.4% 79.4% 92.8% 517 283 200 34 6786 22787 1426  2512 71.9% 0.210 648 405  81
```

- GPU处理器环境训练后评估结果

```text
DATE-DATE-DATE TIME:TIME:TIME [INFO]: Time elapsed: 240.54 seconds, FPS: 22.04
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN  IDs    FM  MOTA  MOTP IDt IDa IDm
MOT16-02 45.1% 49.9% 41.2% 71.0% 86.0%  54  17  31  6 2068  5172  425   619 57.0% 0.215 239  68  14
MOT16-04 69.5% 75.5% 64.3% 80.6% 94.5%  83  45  24 14 2218  9234  175   383 75.6% 0.184  98  28   3
MOT16-05 63.6% 68.1% 59.7% 82.0% 93.7% 125  67  49  9  376  1226  137   210 74.5% 0.203 113  40  40
MOT16-09 55.2% 60.4% 50.8% 78.1% 92.9%  25  16   8  1  316  1152  108   147 70.0% 0.187  76  15  11
MOT16-10 57.1% 59.9% 54.5% 80.1% 88.1%  54  28  26  0 1337  2446  376   569 66.2% 0.228 202  66  16
MOT16-11 75.0% 76.4% 73.7% 89.6% 92.9%  69  50  16  3  626   953   78   137 81.9% 0.159  49  24  12
MOT16-13 64.8% 69.9% 60.3% 78.5% 90.9% 107  58  43  6  900  2463  272   528 68.3% 0.223 200  59  48
OVERALL  63.2% 68.1% 58.9% 79.5% 91.8% 517 281 197 39 7841 22646 1571  2593 71.0% 0.196 977 300 144
```

使用如下命令验证检测（mAP,Precision and Recall metrics）

```bash
python eval_detect.py --device_id [DEVICE_ID] --ckpt_url [CKPT_URL] --dataset_root [DATASET_ROOT]
```

- DEVICE_ID - 设备号
- CKPT_URL - 训练的JDE模型路径
- DATASET_ROOT - 数据集根的路径 (包含 [DATASET_ZOO.md](DATASET_ZOO.md)描述的全部数据)

- Ascend处理器环境训练验证结果

```text
      Image      Total          P          R        mAP
       4000      30353      0.849      0.782      0.771      0.271s
       8000      30353      0.878      0.796      0.785      0.253s
      12000      30353      0.869      0.814      0.801      0.259s
      16000      30353      0.873      0.823      0.811      0.287s
      20000      30353      0.881      0.833      0.822       0.26s
      24000      30353      0.886      0.842      0.832      0.261s
      28000      30353      0.887      0.838      0.828      0.275s
mean_mAP: 0.8214, mean_R: 0.8316, mean_P: 0.8843
```

- GPU处理器环境训练验证结果

```text
      Image      Total          P          R        mAP
       4000      30353      0.829      0.778      0.765      0.426s
       8000      30353      0.863      0.798      0.788       0.42s
      12000      30353      0.854      0.815      0.802      0.419s
      16000      30353      0.857      0.821      0.809      0.582s
      20000      30353      0.865      0.834      0.824      0.413s
      24000      30353      0.868      0.841      0.832      0.415s
      28000      30353      0.874      0.839       0.83      0.419s
mean_mAP: 0.8225, mean_R: 0.8325, mean_P: 0.8700
```

## [导出过程](#contents)

修改default_config.yaml中的参数，如：ckpt_url、img_size、file_format

```bash
python export.py --ckpt_url [CKPT_URL] --file_format [FILE_FORMAT]
```

- CKPT_URL - 训练的JDE模型路径
- file_format - 从["AIR", "MINDIR"]中选择

## [Ascend推理过程](#contents)

在推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

```bash
# Ascend310 inference
bash scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- MINDIR_PATH - 导出模型的路径
- DATA_PATH - 推理数据集MOT16路径，（PATH TO/MOT16/train）
- DEVICE_ID - 设备号

## [模型描述](#contents)

### [性能](#contents)

#### 训练性能

- Ascend处理器环境运行

| Parameters          | Ascend (8p)                                                     |
| ------------------- | ------------------------------------------------------------ |
| 模型               | JDE (1088*608)                                               |
| 硬件            | Ascend: 8 * Ascend-910(32GB)                                 |
| 更新日期         | 06/08/2022 (day/month/year)                                  |
| MindSpore 版本   | 1.5.0                                                        |
| 数据集             | Joint Dataset (see `DATASET_ZOO.md`)                         |
| 训练参数 | epoch=30, batch_size=4 (per device), lr=0.01, momentum=0.9, weight_decay=0.0001 |
| 优化器           | SGD                                                          |
| 损失函数       | SmoothL1Loss, SoftmaxCrossEntropyWithLogits (and apply auto-balancing loss strategy) |
| 输出             | Tensor of bbox cords, conf, class, emb                       |
| 速度               | Eight cards: ~425 ms/step                                   |
| 全部时间          | Eight cards: ~15 hours                                       |

- GPU处理器环境运行

| Parameters          | GPU (8p)                                                     |
| ------------------- | ------------------------------------------------------------ |
| 模型               | JDE (1088*608)                                               |
| 硬件            | 8 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz                              |
| 更新日期         | 02/02/2022 (day/month/year)                                   |
| MindSpore 版本   | 1.5.0                                                        |
| 数据集             | Joint Dataset (see `DATASET_ZOO.md`)                         |
| 训练参数 | epoch=30, batch_size=4 (per device), lr=0.01, momentum=0.9, weight_decay=0.0001 |
| 优化器           | SGD                                                          |
| 损失函数       | SmoothL1Loss, SoftmaxCrossEntropyWithLogits (and apply auto-balancing loss strategy) |
| 输出             | Tensor of bbox cords, conf, class, emb                       |
| 速度               | Eight cards: ~1206 ms/step                                   |
| 全部时间          | Eight cards: ~17 hours                                       |

#### 验证性能

- Ascend处理器环境运行

| 参数        | NPU (1p)                     |
| ----------------- | ---------------------------- |
| 模型             | JDE (1088*608)               |
| 资源          | 8 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2. |
| 更新日期      | 06/08/2022 (day/month/year)  |
| MindSpore版本      | 1.5.0                        |
| 数据集             | MOT-16                       |
| 批大小             | 1                            |
| 输出               | Metrics, .txt predictions    |
| FPS               | 16.39                        |
| 指标           | mAP 82.14, MOTA 71.9%        |

- GPU处理器环境运行

| 参数        | GPU (1p)                     |
| ----------------- | ---------------------------- |
| 模型             | JDE (1088*608)               |
| 资源          | 1 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| 更新日期      | 02/02/2022 (day/month/year)  |
| MindSpore版本      | 1.5.0                        |
| 数据集             | MOT-16                       |
| 批大小             | 1                            |
| 输出               | Metrics, .txt predictions    |
| FPS               | 22.04                        |
| 指标           | mAP 82.2, MOTA 71.0%         |

## [官方主页](#contents)

 请浏览官网[homepage](https://gitee.com/mindspore/models).