# 目录

<!-- TOC -->

- [目录](#目录)
- [AlignedReID++描述](#AlignedReID++描述)
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
        - [单卡训练](#单卡训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
    - [使用流程](#使用流程)
        - [推理](#推理-1)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# AlignedReID++描述

AlignedReID++基本继承了前作AlignedReID的核心，内容更加完善了，还有了轻微的改动，实验做的也有些变化。文章本身很易读，相比AlignedReID讲述的也更清楚，可以作为AlignedReID的补充阅读。

[论文](https://www.sciencedirect.com/science/article/pii/S0031320319302031)：Luo, Hao, Jiang, Wei, Zhang, Xuan, Fan, Xing, Qian, Jingjing and Zhang, Chi. "AlignedReID++: Dynamically matching local information for person re-identification.." *Pattern Recognit.* 94 (2019): 53-61.

# 模型架构

AlignedReID++采用resnet50作为backbone，重新命名了AlignedReID中提出的切片对齐方法——动态匹配局部信息(Dynamically Matching Local Information (DMLI))，它能够在不引入附加监督的情况下自动对齐切片信息，以解决由bounding box error，遮挡，视角偏差，姿态偏差等带来的行人不对齐问题。AlignedReID++通过在local feature引入DMLI，结合global feature和local feature的多细粒度，结合triplet hard loss和ID loss的多种loss学习，达到更好的行人重识别准确率。

# 数据集

使用的数据集：market1501

- 数据集大小：共1501个类、由6个摄像头拍摄到的 1501 个行人、32668 个检测到的行人矩形框。
    - 训练集
        - bounding_box_train: 751 人，包含 12,936 张图像。
    - 测试集
        - query: 3368 张查询图像的行人检测矩形框是人工绘制的。
        - gallery:  750 人，包含 19,732 张图像。其行人检测矩形框是使用DPM检测器检测得到的。
- 数据格式：jpg
    - 注：数据将在data_manager.py 和dataset_loader.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

预训练模型resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt可点击此处下载[链接](https://www.mindspore.cn/resources/hub/details?MindSpore/ascend/1.3/resnet50_v1.3_imagenet2012)

- Ascend处理器环境运行

  ```python
  # 运行分布式训练示例
  # 8卡运行时需提前写好rank_table_8pcs.json文件（文件名可修改），放入scripts文件下
  bash scripts/run_train_ascend.sh [DATA_URL] [PRE_TRAINED] [RANK_TABLE_8PCS_FILE]
  example: bash scripts/run_train_ascend.sh /dataset resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt /scripts/rank_table_8pcs.json
  # 运行单卡训练示例
  bash scripts/run_singletrain_ascend.sh [DATA_URL] [PRE_TRAINED] [DEVICE_ID]
  或
  python train.py --is_distributed=False --data_url=[DATA_URL] --pre_trained=[PRE_TRAINED] --device_id=0
  # 运行评估示例
  python test.py --data_url=[DATA_PATH] --checkpoint_path=[CHECKPOINT_PATH] --device_id=0 > test.log 2>&1 &
  或
  bash scripts/run_test_ascend.sh [data_url] [checkpoint_path] [device_id]
  example: bash scripts/run_test_ascend.sh /dataset resnet50-300_23.ckpt 0
  # 导出air或者mindir模型
  python export.py --ckpt_file=[ckpt_file] --device_id=0
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件，例如上面的rank_table_8pcs.json。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡或单卡训练 market1501 数据集

      ```python
      # (1) 新建一个aligned桶，将代码上传到 aligned 桶的code文件夹下
      # (2) 在网页上设置 "run_modelarts=True"
      # (3) 在网页上设置 "data_url=数据集所在目录"（如果是market1501数据集，该路径则为market1501文件夹的上一层）
      #     例如数据集所在路径/aligned/data/market1501，那么data_url=/aligned/data/
      # (4) 在网页上设置 "pre_trained=预训练模型resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt所在目录（该路径为预训练模型ckpt所在路径）"
      #     例如预训练模型所在路径/aligned/premodels/resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt，那么pre_trained=/aligned/premodels/
      # (5) 在网页上设置 "train_url=/aligned/output/"
      #     在网页上设置 其他参数
      # (6) 上传你的数据集到 data文件夹下 (整个market1501文件)
      # (7) 在网页上设置你的代码路径为 "/aligned/code"
      # (8) 在网页上设置启动文件为 "train.py"
      # (9) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (10) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证 market1501 数据集

      ```python
      # (1) 新建一个aligned桶，将代码上传到 aligned 桶的code文件夹下
      # (2) 在网页上设置 "run_modelarts=True"
      # (3) 在网页上设置 "data_url=数据集所在目录"（如果是market1501数据集，该路径则为market1501文件夹的上一层）
      #     例如数据集所在路径/aligned/data/market1501，那么data_url=/aligned/data/
      # (4) 在网页上设置 "checkpoint_path=待评估模型文件所在目录（该路径为待评估模型ckpt所在路径）"
      #     例如待评估模型所在路径/aligned/code/testmodels/resnet50-300_23.ckpt, 那么 checkpoint_path=/aligned/code/testmodels/
      # (5) 在网页上设置 "modelarts_ckpt_name='resnet50-300_23.ckpt'"(不包含路径，只是ckpt文件名字)
      # (6) 在网页上设置 "train_url=/aligned/output/"
      #     在网页上设置 其他参数
      # (7) 上传你的数据集到 data文件夹下 (整个market1501文件)
      # (8) 在网页上设置你的代码路径为 "/aligned/code"
      # (9) 在网页上设置启动文件为 "test.py"
      # (10) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (11) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── aligned
        ├── README.md                    // aligned相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_singletrain_ascend.sh // Ascend单卡运行的shell脚本
        │   ├──run_train_ascend.sh       // 分布式到Ascend的shell脚本
        │   ├──run_test_ascend.sh        // Ascend评估的shell脚本
        │   ├──run_infer_310.sh          // 310推理脚本
        ├── src
        │   ├──__init__.py
        │   ├──data_manager.py              // 数据加载
        │   ├──dataset_loader.py            // 数据预处理和数据读取
        │   ├──distance.py                  // 计算图片间距离的距离函数
        │   ├──eval_metrics.py              // 评估函数
        │   ├──losses2.py                   // 损失函数
        │   ├──lr_scheduler.py              // 生成每个步骤的学习率
        │   ├──re_ranking.py                // 行人重排序
        │   ├──ResNet.py                    // 网络模型
        │   ├──samplers.py                  // 数据采样器
        │   ├──utils.py                     // 日志定义
        ├── train.py              // 训练脚本
        ├── test.py               // 评估脚本
        ├── export.py             // 将checkpoint文件导出到air/mindir
        ├── postprocess.py        // 310推理后处理脚本
```

## 脚本参数

在train.py中可以配置训练参数。

- 配置market1501数据集。

  ```python
  'is_distributed':True    # 是否分布式训练，默认是
  'run_modelarts':False    # 是否在modelarts上进行训练，默认否
  'device_id':0            # 单卡训练时用于训练数据集的设备ID
  'lr_init':0.             # 初始学习率
  'optim':'momentum'       # 优化器
  'lr_decay_mode':'cosine' # 学习衰减策略
  'lr_max':5e-2            # 最大学习率
  'warmup_epochs':6        # warm up 的epoch数
  'max_epoch':300          # 总计训练epoch数
  'data_url':''            # 数据集路径
  'pre_trained':''         # 预训练模型所在路径
  'train_url':''           # 输出路径
  'num_instances':8        # 训练时对于每个人所抽取图片数量
  'height':256             # 输入到模型的图像高度
  'width':128              # 输入到模型的图像宽度
  'labelsmooth':'True'     # 是否采用标签平滑
  'weight-decay':5e-04     # 权重衰减值
  ```

在test.py中可以配置评估参数。

- 配置market1501数据集。

  ```python
  'run_modelarts':False                         # 是否在modelarts上进行训练，默认否
  'data_url':''                                 # 数据集路径
  'checkpoint_path':''                          # 待评估模型所在路径
  'modelarts_ckpt_name':resnet50-300_23.ckpt    # 待评估模型名称（modelarts运行时使用）
  'height':256                                  # 输入图片的高
  'width':128                                   # 输入图片的宽
  'reranking':'True'                            # 行人重排列
  'device_id':0                                 # 用于评估数据集的设备ID
  ```

更多配置细节请参考脚本`train.py`和`test.py`。

## 训练过程

### 单卡训练

- Ascend处理器环境运行

  ```python
  python train.py --is_distributed=False --data_url=[DATA_URL] --pre_trained=[PRE_TRAINED] --device_id=[DEVICE_ID] > train.log 2>&1 &
  OR
  bash scripts/run_singlrtrain_ascend.sh [DATA_URL] [PRE_TRAINED] [DEVICE_ID]
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```python
  # grep "loss is " train.log
  epoch: 1 step: 93, loss is 9.789021
  epoch: 2 step: 93, loss is 8.736438
  epoch: 3 step: 93, loss is 8.006744
  ...
  epoch: 299 step: 93, loss is 1.1644284
  epoch: 300 step: 93, loss is 1.2903279
  ```

  模型检查点保存在args.train_url下。

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在您在train.py文件中配置的train_url路径下找到模型文件。

### 分布式训练

- Ascend处理器环境运行

  ```python
  bash scripts/run_train_ascend.sh [DATA_URL] [PRE_TRAINED] [RANK_TABLE_8PCS_FILE]
  example: bash scripts/run_train_ascend.sh /dataset_path resnet50_ascend_v130_imagenet2012_official_cv_bs256_top1acc76.97__top5acc_93.44.ckpt /scripts/rank_table_8pcs.json
  ```

  上述shell脚本将在后台运行分布训练。您可以在devicei文件下查看结果。

  ```python
  # grep "result:" device0/log
  epoch: 1 step: 23, loss is 9.789021
  epoch: 2 step: 23, loss is 8.736438
  epoch: 3 step: 23, loss is 8.006744
  ...
  epoch: 299 step: 23, loss is 1.136423
  epoch: 300 step: 23, loss is 1.216754
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估market1501数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/alignedreid++/resnet50-300_23.ckpt”。

  ```bash
  python test.py --data_url=[DATA_URL] --checkpoint_path=[CHECKPOINT_PATH] --device_id=0 > test.log 2>&1 &
  OR
  bash scripts/run_test_ascend.sh [DATA_URL] [CHECKPOINT_PATH] [DEVICE_ID]
  example: bash scripts/run_test_ascend.sh /dataset resnet50-300_23.ckpt 0
  ```

  上述python命令将在后台运行，您可以通过log_test.txt或者test.log文件查看结果。测试数据集的准确性如下：

  ```bash
  Results ----------
  mAP: 79.2%
  CMC curve
  Rank-1  : 91.7%
  Rank-5  : 97.4%
  Rank-10 : 98.6%
  Rank-20 : 99.3%
  starting re_ranking
  Computing CMC and mAP for re_ranking
  Results ----------
  mAP(RK): 91.4%
  CMC curve(RK)
  Rank-1  : 93.3%
  Rank-5  : 96.7%
  Rank-10 : 97.6%
  Rank-20 : 98.5%
  ```

## 导出过程

### 导出

需要修改的配置项为 ckpt_file和生成的模型名字.

```shell
python export.py --device_id=[DEVICE_ID] --ckpt_file=[CKPT_FILE] --file_name=[FILE_NAME]
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。

- 在昇腾310上使用market1501数据集进行推理

  具体使用到market1501数据集的test集和query集进行推理。需要修改的配置项为mindir模型所在路径，以及存放数据集query和gallery所在路径。

  ```bash
  # Ascend310 inference
  bash scripts/run_infer_310.sh [MINFIR_PATH] [QUERY_PATH] [GALLERY_PATH]
  ```

  上述shell脚本运行后。您可以在acc.log文件下查看最终结果。

  ```bash
  Results ----------
  mAP: 79.3%
  CMC curve
  Rank-1  : 91.7%
  Rank-5  : 97.4%
  Rank-10 : 98.6%
  Rank-20 : 99.2%
  Computing local distance...
  Computing local distance...
  Using global and local branches for reranking
  starting re_ranking
  Computing CMC and mAP for re_ranking
  Results ----------
  mAP(RK): 91.4%
  CMC curve(RK)
  Rank-1  : 93.2%
  Rank-5  : 96.8%
  Rank-10 : 97.7%
  Rank-20 : 98.6%
  ```

  推理时每个过程都保存在相应日志中，可查看对应日志了解具体信息。

# 模型描述

## 性能

### 评估性能

market1501上训练AlignedReID++

| 参数          | Ascend                                                       |
| ------------- | ------------------------------------------------------------ |
| 模型版本      | Inception V1                                                 |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8     |
| 上传日期      | 2021-012-14                                                  |
| MindSpore版本 | 1.3.0                                                        |
| 数据集        | market1501                                                   |
| 训练参数      | epoch=300, batch_size = 32, lr_max=5e-2,warmup_epochs=6,num_instances=8 |
| 优化器        | Momentum                                                     |
| 损失函数      | CrossEntropyLabelSmooth,TripletLossAlignedReID               |
| 输出          | 概率,特征向量                                                |
| 损失          | 1.004                                                        |
| 速度          | 单卡：79毫秒/步;  8卡：82毫秒/步                             |
| 总时长        | 单卡：63.85分钟;  8卡：11.28分钟                             |
| 参数(M)       | 13.0                                                         |
| ckpt模型      | 193M (.ckpt文件)                                             |
| 推理模型      | 21.50M (.onnx文件),  21.60M(.air文件)                        |
| 脚本          | [AlignedReID++脚本](https://gitee.com/mindspore/models/tree/master/research/cv/AlignedReID++) |

### 推理性能

market1501上评估AlignedReID++

| 参数          | Ascend                    |
| ------------- | ------------------------- |
| 模型版本      | Inception V1              |
| 资源          | Ascend 910；系统 Euler2.8 |
| 上传日期      | 2021-07-05                |
| MindSpore版本 | 1.3.0                     |
| 数据集        | ImageNet2012              |
| batch_size    | 256                       |
| 输出          | 概率                      |
| 准确性        | 8卡: 71.81%               |

## 使用流程

### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html)。下面是操作步骤示例：

在进行推理之前我们需要先导出模型，mindir可以在本地环境上导出。batch_size默认为1。

在昇腾310上使用market1501数据集的test集和query集进行推理。

```bash
# Ascend310 inference
bash scripts/run_infer_310.sh [MINFIR_PATH] [QUERY_PATH] [GALLERY_PATH]
# example:bash scripts/run_infer_310.sh /home/stu/lh/ascend310_infer/resnet50_imagenet.mindir /home/stu/lh/ascend310_infer/query /home/stu/lh/ascend310_infer/bounding_box_test
```

推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

```bash
Results ----------
mAP: 79.3%
CMC curve
Rank-1  : 91.7%
Rank-5  : 97.4%
Rank-10 : 98.6%
Rank-20 : 99.2%
Computing local distance...
Computing local distance...
Using global and local branches for reranking
starting re_ranking
Computing CMC and mAP for re_ranking
Results ----------
mAP(RK): 91.4%
CMC curve(RK)
Rank-1  : 93.2%
Rank-5  : 96.8%
Rank-10 : 97.7%
Rank-20 : 98.6%
```

### 迁移学习

待补充

# 随机情况说明

在train.py中，我们设置了“set_seed”的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。