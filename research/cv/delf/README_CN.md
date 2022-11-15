# 目录

- [目录](#目录)
- [DELF描述](#delf描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [数据集下载](#数据集下载)
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
    - [评估性能](#评估性能)
        - [Google-Landmarks-Dataset-v2上的DELF](#google-landmarks-dataset-v2上的delf)
    - [推理性能](#推理性能)
        - [Oxford5k上的DELF](#oxford5k上的delf)
        - [Paris6k上的DELF](#paris6k上的delf)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# DELF描述

​ DELF模型是一个图像检索模型，它提供了一种新的局部特征描述符，专门为大规模图像检索应用而设计。DELF在弱监督下进行学习，具体表现为仅需要使用图像级标签。它提供了一种新的语义特征选择注意力机制，在基于cnn的模型中，一次网络前向传递就足以获得关键点和描述符。在大规模的数据集下的评估表明，DELF比许多的全局和局部描述符表现要好得多。同时，DELE也表现出了出色的性能。

[论文](https://arxiv.org/abs/1612.06321)：Noh, H. , et al. "Large-Scale Image Retrieval with Attentive Deep Local Features." *2017 IEEE International Conference on Computer Vision (ICCV)* IEEE, 2017.

# 模型架构

​ DELF模型主要分为两部分：图像特征提取、注意力机制。

​ 特征提取：通过使用经过分类损失训练的CNN的特征提取层，构建一个全卷积网络(FCN)，从图像中提取密集特征。具体来说，DELF使用在ImageNet上完成训练的**ResNet50**模型进行迁移学习，对地标数据集进行微调，使用标准交叉熵损失训练网络进行图像分类训练，训练后使用ResNet50中conv4_x卷积块的输出作为图像中各个接受域的特征的集合。如此一来，局部描述符隐式学习与地标检索问题更相关的表示。通过这种方式，既不需要对象级标签也不需要像素级标签来获得改进的局部描述符。

​ 注意力机制：将提取出来的特征经过两层**1x1卷积层**的处理，得到每个特征对应的分数。将特征与分数进行相乘，之后将所有特征进行加和平均，使用标准交叉熵损失训练网络进行图像分类训练。训练完成后，利用该模型预测的分数作为特征的关键点选择标准。

# 数据集

训练数据集：[Google Landmarks Dataset v2](https://arxiv.org/abs/2004.01804)

- 数据集大小：601G，共200k个类、4.98M张彩色图像

  **[注]本模型仅需使用训练集的clean子集进行训练**

    - 训练集：522G，共4.1M张图像
    - 验证集：71G，共762k张图像
    - 测试集：8G，共118k张图像

    - 注：数据集下载详见[快速入门](#快速入门)或[数据集下载](#数据集下载)

评估数据集：

1. [Oxford5k](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)

   数据集大小：1.84G，共11个类、5062张彩色图像

    - 查询集：共55张图像
    - 索引集：共5007张图像

    - 注：数据集下载详见[快速入门](#快速入门)或[数据集下载](#数据集下载)

2. [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)

   数据集大小：2.43G，共11个类、6412张彩色图像

    - 查询集：共55张图像
    - 索引集：共6357张图像

    - 注：数据集下载详见[快速入门](#快速入门)或[数据集下载](#数据集下载)

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 依赖包
    - h5py版本：3.4.0
    - 其它依赖包信息详见requirements.txt文件

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 数据集下载和预处理

  ```shell
  # Google Landmarks Dataset v2 训练集下载以及转化为mindrecord文件

  # 【注】一共要下载4个csv文件，500个tar文件，500个md5文件，共占用约633G存储空间，请预留足够空间
  # 下载数据集时间较长，并且由于网络波动等原因存在，一次执行可能会下载失败，download_gldv2.sh的三个参数分别代码下载的数据集编号最小值，
  # 最大值，和保存路径
  bash scripts/download_gldv2.sh 0 499 [DATASET_PATH]
  # example: bash scripts/download_gldv2.sh 0 499 /home/gldv2
  # 下载完成后，可以比较下载得到的tar文件的md5值和md5文件，若一致，表明下载正确，否则下载错误，需要重新下载．
  # 重新下载时，修改前两个参数指定要下载的文件，例如指定'1, 1'表示下载images_001.tar，另外，train.csv, train_clean.csv,
  # train_attribution.csv, train_label_to_category.csv若已下载成功，可参考脚本注释进行适当修改

  cd [DATASET_PATH]/train
  # 对下载得到的500个tar文件解压
  tar xvf images_xxx.tar # 000, 001, 002, 003, ...

  python3 src/build_image_dataset.py \
  --train_csv_path=[DATASET_PATH]/train/train.csv \
  --train_clean_csv_path=[DATASET_PATH]/train/train_clean.csv \
  --train_directory=[DATASET_PATH]/train/*/*/*/ \
  --output_directory=[DATASET_PATH]/mindrecord/ \
  --num_shards=128 \
  --validation_split_size=0.2

  # Oxford5k和Paris6k以及它们对应的ground truth文件下载
  bash scripts/download_oxf.sh [DATASET_PATH]
  bash scripts/download_paris.sh [DATASET_PATH]
  ```

- 预训练权重下载

  ```shell
  # 下载ImageNet预训练的Resnet50权重和pca降维预训练转换矩阵
  bash scripts/download_pretrained.sh
  ```

- Ascend处理器环境运行

  ```shell
  # 在delf_config.yaml中修改配置
  # 添加训练数据集、预训练权重路径，例子：
  traindata_path: "/home/gldv2/mindrecord/train.mindrecord000"
  imagenet_checkpoint: "/home/delf/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
  ```

  ```shell
  # 请确保前面步骤的预训练权重已经下载完毕！
  # 运行训练示例，分为两阶段进行训练
  # 微调阶段：
  bash scripts/run_1p_train.sh tuning
  # 注意力训练阶段：
  # 修改[CHECKPOINT]为微调阶段得到的checkpoint文件路径
  bash scripts/run_1p_train.sh attn [CHECKPOINT]
  # example: bash scripts/run_1p_train.sh attn ./ckpt/checkpoint_delf_tuning-1_4989.ckpt

  # 运行分布式训练示例
  # 微调阶段
  bash scripts/run_8p_train.sh [RANK_TABLE_FILE] tuning
  # example: bash scripts/run_8p_train.sh /home/rank_table_8pcs.json tuning
  # 注意力训练阶段
  bash scripts/run_8p_train.sh [RANK_TABLE_FILE] attn [CHECKPOINT]
  # example: bash scripts/run_8p_train.sh /home/rank_table_8pcs.json attn /home/delf/ckpt/checkpoint_delf_tuning-1_4989.ckpt

  # 运行评估示例，参数IMAGES_PATH为数据集路径，GT_PATH为对应的ground truth文件路径
  # 评估方式1：两张图片进行特征匹配。注意在list_images.txt中填写要进行匹配的图片名字。匹配结果保存为eval_match.png
  bash scripts/run_eval_match_images.sh [IMAGES_PATH] [CHECKPOINT] [DEVICES]
  # example: bash scripts/run_eval_match_images.sh /home/oxford5k_images/ /home/delf/ckpt/checkpoint_delf_attn-1_4989.ckpt 0
  # 评估方式2：针对Oxford5k或Paris6k进行图像检索。检索结果保存为mAP.txt
  bash scripts/run_eval_retrieval_images.sh [IMAGES_PATH] [GT_PATH] [CHECKPOINT] [DEVICES]
  # example: bash scripts/run_eval_retrieval_images.sh /home/paris_images/ /home/paris_120310/ /home/delf/ckpt/checkpoint_delf_attn-1_4989.ckpt 01

  # 导出MindIR模型到当前目录，文件名为DELF_MindIR.mindir，示例
  python export.py --device_id=0 --ckpt_path=/home/delf/ckpt/...

  # 运行推理示例
  # 先下载好pca降维的相关转换矩阵
  bash scripts/download_pretrained.sh
  # 方式1：图片匹配。需要在list_images.txt中填写要进行匹配的图片名字，检索结果保存为test_match.png
  # 参数GEN_MINDIR_PATH为导出的MindIR文件路径，IMAGES_PATH为评估数据集路径，GT_PATH为评估数据集对应的ground truth文件路径
  bash scripts/run_infer_310_match_images.sh [GEN_MINDIR_PATH] [IMAGES_PATH] [DEVICE_ID]
  # 方式2：图片检索, 检索结果保存为mAP.txt
  bash scripts/run_infer_310_retrieval_images.sh [GEN_MINDIR_PATH] [IMAGES_PATH] [GT_PATH] [DEVICE_ID]
  ```

  [注]以上参数所使用的路径均需为绝对路径

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练

    ```python
    # (1) 在网页上设置 "config_path='/path_to_code/imagenet_config.yaml'"
    # (2) 执行a或者b
    #       a. 在 imagenet_config.yaml 文件中设置 "enable_modelarts: True"
    #          在 imagenet_config.yaml 文件中设置 "train_state: 'tuning'"或"train_state: 'attn'"
    #          在 imagenet_config.yaml 文件中设置 "traindata_path: '/cache/data/mindrecord/train.mindrecord000'"
    #          在 imagenet_config.yaml 文件中设置 "imagenet_checkpoint: '/cache/checkpoint_path/resnet50_weights_
    #          tf_dim_ordering_tf_kernels_notop.h5'"
    #          在 imagenet_config.yaml 文件中设置 "checkpoint_path: '/cache/checkpoint_path/...'"（attn阶段）
    #          在 imagenet_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置设置 "train_state='tuning'"或"train_state: 'attn'"
    #          在网页上设置设置 "traindata_path='/cache/data/mindrecord/train.mindrecord000'"
    #          在网页上设置设置 "imagenet_checkpoint='/cache/checkpoint_path/resnet50_weights_tf_dim_
    #           ordering_tf_kernels_notop.h5'"
    #          在网页上设置设置 "checkpoint_path='/cache/checkpoint_path/...'"（attn阶段）
    #          在网页上设置设置 其他参数
    # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
    # (4) 在网页上设置你的代码路径为 "/path/delf"
    # (5) 在网页上设置启动文件为 "train.py"
    # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"checkpoint文件路径"、"作业日志路径"等
    # (7) 创建训练作业
    ```

# 脚本说明

## 脚本及样例代码

```bash
.
└─ cv
   └─ delf
    ├── ascend310_infer                                      # 310推理源代码
    ├── model_utils                                          # model_utils工具包
    ├── scripts
    │   ├── download_gldv2.sh                                # 下载 Google Landmarks Dataset v2 的shell脚本
    │   ├── download_oxf.sh                                  # 下载 Oxford5k 的shell脚本
    │   ├── download_paris.sh                                # 下载 Paris6k 的shell脚本
    │   ├── download_pretrained.sh                           # 下载预训练权重的shell脚本
    │   ├── run_1p_train.sh                                  # 单卡训练的shell脚本
    │   ├── run_8p_train.sh                                  # 分布式训练的shell脚本
    │   ├── run_eval_match_images.sh                         # 图像匹配评估的shell脚本
    │   ├── run_eval_retrieval_images.sh                     # 图像检索评估的shell脚本
    │   ├── run_infer_310_match_images.sh                    # 310推理图像匹配的shell脚本
    │   └── run_infer_310_retrieval_images.sh                # 310推理图像检索的shell脚本
    ├── src
    │   ├── __init__.py                                      # 初始化文件
    │   ├── box_list_np.py                                   # 边界框定义
    │   ├── box_list_ops_np.py                               # 边界框操作定义
    │   ├── build_feature_dataset.py                         # 创建特征数据库
    │   ├── build_image_dataset.py                           # 创建训练数据集
    │   ├── convert_h5_to_weight.py                          # 转换h5权重文件为ckpt权重
    │   ├── data_augmentation_parallel.py                    # 训练数据预处理和加载
    │   ├── dataset.py                                       # 读取查询集、索引集和gt
    │   ├── delf_config.py                                   # 图像金字塔配置
    │   ├── delf_model.py                                    # delf网络定义
    │   ├── extract_feature.py                               # 提取图像特征
    │   ├── extract_utils_np.py                              # 提取图像特征工具（pca降维）
    │   ├── feature_io.py                                    # 读写图像特征
    │   ├── preprocess.py                                    # 310推理预处理脚本
    │   ├── postprocess.py                                   # 310推理后处理脚本
    │   ├── match_images.py                                  # 图像特征匹配
    │   └── perform_retrieval.py                             # 图像检索
    ├── list_images.txt                                      # 图像匹配配置文件
    ├── train.py                                             # 训练脚本
    ├── delf_config.yaml                                     # 训练配置文件
    ├── eval.py                                              # mAP评估脚本
    ├── export.py                                            # MINDIR模型导出脚本
    └── README_CN.md                                         # delf的文件描述
```

## 脚本参数

- 在delf_config.yaml中可以配置训练参数。

  ```yaml
  train_state: "attn"      # 训练阶段
  seed: 0                  # 数据集shuffle种子
  batch_size: 32           # 训练批次大小
  start_iter: 1            # 训练开始时的步数，动态学习率根据此参数进行初始化
  max_iters: 500000        # 训练最大步数
  image_size: 321          # 数据预处理后图像大小
  initial_lr: 0.005        # 学习率初始值
  attention_loss_weight: 1.0  # 注意力比重，不推荐修改
  traindata_path: "/gldv2/mindrecord/train.mindrecord000" # 训练数据集路径
  keep_checkpoint_max: 1                                  # checkpoint最大保存数
  imagenet_checkpoint: "/delf/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"  # Resnet50预训练权重
  checkpoint_path: "/delf/checkpoint_delf_tuning-1_4989.ckpt"  # 加载checkpoint路径
  save_ckpt: "./ckpt/"                                         # 保存checkpoint路径
  save_ckpt_step: 2000                                         # checkpoint保存间隔（单位：训练步数）
  need_summary: False                                          # 是否需要使用summary collector
  save_summary: "./summary_test"                               # summary和profier的保存路径
  need_profile: False                                          # 是否需要进行性能分析
  ```

  更多配置细节请参考脚本`delf_config.yaml`。

## 数据集下载

​ 使用以下命令可以下载`Google Landmarks Dataset v2`数据集的训练集，并且自动提取它的clean子集（具体定义参考[数据集](#数据集)中数据集对应的论文）转化为mindrecord格式：

```shell
bash scripts/download_gldv2.sh 0 499 [DATASET_PATH]
# example: bash scripts/download_gldv2.sh 0 499 /home/gldv2
```

​ 目录以及说明：

```shell
.
└─ gldv2
    ├── train
    │   ├── 0                                                # 图片存放的文件夹，共15个
    │   ├── 1
    │   ├── ...
    │   ├── f
    │   ├── md5.images_000.txt                               # md5校验文件，共500个
    │   ├── md5.images_001.txt
    │   ├── ...
    │   ├── md5.images_499.txt
    │   ├── images_000.tar                                   # 图像压缩包，共500个
    │   ├── images_001.tar
    │   ├── ...
    │   ├── images_499.tar
    │   ├── train_attribution.csv                            # 保存图像标签属性等信息的文件
    │   ├── train_clean.csv
    │   ├── train.csv
    │   └── train_label_to_category.csv
    ├── mindrecord
    │   ├── relabeling.csv                                   # 图像标签文件
    │   ├── train.mindrecord000                              # mindrecord文件，，共128x2个
    │   ├── train.mindrecord000.db
    │   ├── train.mindrecord001
    │   ├── train.mindrecord001.db
    │   ├── ...
    │   ├── ...
    │   ├── train.mindrecord127
    │   └── train.mindrecord127.db
```

​ 使用以下命令可以下载Oxford5k和Paris6k的数据集：

```shell
bash scripts/download_oxf.sh [DATASET_PATH]
bash scripts/download_paris.sh [DATASET_PATH]
```

​ 目录以及说明：

```sh
.
└─ dataset
    ├── oxbuild_images                         # Oxford5k图像文件夹
    ├── gt_files_170407                        # Oxford5k的gt文件夹
    ├── paris6k_images                         # Paris6k图像文件夹
    └── paris_120310                           # Paris6k的gt文件夹
```

## 训练过程

### 训练

- Ascend处理器环境运行

  在训练前需要下载预训练模型用于迁移学习，执行以下命令：

  ```shell
  # 下载ImageNet预训练的Resnet50权重和pca降维预训练转换矩阵
  bash scripts/download_pretrained.sh
  ```

  之后在delf_config中配置好预训练权重路径以及其它参数，路径要写为绝对路径：

  ```yaml
  imagenet_checkpoint: "/home/delf/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
  ```

  为了让模型达到更好更快地收敛，delf模型需要进行两个阶段的训练，微调阶段和注意力训练阶段，微调阶段运行命令：

  ```bash
  bash scripts/run_1p_train.sh tuning
  ```

  注意力训练阶段运行命令：

  ```bash
  bash scripts/run_1p_train.sh attn [CHECKPOINT]
  # example: bash scripts/run_1p_train.sh attn ./ckpt/checkpoint_delf_tuning-1_4989.ckpt
  ```

  需要注意的是注意力训练阶段要载入微调阶段已经训练好的checkpoint权重。

  上述命令将在后台运行，您可以通过train_tuning.log或train_attn.log文件查看结果。

  训练结束后，您可在默认脚本文件夹的`ckpt`目录下找到检查点文件。

  可以通过log文件查看损失值：

  ```bash
  epoch: 1 step: 39000, loss is 1.5936852
  epoch: 1 step: 39100, loss is 1.5249567
  ...
  ```

  也可以将`delf_config.yaml`中的`need_summary`设置为`True`，之后利用mindinsight工具打开默认脚本文件中生成的`summary_test`目录便可以看到可视化后的损失值收敛情况。

  mindinsight的使用方式可参考一下链接说明：[Mindinsight](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.5/training_visual_design.html)

### 分布式训练

- Ascend处理器环境运行

  ```bash
  # 微调阶段
  bash scripts/run_8p_train.sh [RANK_TABLE_FILE] tuning
  # 注意力训练阶段
  bash scripts/run_8p_train.sh [RANK_TABLE_FILE] attn [CHECKPOINT]
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/.log文件查看结果，同样也可以通过summary查看收敛情况。

## 评估过程

### 评估

在执行评估前确保已经下载好pca降维预训练参数，并且其放置在默认脚本目录内（该过程在[训练](#训练)的下载预训练模型中已经完成）。其文件结构：

```bash
└─ delf
  └─ pca
    ├── mean.npy
    ├── pca_proj_mat.npy
    └── pca_variances.npy
```

有两种评估方式，分别为图像特征匹配和计算图像检索mAP。以下评估过程均需在Ascend环境下完成。

- **图像特征匹配**

  图像特征匹配是比较主观的一种评估方式，具体过程为：选择两张需要进行特征匹配的图片，使用训练好的DELF模型提取这两张图片的特征，然后在两张图像相似的特征之间进行连线。

  首先是输入选择好的两张图片，在默认脚本目录下的`list_images.txt`中，输入图片去除后缀后的文件名，例如：

  ```yaml
  # vim list_images.txt
  hertford_000056
  oxford_000317
  ```

  之后运行图像特征匹配脚本。参数为图像所在的文件夹路径，检查点路径，运行设备。请将路径设置为绝对全路径，例如`/username/delf/ckpt/attn.ckpt`。由于提取特征时，设备之间没有通信，因此可以多用几个设备进行特征提取，如`DEVICES`可以设置为`03`，代表使用device_id为0和3的设备进行特征提取：

  ```bash
  bash scripts/run_eval_match_images.sh [IMAGES_PATH] [CHECKPOINT] [DEVICES]
  ```

  上述命令将在后台打印日志，您可以通过`extract_feature.log`文件查看特征提取情况，通过`match_images.log`文件查看特征匹配情况。最后得到展示两个图像匹配特征的图片，它会保存为默认脚本目录下的`eval_match.png`。

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如`/username/delf/train_parallel0/ckpt/attn-1-4898.ckpt`。

- **计算图像检索mAP**

  图像检索是在Oxford5k和Paris6k这两个数据集上做的，数据集把图片划分成了查询集和检索集，并且提供了查询集中图片对应的ground truth图片，利用这些文件可以计算出DELF模型“以图的内容检索相关图片”能力的评价指标mAP。

  整个过程分为三步：提取特征，利用特征进行检索，计算mAP。这三个过程被整合在一个脚本中。

  在运行以下命令之前，请将路径设置为绝对全路径，例如`/username/delf/ckpt/attn.ckpt`。

  ```bash
  bash scripts/run_eval_retrieval_images.sh [IMAGES_PATH] [GT_PATH] [CHECKPOINT] [DEVICES]
  ```

  上述命令将在后台打印日志，您可以通过`extract_feature.log`文件查看特征提取情况，通过`./retrieval_dataset/process[X]/retrieval[X].log`文件查看检索情况，通过`calculate_mAP.log`文件查看计算mAP的情况。计算得到的mAP结果会在脚本运行完成后打印在终端，也会保存到默认脚本目录下的mAP.txt文件中：

  ```matlab
  # cat mAP.txt
  easy
    mAP=91.85
    mP@k[10 20 30 40 50 60 70 80 90] [91.07 83.47 78.89 74.85 71.79 69.44 67.51 65.84 64.65]
    mR@k[10 20 30 40 50 60 70 80 90] [61.07 75.89 83.06 86.52 89.47 91.94 93.47 94.39 95.32]
  hard
    mAP=70.48
    mP@k[10 20 30 40 50 60 70 80 90] [78.18 67.34 60.84 56.26 52.63 49.89 47.63 45.85 44.29]
    mR@k[10 20 30 40 50 60 70 80 90] [46.36 59.41 66.55 71.33 74.33 76.97 79.62 81.93 82.98]
  medium
    mAP=82.09
    mP@k[10 20 30 40 50 60 70 80 90] [89.36 84.21 78.45 74.3  70.04 66.04 62.77 59.81 57.25]
    mR@k[10 20 30 40 50 60 70 80 90] [40.62 55.01 63.5  69.98 74.58 77.81 80.85 82.99 84.58]
  ```

  根据不同的标准设置了三种mAP指标。除此之外，mP表示平均准确率，mR表示平均召回率，每行左边中括号内的是检索次数，右边中括号内的是对应的精确率或召回率。

## 导出过程

### 导出

导出MindIR模型到当前目录，文件名为DELF_MindIR.mindir：

```shell
python export.py --device_id=0 --ckpt_path=/home/delf/ckpt/...
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在进行推理之前我们需要先导出mindir模型。

- 在昇腾310上进行推理

  这里与评估相似，同样提供了图像特征匹配和计算图像检索mAP的两个脚本。推理结果查看方式与评估类似。

  ```shell
  # 方式1：图片匹配。同样在list_images.txt中填写要进行匹配的图片名字。
  bash scripts/run_infer_310_match_images.sh [GEN_MINDIR_PATH] [IMAGES_PATH] [DEVICE_ID]
  # 方式2：图片检索
  bash scripts/run_infer_310_retrieval_images.sh [GEN_MINDIR_PATH] [IMAGES_PATH] [GT_PATH] [DEVICE_ID]
  ```

# 模型描述

## 性能

### 评估性能

#### Google-Landmarks-Dataset-v2上的DELF

| 参数          | Ascend 910                                                                              |
| ------------- | --------------------------------------------------------------------------------------- |
| 模型版本      | DELF                                                                                    |
| 资源          | Ascend 910                                                                              |
| 上传日期      | 2021-12-11                                                                              |
| MindSpore版本 | 1.3.0                                                                                   |
| 数据集        | Google Landmarks Dataset v2                                                             |
| 训练参数      | steps=500000, batch_size = 32, lr=0.01（初始）                                          |
| 优化器        | Momentum                                                                                |
| 损失函数      | Softmax交叉熵                                                                           |
| 输出          | 概率                                                                                    |
| 损失          | 微调阶段：0.50413    注意力训练阶段：1.34352                                            |
| 速度          | 微调阶段：171毫秒/步;  注意力训练阶段：156毫秒/步                                       |
| 总时长        | 微调阶段：23小时36.66分钟;  注意力训练阶段：21小时39.96分                               |
| 微调检查点    | 2.89G (.ckpt文件)                                                                       |
| 推理模型      | 92.26M (.mindir文件)                                                                    |
| 脚本          | [delft脚本](https://gitee.com/mindspore/models/tree/master/research/cv/delf)            |

### 推理性能

#### Oxford5k上的DELF

| 参数          | Ascend            |
| ------------- | ----------------- |
| 模型版本      | DELF              |
| 资源          | Ascend            |
| 上传日期      | 2021-12-11        |
| MindSpore版本 | 1.3.0             |
| 数据集        | 5062张图像        |
| batch_size    | 7                 |
| 输出          | feature和location |
| mAP           | 91.85             |

#### Paris6k上的DELF

| 参数          | Ascend            |
| ------------- | ----------------- |
| 模型版本      | DELF              |
| 资源          | Ascend            |
| 上传日期      | 2021-12-11        |
| MindSpore版本 | 1.3.0             |
| 数据集        | 6412张图像        |
| batch_size    | 7                 |
| 输出          | feature和location |
| mAP           | 87.87             |

# 随机情况说明

在data_augmentation_parallel.py中，设置了“shuffle”函数内的种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
