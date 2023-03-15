# 目录

- [MaskRCNN描述](#maskrcnn描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [训练脚本参数](#训练脚本参数)
        - [参数配置](#参数配置)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
        - [训练结果](#训练结果)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [评估结果](#评估结果)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
        - [推理结果](#推理结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [MaskRCNN描述](#目录)

MaskRCNN是一个概念简单、灵活、通用的目标实例分割框架。该方法能够有效检测图像中的目标，同时为每个实例生成高质量的分割掩码。MaskRCNN通过添加与现有边界框平行的分支来预测目标掩码，在Faster R-CNN基础上进行了扩展。
 MaskRCNN易于训练，且在Faster R-CNN基础上只增加了很小的开销，运行速度可达到5FPS。此外，MaskRCNN易于推广到其他任务，例如，允许在同一框架中估计人体姿态。
在COCO挑战套件的全部三个赛道中，包括实例分割、边界框检测和人体关键点检测，结果均达到TOP水平。MaskRCNN在单项任务上的表现均优于所有现有的单模参赛作品，包括COCO 2016挑战赛的获奖作品。

# [模型架构](#目录)

MaskRCNN是一个两阶段目标检测网络，通过添加与现有边界框平行的分支来预测目标掩码，进一步扩展了Faster R-CNN。该网络使用区域候选网络（RPN），与检测网络共享整个图像的卷积特征，因此区域候选的计算几乎不需要额外的代价。整个网络通过共享卷积特征，进一步将RPN和掩码分支组合成一个网络，
使用MobileNetV1作为maskrcnn_mobilenetv1网络的主干。

[论文](http://cn.arxiv.org/pdf/1703.06870v3): Kaiming He, Georgia Gkioxari, Piotr Dollar and Ross Girshick. "MaskRCNN"

# [数据集](#目录)

您可以基于原始论文中提到的数据集运行脚本，也可以采用在相关域/网络架构中广泛使用的脚本。接下来我们将介绍如何使用下面的数据集运行脚本。

- [COCO2017](https://cocodataset.org/)是一个广受欢迎的数据集，包括边界框和像素级的内容标注。这些标注可用于场景理解任务，如语义分割、目标检测和图像描述。分别提供了11.8万和5000张图像用于训练和评估。

- 数据集大小：19G
    - 训练集：120G，11.8万张图像
    - 评估集: 1G，5000张图像
    - 标注：241M，包括实例、说明、人体关键点等。

- 数据格式：图像和.json文件
    - 注：数据将在dataset.py中处理。

# [环境要求](#目录)

- 硬件（Ascend或GPU）
    - 使用Ascend、CPU或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

- 第三方库

    ```bash
    pip install Cython
    pip install pycocotools
    pip install mmcv=0.2.14
    ```

# [快速入门](#目录)

1. 下载COCO2017数据集。

2. 按需修改`default_config.yaml`中的COCO_ROOT和其他参数。目录结构如下所示：

    ```
    .
    └─cocodataset
      ├─annotations
        ├─instance_train2017.json
        └─instance_val2017.json
      ├─val2017
      └─train2017
    ```

    如您使用自己的数据集进行网络训练，**请在运行脚本时选择“其他”数据集**。
    创建一个txt文件，用于存储数据集信息，格式如下：

    ```
    train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
    ```

    每行一个图像标注，用空格分隔。第一列是图像的相对路径，其他列为框和类信息，格式为[xmin,ymin,xmax,ymax,class]。从图像路径中读取图像，该路径包括`IMAGE_DIR`（数据集目录）和`ANNO_PATH`（TXT文件路径）中的相对路径。您可以在default_config.yaml中设置该路径。

3. 执行训练脚本。
    数据集准备完成后，您可以按以下方式开始训练：

    ```bash
    在Ascend上运行：

    # 分布式训练
    bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT(optional)]
    # 示例：bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cocodataset/

    # 单机训练
    bash run_standalone_train.sh [DATA_PATH] [PRETRAINED_CKPT(optional)]
    # 示例：bash run_standalone_train.sh /home/DataSet/cocodataset/

    在CPU上运行：

    # 单机训练
    bash run_standalone_train_cpu.sh [PRETRAINED_PATH](optional)

    在GPU上运行：

    # 分布式训练
    bash run_distribute_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH]（可选）
    ```

    注：
    1. 为了加快数据预处理速度，MindSpore提供了一种名为MindRecord的数据格式，因此第一步是在训练前根据COCO2017数据集生成MindRecord文件。将原始COCO2017数据集转换为MindRecord格式的过程可能需要4小时左右。
    2. 对于分布式训练，需要提前创建JSON格式的[HCCL配置文件](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。
    3. 对于像maskrcnn_Mobilenetv1这样的大模型，最好设置外部环境变量`export HCCL_CONNECT_TIMEOUT=600`，将hccl连接检查时间从默认的120秒延长到600秒。否则，可能会连接超时，因为编译时间会随着模型增大而增加。

4. 执行评估脚本。

    训练结束后，您可以按以下方式开始评估：

    ```bash
    # 在Ascend上运行评估
    bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    # 示例：bash run_eval.sh /home/DataSet/cocodataset/annotations/instances_val2017.json /home/model/maskrcnn_mobilenetv1/ckpt/mask_rcnn-5_7393.ckpt /home/DataSet/cocodataset/ "Ascend" 0

    # 在CPU上运行评估
    bash run_eval_cpu.sh [ANN_FILE] [CHECKPOINT_PATH]

    # 在GPU上运行评估
    bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    ```

    注：
    1. VALIDATION_JSON_FILE是用于评估的标签JSON文件。

5. 执行推理脚本。

    训练结束后，您可以按以下方式开始推理：

    ```shell
    # 推理
    bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [ANN_FILE_PATH]
    ```

    注：
    1. MODEL_PATH是一个模型文件，可通过脚本导出。
    2. AN_FILE_PATH是用于推理的标注文件。

- [ModelArts](https://support.huaweicloud.com/modelarts/)环境上运行

    ```bash
    # Ascend环境上运行8卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"distribute=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='cocodataset'"。
    #          在default_config.yaml文件中设置"base_lr=0.02"。
    #          在default_config.yaml文件中设置"mindrecord_dir='./MindRecord_COCO'"。
    #          在default_config.yaml文件中设置"data_path='/cache/data'"。
    #          在default_config.yaml文件中设置"ann_file='./annotations/instances_val2017.json'"。
    #          在default_config.yaml文件中设置"epoch_size=12"。
    #          在default_config.yaml文件中设置"ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'"。
    #          （可选）在default_config.yaml文件上设置"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='cocodataset'"。
    #          在网页上添加"distribute=True"。
    #          在网页上添加"base_lr=0.02"。
    #          在网页上添加"mindrecord_dir='./MindRecord_COCO'"。
    #          在网页上添加"data_path='/cache/data'"。
    #          在网页上添加"ann_file='./annotations/instances_val2017.json'"。
    #          在网页上添加"epoch_size=12"。
    #          在default_config.yaml文件中设置"ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'"。
    #          （可选）在网页上添加"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）如需微调，请将预训练的模型上传或复制到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，在本地运行"train.py"脚本从coco2017数据集创建MindRecord数据集。
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --data_path=$DATA_PATH --ann_file=$ANNO_PATH"
    #          再将MindRecord数据集压缩到一个zip文件中。
    #          最后将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/maskrcnn"。
    # （6）在网页上设置启动文件为“train.py”。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    #
    # Ascend环境上运行单卡训练
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='cocodataset'"。
    #          在default_config.yaml文件中设置"mindrecord_dir='./MindRecord_COCO'"。
    #          在default_config.yaml文件中设置"data_path='/cache/data'"。
    #          在default_config.yaml文件中设置"ann_file='./annotations/instances_val2017.json'"。
    #          在default_config.yaml文件中设置"epoch_size=12"。
    #          在default_config.yaml文件中设置"ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'"。
    #          （可选）在default_config.yaml文件上设置"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='cocodataset'"。
    #          在网页上添加"mindrecord_dir='./MindRecord_COCO'"。
    #          在网页上添加"data_path='/cache/data'"。
    #          在网页上添加"ann_file='./annotations/instances_val2017.json'"。
    #          在网页上添加"epoch_size=12"。
    #          在default_config.yaml文件中设置"ckpt_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'"。
    #          （可选）在网页上添加"checkpoint_url='s3://dir_to_your_pretrained/'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）如需微调，请将预训练的模型上传或复制到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，在本地运行"train.py"脚本从coco2017数据集创建MindRecord数据集。
    #             "python train.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --data_path=$DATA_PATH --ann_file=$ANNO_PATH"
    #          再将MindRecord数据集压缩到一个zip文件中。
    #          最后将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/maskrcnn"。
    # （6）在网页上设置启动文件为“train.py”。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    #
    # Ascend环境上运行单卡评估
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"need_modelarts_dataset_unzip=True"。
    #          在default_config.yaml文件中设置"modelarts_dataset_unzip_name='cocodataset'"。
    #          在base_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_trained_ckpt/'"。
    #          在default_config.yaml文件中设置"checkpoint_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'"。
    #          在default_config.yaml文件中设置"mindrecord_file='/cache/data/cocodataset/MindRecord_COCO'"。
    #          在default_config.yaml文件中设置"data_path='/cache/data'"。
    #          在default_config.yaml文件中设置"ann_file='./annotations/instances_val2017.json'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"need_modelarts_dataset_unzip=True"。
    #          在网页上添加"modelarts_dataset_unzip_name='cocodataset'"。
    #          在网页上添加"checkpoint_url='s3://dir_to_your_trained_model/'"。
    #          在网页上添加"checkpoint_path='./ckpt_maskrcnn/mask_rcnn-12_7393.ckpt'"。
    #          在default_config.yaml文件中设置"mindrecord_file='/cache/data/cocodataset/MindRecord_COCO'"。
    #          在网页上添加"data_path='/cache/data'"。
    #          在default_config.yaml文件中设置"ann_file='./annotations/instances_val2017.json'"。
    #          在网页上添加其他参数。
    # （2）准备模型代码。
    # （3）上传或复制训练好的模型到S3桶。
    # （4）执行a或b（建议执行a）。
    #       a. 首先，在本地运行"eval.py"脚本从coco2017数据集创建MindRecord数据集。
    #             "python eval.py --only_create_dataset=True --mindrecord_dir=$MINDRECORD_DIR --data_path=$DATA_PATH --ann_file=$ANNO_PATH \
    #              --checkpoint_path=$CHECKPOINT_PATH"
    #          再将MindRecord数据集压缩到一个zip文件中。
    #          最后将zip数据集上传到S3桶。（您也可以上传mindrecord数据集，但可能比较耗时。）
    #       b. 将原始的coco数据集上传到S3桶中。
    #           （数据集会在每次训练时进行转换，可能会比较耗时。）
    # （5）在网页上设置代码目录为"/path/maskrcnn"。
    # （6）在网页上设置启动文件为"eval.py"。
    # （7）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （8）创建作业。
    ```

- 在ModelArts上导出并开始评估（如果你想在ModelArts上运行，可以参考[ModelArts](https://support.huaweicloud.com/modelarts/)官方文档。

1. 使用voc val数据集在ModelArts上导出并评估多尺度翻转s8。

    ```python
    # （1）执行a或b。
    #       a. 在base_config.yaml文件中设置"enable_modelarts=True"。
    #          在base_config.yaml文件中设置"file_name='maskrcnn_mobilenetv1'"。
    #          在base_config.yaml文件中设置"file_format='MINDIR'"。
    #          在beta_config.yaml文件中设置"checkpoint_url='/The path of checkpoint in S3/'"。
    #          在base_config.yaml文件中设置"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在base_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"file_name='maskrcnn_mobilenetv1'"。
    #          在网页上添加"file_format='MINDIR'"。
    #          在网页上添加"checkpoint_url='/The path of checkpoint in S3/'"。
    #          在网页上添加"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在网页上添加其他参数。
    # （2）上传或复制训练好的模型到S3桶。
    # （3）在网页上设置代码目录为"/path/maskrcnn_mobilenetv1"。
    # （4）在网页上设置启动文件为"export.py"。
    # （5）在网页上设置"Dataset path"、"Output file path"和"Job log path"。
    # （6）创建作业。
    ```

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```shell
.
└─MaskRcnn_Mobilenetv1
  ├─README.md                             # README
  ├─ascend310_infer                       # Ascend 310推理实现
  ├─scripts                               # shell脚本
    ├─run_standalone_train.sh             # Ascend上单机训练（单卡）
    ├─run_standalone_train_cpu.sh         # CPU上单机训练（单卡）
    ├─run_distribute_train_gpu.sh         # GPU上并行训练(8卡)
    ├─run_distribute_train.sh             # Ascend上并行训练(8卡)
    ├─run_infer_310.sh                    # 用于Ascend 310上运行推理的shell脚本
    ├─run_eval_cpu.sh                     # CPU上运行评估
    └─run_eval.sh                         # Ascend或GPU上运行评估
  ├─src
    ├─maskrcnn_mobilenetv1
      ├─__init__.py
      ├─anchor_generator.py               # 生成基准边界框锚点
      ├─bbox_assign_sample.py             # 筛选第一阶段学习的正负边界框
      ├─bbox_assign_sample_stage2.py      # 筛选第二阶段学习的正负边界框
      ├─mask_rcnn_mobilenetv1.py          # maskrcnn_Mobilenetv1的主要网络架构
      ├─fpn_neck.py                       # FPN网络
      ├─proposal_generator.py             # 根据特征图生成候选区
      ├─rcnn_cls.py                       # RCNN边界框回归分支
      ├─rcnn_mask.py                      # RCNN掩码分支
      ├─mobilenetv1.py                    # 骨干网
      ├─roi_align.py                      # ROI对齐网络
      └─rpn.py                            # 区域生成网络
    ├─util.py                             # 例行操作
    ├─model_utils                         # 网络配置
      ├─__init__.py
      ├─config.py                         # 网络配置
      ├─device_adapter.py                 # 获取云ID
      ├─local_adapter.py                  # 获取本地ID
      ├─moxing_adapter.py                 # 参数处理
    ├─dataset.py                          # 数据集工具
    ├─lr_schedule.py                      # 学习率生成器
    ├─network_define.py                   # maskrcnn_Mobilenetv1的网络定义
    └─util.py                             # 例行操作
  ├─default_config.yaml                   # 默认配置
  ├─mindspore_hub_conf.py                 # MindSpore HUB接口
  ├─export.py                             #用于导出AIR、MindIR模型的脚本
  ├─eval.py                               # 评估脚本
  ├─postprogress.py                       #Ascend 310上推理的后处理脚本
  └─train.py                              # 训练脚本
```

## [脚本参数](#目录)

### [训练脚本参数](#目录)

```bash
在Ascend上运行：

# 分布式训练
用法：bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_CKPT(optional)]
# 示例：bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cocodataset/

# 单机训练
用法：bash run_standalone_train.sh [DATA_PATH] [PRETRAINED_CKPT(optional)]
# 示例：bash run_standalone_train.sh /home/DataSet/cocodataset/

在CPU上运行：

# 单机训练
用法：bash run_standalone_train_cpu.sh [PRETRAINED_MODEL](optional)

在GPU上运行：

# 分布式训练
用法：bash run_distribute_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH]（可选）
```

### [参数配置](#目录)

```default_config.yaml
"img_width": 1280,          # 输入图像的宽度
"img_height": 768,          # 输入图像的高度

# 数据增强中的随机阈值
"keep_ratio": True,
"keep_ratio": True,
"expand_ratio": 1.0,

"max_instance_count": 128, # 每个图像的最大边界框数
"mask_shape": (28, 28),    # rcnn_mask中掩码的形状

# 锚点
"feature_shapes": [(192, 320), (96, 160), (48, 80), (24, 40), (12, 20)], # FPN特征图的形状
"anchor_scales": [8],                                                    # 基锚点面积
"anchor_ratios": [0.5, 1.0, 2.0],                                        # 基准锚点高宽比
"anchor_strides": [4, 8, 16, 32, 64],                                    # 每个特征图级别的步长
"num_anchors": 3,                                                        # 每个像素的锚点数

# FPN
"fpn_in_channels": [128, 256, 512, 1024],                               # 每层的通道大小
"fpn_out_channels": 256,                                                 # 每层的输出通道大小
"fpn_num_outs": 5,                                                       # 输出特征图大小

# RPN
"rpn_in_channels": 256,                                                  # 输入通道大小
"rpn_feat_channels": 256,                                                # 特征输出通道大小
"rpn_loss_cls_weight": 1.0,                                              # RPN损失中边界框分类的权重
"rpn_loss_reg_weight": 1.0,                                              # RPN损失中边界框回归的权重
"rpn_cls_out_channels": 1,                                               # 分类输出通道大小
"rpn_target_means": [0., 0., 0., 0.],                                    # 边界框编解码的均值
"rpn_target_stds": [1.0, 1.0, 1.0, 1.0],                                 # 边界框编解码标准

# bbox_assign_sampler
"neg_iou_thr": 0.3,                                                      # 交并比后的负样本阈值
"pos_iou_thr": 0.7,                                                      # 交并比后的正样本阈值
"min_pos_iou": 0.3,                                                      # 交并比后的最小正样本阈值
"num_bboxes": 245520,                                                    # 边界框总数
"num_gts": 128,                                                          # 总地面真值数
"num_expected_neg": 256,                                                 # 负样本数
"num_expected_pos": 128,                                                 # 正样本数

# 候选
"activate_num_classes": 2,                                               # RPN分类数
"use_sigmoid_cls": True,                                                 # 是否在RPN分类中使用sigmoid损失函数

# roi_alignj
"roi_layer": dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2), # ROIAlign参数
"roi_align_out_channels": 256,                                                  # ROIAlign输出通道大小
"roi_align_featmap_strides": [4, 8, 16, 32],                                    # 不同级别ROIAlign特征图的步长大小
"roi_align_finest_scale": 56,                                                   # ROIAlign的最佳比例
"roi_sample_num": 640,                                                          # ROIAlign层中的样本数

# bbox_assign_sampler_stage2                                                    # 边界框为第二阶段分配样本，参数含义与bbox_assign_sampler类似
"neg_iou_thr_stage2": 0.5,
"pos_iou_thr_stage2": 0.5,
"min_pos_iou_stage2": 0.5,
"num_bboxes_stage2": 2000,
"num_expected_pos_stage2": 128,
"num_expected_neg_stage2": 512,
"num_expected_total_stage2": 512,

# rcnn                                                                          # 第二阶段的RCNN参数，参数含义与FPN类似
"rcnn_num_layers": 2,
"rcnn_in_channels": 256,
"rcnn_fc_out_channels": 1024,
"rcnn_mask_out_channels": 256,
"rcnn_loss_cls_weight": 1,
"rcnn_loss_reg_weight": 1,
"rcnn_loss_mask_fb_weight": 1,
"rcnn_target_means": [0., 0., 0., 0.],
"rcnn_target_stds": [0.1, 0.1, 0.2, 0.2],

# 训练候选
"rpn_proposal_nms_across_levels": False,
"rpn_proposal_nms_pre": 2000,                                                  # RPN中NMS之前的候选区域数
"rpn_proposal_nms_post": 2000,                                                 # RPN中NMS之后的候选区域数
"rpn_proposal_max_num": 2000,                                                  # RPN中的最大候选区域数
"rpn_proposal_nms_thr": 0.7,                                                   # RPN中的NMS阈值
"rpn_proposal_min_bbox_size": 0,                                               #RPN中框的最小值

# test proposal                                                                # 部分参数与训练候选相似
"rpn_nms_across_levels": False,
"rpn_nms_pre": 1000,
"rpn_nms_post": 1000,
"rpn_max_num": 1000,
"rpn_nms_thr": 0.7,
"rpn_min_bbox_min_size": 0,
"test_score_thr": 0.05,                                                        # 分数阈值
"test_iou_thr": 0.5,                                                           # 交并比阈值
"test_max_per_img": 100,                                                       # 最大实例数
test_batch_size": 2,                                                          # batch size

"rpn_head_use_sigmoid": True,                                                  # 在RPN中是否使用sigmoid
"rpn_head_weight": 1.0,                                                        # RPN头损失权重
"mask_thr_binary": 0.5,                                                        # RCNN中的掩码阈值

# 学习率
"base_lr": 0.02,                                                               # 基准学习率
"base_step": 58633,                                                            # 学习率生成器中的基准步长
"total_epoch": 13,                                                             # 学习率生成器的总epochs数
"warmup_step": 500,                                                            # 学习率生成器中的预热步骤
"warmup_ratio": 1/3.0,                                                         # 预热比例
"sgd_momentum": 0.9,                                                           # 优化器动量

# 训练
"batch_size": 2,
"loss_scale": 1,
"momentum": 0.91,
"weight_decay": 1e-4,
"pretrain_epoch_size": 0,                                                      # 预训练epoch size
"epoch_size": 12,                                                              # 总epoch size
"save_checkpoint": True,                                                       # 是否保存检查点
"save_checkpoint_epochs": 1,                                                   # 保存检查点间隔
"keep_checkpoint_max": 12,                                                     # 保存的检查点的最大数量
"save_checkpoint_path": "./",                                                  # 检查点路径

"mindrecord_dir": "/home/maskrcnn/MindRecord_COCO2017_Train",                  # mindrecord文件路径
"coco_root": "/home/maskrcnn/",                                                # coco根数据集的路径
"train_data_type": "train2017",                                                # 训练数据集
"val_data_type": "val2017",                                                    # 评估数据集
"val_data_type": "val2017",                                                    # 标注
"coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush'),
"num_classes": 81
```

## [训练过程](#目录)

- 在`default_config.yaml`中设置选项，包括损失缩放、学习率和网络超参。有关数据集的更多信息，请单击[此处](https://www.mindspore.cn/tutorials/en/master/advanced/dataset.html) for more information about dataset.。

### [训练](#目录)

- 在Ascend上运行`run_standalone_train.sh`，进行maskrcnn_mobilenetv1模型的非分布式训练。

    ```bash
    # 单机训练
    bash run_standalone_train.sh [DATA_PATH] [PRETRAINED_CKPT(optional)]
    # 示例：bash run_standalone_train.sh /home/DataSet/cocodataset/
    ```

- 在CPU上运行`run_standalone_train_cpu.sh`，进行maskrcnn_mobilenetv1模型的非分布式训练。

    ```bash
    # 单机训练
    bash run_standalone_train_cpu.sh [PRETRAINED_MODEL](optional)
    ```

### [分布式训练](#目录)

- 在Ascend上运行`run_distribute_train.sh`进行Mask模型的分布式训练。

    ```bash
    bash run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [PRETRAINED_MODEL(optional)]
    # 示例：bash run_distribute_train.sh ~/hccl_8p.json /home/DataSet/cocodataset/
    ```

> 运行分布式训练时，需要使用RANK_TABLE_FILE指定的hccl.json。您可以使用[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)来生成该文件。
> 至于PRETRAINED_MODEL，如果未设置，模型将从头开始训练。暂无预训练模型可用，请持续关注。
> 绑核操作取决于`device_num`参数值及处理器总数。如果不需要，删除`scripts/run_distribute_train.sh`脚本中的`taskset`操作任务集即可。

- 在GPU上运行`run_distribute_train_gpu.sh`进行Mask模型的分布式训练。

    ```bash
    bash run_distribute_train_gpu.sh [DATA_PATH] [PRETRAINED_PATH]
    ```

### [训练结果](#目录)

训练结果存储在示例路径中，文件夹名称以"train"或"train_parallel"开头。您可以在Los_rankid.log中查看检查点文件及如下结果。

```log
# 分布式训练示例（8卡）
2123 epoch: 1 step: 7393 ,rpn_loss: 0.24854, rcnn_loss: 1.04492, rpn_cls_loss: 0.19238, rpn_reg_loss: 0.05603, rcnn_cls_loss: 0.47510, rcnn_reg_loss: 0.16919, rcnn_mask_loss: 0.39990, total_loss: 1.29346
3973 epoch: 2 step: 7393 ,rpn_loss: 0.02769, rcnn_loss: 0.51367, rpn_cls_loss: 0.01746, rpn_reg_loss: 0.01023, rcnn_cls_loss: 0.24255, rcnn_reg_loss: 0.05630, rcnn_mask_loss: 0.21484, total_loss: 0.54137
5820 epoch: 3 step: 7393 ,rpn_loss: 0.06665, rcnn_loss: 1.00391, rpn_cls_loss: 0.04999, rpn_reg_loss: 0.01663, rcnn_cls_loss: 0.44458, rcnn_reg_loss: 0.17700, rcnn_mask_loss: 0.38232, total_loss: 1.07056
7665 epoch: 4 step: 7393 ,rpn_loss: 0.14612, rcnn_loss: 0.56885, rpn_cls_loss: 0.06186, rpn_reg_loss: 0.08429, rcnn_cls_loss: 0.21228, rcnn_reg_loss: 0.08105, rcnn_mask_loss: 0.27539, total_loss: 0.71497
...
16885 epoch: 9 step: 7393 ,rpn_loss: 0.07977, rcnn_loss: 0.85840, rpn_cls_loss: 0.04395, rpn_reg_loss: 0.03583, rcnn_cls_loss: 0.37598, rcnn_reg_loss: 0.11450, rcnn_mask_loss: 0.36816, total_loss: 0.93817
18727 epoch: 10 step: 7393 ,rpn_loss: 0.02379, rcnn_loss: 1.20508, rpn_cls_loss: 0.01431, rpn_reg_loss: 0.00947, rcnn_cls_loss: 0.32178, rcnn_reg_loss: 0.18872, rcnn_mask_loss: 0.69434, total_loss: 1.22887
20570 epoch: 11 step: 7393 ,rpn_loss: 0.03967, rcnn_loss: 1.07422, rpn_cls_loss: 0.01508, rpn_reg_loss: 0.02461, rcnn_cls_loss: 0.28687, rcnn_reg_loss: 0.15027, rcnn_mask_loss: 0.63770, total_loss: 1.11389
22411 epoch: 12 step: 7393 ,rpn_loss: 0.02937, rcnn_loss: 0.85449, rpn_cls_loss: 0.01704, rpn_reg_loss: 0.01234, rcnn_cls_loss: 0.20667, rcnn_reg_loss: 0.12439, rcnn_mask_loss: 0.52344, total_loss: 0.88387
```

## [评估过程](#目录)

### [评估](#目录)

- 运行`run_eval.sh`进行评估。

    ```bash
    # 推理
    bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    # 示例：bash run_eval.sh /home/DataSet/cocodataset/annotations/instances_val2017.json /home/model/maskrcnn_mobilenetv1/ckpt/mask_rcnn-5_7393.ckpt /home/DataSet/cocodataset/ "Ascend" 0
    ```

> 对于COCO2017数据集，VALIDATION_ANN_FILE_JSON指的是数据集目录中的annotations/instances_val2017.json。
> 可以在训练过程中生成和保存检查点，文件夹名称以"train/checkpoint"或"train_parallel*/checkpoint"开头。

- 在GPU上运行`run_eval.sh`进行评估。

    ```bash
    # 推理
    bash run_eval.sh [ANN_FILE] [CHECKPOINT_PATH] [DATA_PATH] [DEVICE_TARGET] [DEVICE_ID]
    # 示例：bash run_eval.sh /home/DataSet/cocodataset/annotations/instances_val2017.json /home/model/maskrcnn_mobilenetv1/ckpt/mask_rcnn-12_7393.ckpt /home/DataSet/cocodataset/ "GPU" 0
    ```

### [评估结果](#目录)

推理结果存储在示例路径中，文件夹名称为“eval”。在此目录下，您可以在日志中查看以下结果。

```log
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.388
```

## [导出过程](#目录)

### [导出](#目录)

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT`取值为["AIR", "MINDIR"]。

## [推理过程](#目录)

**推理前，请参考[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)设置环境变量。**

### [推理](#目录)

在进行推理之前，我们需要先导出模型。AIR模型只能在Ascend 910环境中导出，mindir模型可以在任何环境中导出。
当前batch_ Size只能设置为1。推理过程需要600G左右的硬盘空间来保存推理结果。

```shell
# Ascend 310上运行推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [ANN_FILE] [DEVICE_ID]
```

### [推理结果](#目录)

推理结果保存在当前路径中，您可以在acc.log文件中查看。

```log
Evaluate annotation type *bbox*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501

Evaluate annotation type *segm*
Accumulating evaluation results...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.388
```

# 模型描述

## 性能

### 评估性能

| 参数                | Ascend                                                     | GPU                                                        |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| 模型版本             | V1                                                         | V1                                                         |
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；操作系统EulerOS 2.8| GPU(Tesla V100-PCIE); CPU 2.60 GHz, 26核；内存790G；操作系统EulerOS 2.0            |
| 上传日期             | 12/01/2020                                | 09/21/2021                                |
| MindSpore版本         | 1.0.0                                                      | 1.5.0                                                      |
| 数据集                   | COCO2017                                                   | COCO2017                                                   |
| 训练参数       | epoch=12,  batch_size = 2                                   | epoch=12,  batch_size = 2                                   |
| 优化器                 | 动量                                                   | 动量                                                        |
| 损失函数             | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  |
| 输出                    | 概率                                                | 概率                                                |
| 损失                      | 0.88387                                                     | 0.60763                                                     |
| 速度                     | 8卡: 249 ms/step                                           | 8卡: 795.645 ms/step                                      |
| 总时长                | 8卡：6.23小时                                           | 8卡：19.6小时                                           |
| 脚本                   | [Mask R-CNN脚本](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |  [Mask R-CNN脚本](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1)|

### 推理性能

| 参数         | Ascend                     |
| ------------------- | --------------------------- |
| 模型版本      | V1                         |
| 资源           | Ascend 910；操作系统EulerOS 2.8                  |
| 上传日期      | 12/01/2020|
| MindSpore版本  | 1.0.0                      |
| 数据集            | COCO2017                   |
| batch_size         | 2                           |
| 输出            | mAP                        |
| 准确率           | IoU=0.50:0.95（边界框22.7%，掩码17.6%）|
| 推理模型| 107M（.ckpt文件）          |

# [随机情况说明](#目录)

在`dataset.py`中，我们设置了`create_dataset`函数内的种子，我们还在train.py中使用随机种子进行权重初始化。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
