# 目录

- [DeepText描述](#deeptext描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
        - [评估](#评估)
        - [ONNX评估](#onnx评估)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [DeepText描述](#目录)

DeepText是一种卷积神经网络架构，用于非特定场景下的文本检测。DeepText系统基于Faster R-CNN的优雅框架。这一想法出自2017年发表的论文《DeepText: A new approach for text proposal generation and text detection in natural images.》。

[论文](https://arxiv.org/pdf/1605.07314v1.pdf) Zhuoyao Zhong, Lianwen Jin, Shuangping Huang, South China University of Technology (SCUT), Published in ICASSP 2017.

# [模型架构](#目录)

InceptionV4的整体网络架构如下：

[链接](https://arxiv.org/pdf/1605.07314v1.pdf)

# [数据集](#目录)

我们使用了4个数据集进行训练，1个数据集用于评估。

- 数据集1：[ICDAR 2013: Focused Scene Text](https://rrc.cvc.uab.es/?ch=2&com=tasks)：
    - 训练集：142 MB，229张图像
    - 测试集：110 MB，233张图像
- 数据集2：[ICDAR 2011: Born-Digital Images](https://rrc.cvc.uab.es/?ch=1&com=tasks)：
    - 训练集：27.7 MB，410张图像
- 数据集3：[SCUT-FORU: Flickr OCR Universal Database](https://github.com/yan647/SCUT_FORU_DB_Release)：
    - 训练集：388 MB，1715张图像
- 数据集4：[CocoText v2（MSCCO2017的子集）](https://rrc.cvc.uab.es/?ch=5&com=tasks)：
    - 训练集：13 GB，63686张图像

# [特性](#目录)

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```shell
.
└─deeptext
  ├─README.md
  ├─ascend310_infer                     # Ascend 310推理应用
  ├─model_utils
    ├─__init__.py                       # 初始化包文件
    ├─config.py                         # 解析参数
    ├─device_adapter.py                 # ModelArts的设备适配器
    ├─local_adapter.py                  # 本地适配器
    └─moxing_adapter.py                 # ModelArts的moxing适配器
  ├─scripts
    ├─run_standalone_train_ascend.sh    # 在Ascend平台启动单机训练（单卡）
    ├─run_standalone_train_gpu.sh       # 在GPU平台启动单机训练（单卡）
    ├─run_distribute_train_ascend.sh    # 在Ascend平台启动分布式训练（8卡）
    ├─run_distribute_train_gpu.sh       # 在GPU平台启动分布式训练（8卡）
    ├─run_infer_310.sh                  # Ascend 310推理的shell脚本
    ├─run_eval_gpu.sh                   # 在GPU平台启动评估
    └─run_eval_ascend.sh                # 在Ascend平台启动评估
    └─run_eval_onnx.sh                  # 启动ONNX模型评估
  ├─src
    ├─DeepText
      ├─__init__.py                     # 初始化包文件
      ├─anchor_genrator.py              # Anchor生成器
      ├─bbox_assign_sample.py           # 阶段1的候选区域层
      ├─bbox_assign_sample_stage2.py    # 阶段2的候选区域层
      ├─deeptext_vgg16.py               # 主网络定义
      ├─proposal_generator.py           # 候选区域生成器
      ├─rcnn.py                         # RCNN
      ├─roi_align.py                    # roi_align cell wrapper
      ├─rpn.py                          # 区域候选网络
      └─vgg16.py                        # 骨干
    ├─aipp.cfg                        # aipp配置文件
    ├─dataset.py                      # 数据预处理
    ├─lr_schedule.py                  # 学习率调度器
    ├─network_define.py               # 网络定义
    └─utils.py                        # 常用函数
  ├─default_config.yaml               # 配置
  ├─eval.py                           # eval网络
  ├─eval_onnx.py                      # Eval ONNX模型
  ├─export.py                         # 导出检查点，支持.onnx, .air, .mindir转换
  ├─postprogress.py                   # Ascend 310推理后处理
  └─train.py                          # 训练网络
```

## [训练过程](#目录)

### 用法

- Ascend处理器：

    ```bash
    # 分布式训练示例（8卡）
    bash run_distribute_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [RANK_TABLE_FILE] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH]
    # 单机训练
    bash run_standalone_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
    # 评估：
    bash run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
    ```

- GPU：

    ```bash
    # 分布式训练示例（8卡）
    sh run_distribute_train_gpu.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH]
    # 单机训练
    sh run_standalone_train_gpu.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
    # 评估：
    sh run_eval_gpu.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
    ```

> 注：
> RANK_TABLE_FILE文件，请参考[链接](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)。如需获取设备IP，请点击[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。对于InceptionV4等大模型，最好导出外部环境变量`export HCCL_CONNECT_TIMEOUT=600`，将hccl连接检查时间从默认的120秒延长到600秒。否则，连接可能会超时，因为随着模型增大，编译时间也会增加。
>
> 处理器绑核操作取决于`device_num`和总处理器数。如果不希望这样做，请删除`scripts/run_distribute_train.sh`中的`taskset`操作。
>
> `pretrained_path`应该是在Imagenet2012上训练的vgg16的检查点。dict中的权重名称应该完全一样，且在vgg16训练中启用batch_norm，否则将导致后续步骤失败。
> 有关COCO_TEXT_PARSER_PATH coco_text.py文件，请参考[链接](https://github.com/andreasveit/coco-text)。
>

- ModelArts（如果想在ModelArts中运行，请查看[ModelArts官方文档](https://support.huaweicloud.com/modelarts/)，并按照以下方式开始训练）

    ```bash
    # ModelArts上8卡训练
    # (1)将[COCO_TEXT_PARSER_PATH]文件复制到/CODE_PATH/deeptext/src/
    # （2）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"imgs_path='/YOUR IMGS_PATH/'"。
    #          在default_config.yaml文件中设置"annos_path='/YOUR ANNOS_PATH/'"。
    #          在default_config.yaml文件中设置"run_distribute=True"。
    #          在default_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_pretrain/'"。
    #          在default_config.yaml文件中设置"pre_trained='/cache/checkpoint_path/YOUR PRETRAINED_PATH/'"。
    #          在default_config.yaml文件中设置"mindrecord_dir='/cache/data/deeptext_dataset/mindrecord'"。
    #          在default_config.yaml文件中设置"coco_root='/cache/data/deeptext_dataset/coco2017'"。
    #          在default_config.yaml文件中设置"cocotext_json='/cache/data/deeptext_dataset/cocotext.v2.json'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"imgs_path=/YOUR IMGS_PATH/"。
    #          在网页上添加"annos_path=/YOUR ANNOS_PATH/"。
    #          在网页上添加"run_distribute=True"。
    #          在网页上上添加"checkpoint_url='s3://dir_to_your_pretrain/'"。
    #          在网页上上添加"pre_trained=/cache/checkpoint_path/YOUR PRETRAINED_PATH/"。
    #          在网页上上添加"mindrecord_dir=/cache/data/deeptext_dataset/mindrecord"。
    #          在网页上上添加"coco_root=/cache/data/deeptext_dataset/coco2017"。
    #          在网页上上添加"cocotext_json=/cache/data/deeptext_dataset/cocotext.v2.json"。
    #          在网页上添加其他参数。
    # （3）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
    # （4）在网页上上设置代码目录为"/path/deeptext"。
    # （5）在网页上上设置启动文件为"train.py"。
    # （6）在网页上上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （7）创建作业。
    #
    # ModelArts上运行单卡训练
    # (1)将[COCO_TEXT_PARSER_PATH]文件复制到/CODE_PATH/deeptext/src/
    # （2）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"imgs_path='/YOUR IMGS_PATH/'"。
    #          在default_config.yaml文件中设置"annos_path='/YOUR ANNOS_PATH/'"。
    #          在default_config.yaml文件中设置"run_distribute=False"。
    #          在default_config.yaml文件中设置"checkpoint_url='s3://dir_to_your_pretrain/'"。
    #          在default_config.yaml文件中设置"pre_trained='/cache/checkpoint_path/YOUR PRETRAINED_PATH/'"。
    #          在default_config.yaml文件中设置"mindrecord_dir='/cache/data/deeptext_dataset/mindrecord'"。
    #          在default_config.yaml文件中设置"coco_root='/cache/data/deeptext_dataset/coco2017'"。
    #          在default_config.yaml文件中设置"cocotext_json='/cache/data/deeptext_dataset/cocotext.v2.json'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"imgs_path=/YOUR IMGS_PATH/"。
    #          在网页上添加"annos_path=/YOUR ANNOS_PATH/"。
    #          在网页上添加"run_distribute=False"。
    #          在网页上上添加"checkpoint_url='s3://dir_to_your_pretrain/'"。
    #          在网页上上添加"pre_trained=/cache/data/YOUR PRETRAINED_PATH/"。
    #          在网页上上添加"mindrecord_dir=/cache/data/deeptext_dataset/mindrecord"。
    #          在网页上上添加"coco_root=/cache/data/deeptext_dataset/coco2017"。
    #          在网页上上添加"cocotext_json=/cache/data/deeptext_dataset/cocotext.v2.json"。
    #          在网页上添加其他参数。
    # （3）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
    # （4）在网页上上设置代码目录为"/path/deeptext"。
    # （5）在网页上上设置启动文件为"train.py"。
    # （6）在网页上上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （7）创建作业。
    #
    # ModelArts上运行单卡评估
    # (1)将[COCO_TEXT_PARSER_PATH]文件复制到/CODE_PATH/deeptext/src/
    # （2）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"imgs_path='/YOUR IMGS_PATH/'"。
    #          在default_config.yaml文件中设置"annos_path='/YOUR ANNOS_PATH/'"。
    #          在default_config.yaml文件中设置"checkpoint_url='s3://dir_to_trained_model/'"。
    #          在default_config.yaml文件中设置"checkpoint_path='/cache/checkpoint_path/YOUR CHECKPOINT_PATH/'"。
    #          在default_config.yaml文件中设置"mindrecord_dir='/cache/data/deeptext_dataset/mindrecord'"。
    #          在default_config.yaml文件中设置"coco_root='/cache/data/deeptext_dataset/coco2017'"。
    #          在default_config.yaml文件中设置"cocotext_json='/cache/data/deeptext_dataset/cocotext.v2.json'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上添加"imgs_path=/YOUR IMGS_PATH/"。
    #          在网页上添加"annos_path=/YOUR ANNOS_PATH/"。
    #          在网页上上添加"checkpoint_url='s3://dir_to_trained_model/'"。
    #          在网页上上添加"checkpoint_path=/cache/checkpoint_path/YOUR CHECKPOINT_PATH/"。
    #          在网页上上添加"mindrecord_dir=/cache/data/deeptext_dataset/mindrecord"。
    #          在网页上上添加"coco_root=/cache/data/deeptext_dataset/coco2017"。
    #          在网页上上添加"cocotext_json=/cache/data/deeptext_dataset/cocotext.v2.json"。
    #          在网页上添加其他参数。
    # （3）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
    # （4）在网页上上设置代码目录为"/path/deeptext"。
    # （5）在网页上上设置启动文件为"eval.py"。
    # （6）在网页上上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （7）创建作业。
    #
    # ModelArts上单卡导出
    # （1）执行a或b。
    #       a. 在default_config.yaml文件中设置"enable_modelarts=True"。
    #          在default_config.yaml文件中设置"checkpoint_url='s3://dir_to_trained_model/'"。
    #          在default_config.yaml文件中设置"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在default_config.yaml文件中设置"device_target=Ascend"。
    #          在default_config.yaml文件上设置"file_format='MINDIR'"。
    #          在default_config.yaml文件中设置其他参数。
    #       b. 在网页上添加"enable_modelarts=True"。
    #          在网页上上添加"checkpoint_url='s3://dir_to_trained_model/'"。
    #          在网页上添加"ckpt_file='/cache/checkpoint_path/model.ckpt'"。
    #          在网页上添加"device_target=Ascend"。
    #          在网页上添加"file_format='MINDIR'"。
    #          在网页上添加其他参数。
    # （2）在网页上设置代码目录为"/path/deeptext"。
    # （3）在网页上设置启动文件为"export.py"。
    # （4）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
    # （5）创建作业。
    ```

### 启动

```bash
# 训练示例
  shell:
    Ascend处理器：
      # 分布式训练示例（8卡）
      bash run_distribute_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [RANK_TABLE_FILE] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH]
      # 单机训练
      bash run_standalone_train_ascend.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
```

### 结果

训练结果将存储在示例路径中。默认情况下，检查点将存储在`ckpt_path`中，训练日志将重定向到`./log`，损失也将重定向到`./loss_0.log`，如下所示。

```python
469 epoch: 1 step: 982 ,rpn_loss: 0.03940, rcnn_loss: 0.48169, rpn_cls_loss: 0.02910, rpn_reg_loss: 0.00344, rcnn_cls_loss: 0.41943, rcnn_reg_loss: 0.06223, total_loss: 0.52109
659 epoch: 2 step: 982 ,rpn_loss: 0.03607, rcnn_loss: 0.32129, rpn_cls_loss: 0.02916, rpn_reg_loss: 0.00230, rcnn_cls_loss: 0.25732, rcnn_reg_loss: 0.06390, total_loss: 0.35736
847 epoch: 3 step: 982 ,rpn_loss: 0.07074, rcnn_loss: 0.40527, rpn_cls_loss: 0.03494, rpn_reg_loss: 0.01193, rcnn_cls_loss: 0.30591, rcnn_reg_loss: 0.09937, total_loss: 0.47601
```

## [评估过程](#目录)

### 用法

您可以使用python或shell脚本开始训练。shell脚本的用法如下：

- Ascend处理器：

    ```bash
    bash run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
    ```

### 启动

```bash
# 评估示例
  shell:
      Ascend处理器：
            bash run_eval_ascend.sh [IMGS_PATH] [ANNOS_PATH] [CHECKPOINT_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]
```

> 在训练过程中可能会产生检查点。

### 结果

评估结果将存储在示例路径中，您可以在`log`中查看如下结果。

```python
========================================

class 1 precision is 88.01%, recall is 82.77%
```

GPU评估结果如下：

```python
========================================

class 1 precision is 84.49%, recall is 88.28%
```

## 模型导出

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT`：取值范围["AIR", "MINDIR"]。

## 推理过程

### 用法

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

在执行推理之前，必须在Ascend 910环境上通过导出脚本导出AIR文件。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DEVICE_ID]
```

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中查看如下结果。

```python
========================================

class 1 precision is 84.24%, recall is 87.40%, F1 is 85.79%
```

### [ONNX评估](#目录)

- 将模型导出到ONNX：

  ```shell
  python export.py --device_target GPU --ckpt_file /path/to/deeptext.ckpt --export_device_target GPU --file_name deeptext --file_format ONNX --export_batch_size 2
  ```

- 运行ONNX评估脚本：

  ```shell
  bash scripts/run_eval_onnx.sh [IMGS_PATH] [ANNOS_PATH] [ONNX_MODEL] [MINDRECORD_DIR] [<DEVICE_TARGET>]
  ```

  评估结果将保存在log文件中，格式如下：

  > class 1 precision is 84.04%, recall is 87.51%,F1 is 85.74%

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend                                                                                              | GPU                           |
| -------------------------- | --------------------------------------------------------------------------------------------------- |--------------------------------------- |
| 模型版本             | Deeptext                                                                                            | Deeptext                       |
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8                                        | Tesla V100 PCIe 32GB；CPU 2.70GHz, 52核；内存1510G；操作系统Ubuntu 18.04.5       |
| 上传日期             | 12/26/2020                                                                                          | 7/29/2021    |
| MindSpore版本         | 1.1.0                                                                                               | 1.3.0                        |
| 数据集                   | 66040张图像                                                                                       | 66040张图像                |
| Batch_size                 | 2                                                                                                   | 2                        |
| 训练参数       | src/config.py                                                                                       | src/config.py            |
| 优化器                 | 动量                                                                                           | 动量            |
| 损失函数             | SoftmaxCrossEntropyWithLogits用于分类, SmoothL2Loss用于bbox回归                 | SoftmaxCrossEntropyWithLogits用于分类, SmoothL2Loss用于bbox回归 |
| 损失                      | ~0.008                                                                                              | ~0.116               |
| 总时长（8卡）           | 4h                                                                                                  | 9h                   |
| 脚本                   | [deeptext](https://gitee.com/mindspore/models/tree/master/official/cv/DeepText) | [deeptext](https://gitee.com/mindspore/models/tree/master/official/cv/DeepText)  |

#### 推理性能

| 参数         | Ascend                                                       | GPU                       |
| ------------------- | -------------------------------------------------------------| --------------------------- |
| 模型版本      | Deeptext                                                     | Deeptext
| 资源           | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8 | Tesla V100 PCIe 32GB；CPU 2.70GHz, 52核；内存1510G；操作系统Ubuntu 18.04.5  |
| 上传日期      | 12/26/2020                                                   | 7/29/2021   |
| MindSpore版本  | 1.1.0                                                        | 1.3.0                         |
| 数据集            | 229张图像                                                  | 229张图像             |
| Batch_size          | 2                                                            | 2                       |
| 准确率           | F1 score：84.50%                                           | F1 score：86.34%      |
| 总时长         | 1 min                                                        | 1 min                   |
| 推理模型| 3492M（.ckpt）                                          | 3492M（.ckpt）          |

#### 训练性能结果

| **Ascend** | 训练性能|
| :--------: | :---------------: |
|     单卡    |     14 img/s      |

| **Ascend** | 训练性能|
| :--------: | :---------------: |
|     8卡     |     50 img/s      |

|   **GPU**   |  训练性能 |
| :---------: | :---------------: |
|     单卡     |     5 img/s       |

|   **GPU**   |  训练性能 |
| :---------: | :-----------------: |
|     8卡      |     25 img/s     |

# [随机情况说明](#目录)

我们在train.py中将种子设置为1。

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
