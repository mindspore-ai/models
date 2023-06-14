# 目录

<!-- TOC -->

- [目录](#目录)
- [Semantic Human Matting描述](#semantic-human-matting描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [下载数据集](#下载数据集)
    - [制作数据集](#制作数据集)
    - [获取并转换torch网络的权重文件](#获取并转换torch网络的权重文件)
    - [训练](#训练)
    - [评估](#评估)
    - [昇腾310推理](#昇腾310推理)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
        - [数据集制作配置](#数据集制作配置)
        - [训练配置](#训练配置)
        - [测试、推理、模型导出配置](#测试推理模型导出配置)
    - [数据集制作过程](#数据集制作过程)
        - [数据集制作](#数据集制作)
    - [获取初始化权重文件过程](#获取初始化权重文件过程)
        - [获取初始化权重](#获取初始化权重)
    - [训练过程](#训练过程)
        - [训练](#训练-1)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估-1)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Semantic Human Matting描述

**Semantic Human Matting（SHM）** 是阿里巴巴开创的一种全自动的人物抠图方法，是第一种全自动抠图算法，提出了一种新的融合策略，可以学习将语义信息和高质量细节与深层网络联合起来。 它可以自适应地在每个像素上集成粗略的语义和细节结果，这对于实现端到端训练至关重要。同时，创建了大规模高质量的人像数据集，它包含35,513个具有相应alpha配对的人类图像。该数据集不仅能够在SHM中对深层网络进行有效的训练，而且还有助于其在人像抠图上面的研究，但数据集未开源。

由于作者未开源源代码及数据集，该网络参考自**lizhengwei1992**的代码实现 https://github.com/lizhengwei1992/Semantic_Human_Matting ， 及**zzZ_CMing**的[【SHM】Semantic Human Matting抠图算法调试](https://blog.csdn.net/zzZ_CMing/article/details/109490676)

论文[Semantic Human Matting](https://arxiv.org/pdf/1809.01354.pdf): Quan Chen, Tiezheng Ge, Yanyu Xu, Zhiqiang Zhang, Xinxin Yang, Kun Gai.

# 模型架构

**SHM**是由**T-Net**、**M-Net**和**Fusion Module**三部分组成:
**T-Net**：采用的是MobileNetV2+Unet，输出是3-channel的特征图，采用的是语义分割中的方法，分别表示了每个像素属于各自类别的概率。
**M-Net**：是编码-解码网络（结构和论文稍有改变）。encoder网络有4个卷积层和4个max-pooling层；decoder网络有4个卷积层和4个转置卷积层。M-Net在每个卷积层后加入Batch Normalization层（转置卷积除外），以加速收敛。
**Fusion Module**：直接对T-Net和M-Net的输出进行融合，生成最终的alpha matte结果。

# 数据集

使用的数据集：[Matting Human Datasets](<https://github.com/aisegmentcn/matting_human_datasets>)

- 数据集大小：28.7GB，包含137706张图像和对应的matting结果图

  ```text
  ├── archive
      ├── clip_img                        // 半身人像图像
      ├── matting                         // clip_img对应的matting图像
      ├── matting_human_half              // 目录包含clip_img和matting子目录（未使用）
  ```

- 数据格式：RBG图片文件

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html) 的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索`reduce precision`查看精度降低的算子。

# 环境要求

- 硬件（昇腾处理器）
    - 使用昇腾处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

执行前的具体配置，请参看[脚本参数](#脚本参数)

## 下载数据集

  下载数据集[Matting Human Datasets](<https://github.com/aisegmentcn/matting_human_datasets>)，并解压到`/cache`目录下，修改名称为`human_matting`，目录和文件夹名称可自定义。

## 制作数据集

  执行前需配置`config.yaml`下的`generate_data`字段，详情请参考[数据集制作过程](#数据集制作过程)

  ```text
  # 执行以生成数据集
  python3 generate_datasets.py --yaml_path=../config.yaml
  ```

## 获取并转换torch网络的权重文件

  1. 从github获取torch版源代码： https://github.com/lizhengwei1992/Semantic_Human_Matting

  2. 将`src`目录下的`get_init_weight.py`文件拷贝到`torch版源代码的根目录`下，并执行以下命令

  ```text
  python3 get_init_weight.py
  ```

  3. 执行命令后，会在`torch版源代码根目录`生成`init_weight.ckpt`初始化权重文件，可将**初始化权重文件**拷贝到该**项目根目录**下

  详情请参看[获取初始化权重文件过程](#获取初始化权重文件过程)

## 训练

  执行前需配置`config.yaml`下的`pre_train_t_net`、`pre_train_m_net`、`train_phase`字段以及其他字段，详情参考[脚本参数](#脚本参数)下的`训练配置`和[训练过程](#训练过程)章节

- 单卡训练

  ```text
  # 运行训练示例
  python3 train.py --yaml_path=[YAML_PATH] --data_url=[DATASETS] --train_url=[OUTPUT] --init_weight=[INIT_WEIGHT][OPTIONAL] > train.log 2>&1 &
  # example:  python3 train.py --yaml_path=./config.yaml --data_url=/cache/datasets --train_url=/cache/output --init_weight=./init_weight.ckpt > train.log 2>&1 &
  ```

- 分布式训练

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。可将文件命名为`hccl_8p.json`，并存放在当前工程的根目录下。
  请遵循以下链接中的说明：<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

  ```text
  # 运行分布式训练示例
  bash run_train.sh [RANK_TABLE_FILE] [YAML_PATH] [DATASETS] [OUTPUT] [INIT_WEIGHT][OPTIONAL]
  # example: bash run_train.sh ../hccl_8p.json ../config.yaml /cache/datasets /cache/output ../init_weight.ckpt
  ```

## 评估

  执行前需配置`config.yaml`下的`test`字段，详情参考[评估过程](#评估过程)

  ```text
  # 运行评估示例
  python3 eval.py --yaml_path=[YAML_PATH] > eval.log 2>&1 &
  # example: python3 eval.py --yaml_path=./config.yaml > eval.log 2>&1 &
  ```

  或者

  ```text
  bash run_eval.sh [DEVICE_ID][OPTIONAL]
  # example: bash run_eval.sh
  ```

## 昇腾310推理

- **模型导出**

    在模型导出之前需要修改`config.yaml`中的`export`字段下面的参数，以及配置模型导出的相关参数，当前导出格式为`MINDIR`。更多配置请参考[脚本参数](#脚本参数)，详细导出过程请参考[导出过程](#导出过程)

    ```text
    python3 export.py --config_path=[CONFIG_PATH]
    # example: python3 export.py --config_path=./config.yaml
    ```

- **运行推理**

    执行前需要修改`config.yaml`中的`infer`字段下面的参数，其中`file_test_list`为绝对路径，是`制作数据集`步骤中生成的`test.txt`，`size`固定为320 。更多配置请参考[脚本参数](#脚本参数)，详细导出过程请参考[推理过程](#推理过程)

    ```text
    # 运行推理示例
    bash run_infer_310.sh [MINDIR_PATH]
    # [MINDIR_PATH]：上个步骤配置的导出模型文件路径
    # example: bash run_infer_310.sh ../shm_export.mindir
    ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                        // 所有模型相关说明
    ├── SemanticHumanMatting
        ├── README.md                    // SHM相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_train.sh              // 分布式到Ascend的shell脚本
        │   ├──run_eval.sh               // Ascend评估的shell脚本
        │   ├──run_infer_310.sh          // Ascend推理shell脚本
        ├── src
        │   ├──model
        │   │  ├──T_Net.py               // T网络
        │   │  ├──M_Net.py               // M网络
        │   │  ├──network.py             // 端到端网络
        │   ├──dataset.py                // 数据集加载
        │   ├──loss.py                   // loss
        │   ├──metric.py                 // 指标
        │   ├──config.py                 // 配置解析
        │   ├──callback.py               // 回调函数
        │   ├──generate_datasets.py      // 数据集制作
        │   ├──load_model.py             // 加载模型
        │   ├──get_init_weight.py        // 获取初始化权重文件
        ├── train.py                     // 训练脚本
        ├── eval.py                      // 评估脚本
        ├── export.py                    // 将checkpoint文件导出到air/mindir
        ├── preprocess.py                // 310推理前处理脚本
        ├── postprogress.py              // 310推理后处理脚本
        ├── config.yaml                  // 配置文件
```

## 脚本参数

### 数据集制作配置

数据集制作的相关配置放置在`config.yaml`文件下的`generate_data`字段，可进行的配置有：

1. 制作完成后的保存路径：`path_save`

2. 方便调试用的调试数据集路径： `path_debug`

3. 训练集、验证集、测试集划分比例： `proportion`，当前使用的划分比例是`6:2:2`，用户可根据需要进行调整

4. 调试数据集的生成数量：`debug_pic_nums`

5. 首次运行时，下列参数全置为`True`，后续可根据需要进行调整

    ```text
    copy_pic: True                               # 是否从下载的数据集中复制图片到制作的数据集保存路径
    generate_mask: True                          # 是否生成mask图片集
    generate_txt: True                           # 是否生成训练、评估、测试用的图片路径列举文本文件
    generate_trimap: True                        # 是否生成trimap图片集
    fixed_ksize: True                            # 是否生成trimap图片集时，设置kernel尺寸固定或者随机扰动
    generate_alpha: True                         # 是否生成alpha图片集
    generate_debug: True                         # 是否生成调试图片集
    generate_mean_std: True                      # 是否计算训练集和验证集的均值和方差
    ```

6. `kernel size`的设置策略: `kernel size`是制作`trimap`图片集时针对图片的`alpha`通道进行腐蚀膨胀时的窗口核尺寸。其中`fixed_ksize`为`True`时，`kernel size`为固定值，为`False`时，`kernel size`会随机变化

    详细配置如下

    ```text
    generate_data:
    path_mt_human: /cache/human_matting          # 下载的数据集
    path_save: /cache/datasets                   # 制作的数据集保存路径
    path_debug: /cache/datasets_debug            # 生成的调试用数据集保存路径

    proportion: '6:2:2'                          # 训练集、验证集和测试集的划分比例
    debug_pic_nums: 400                          # 调试集的总图片数量
    copy_pic: True                               # 是否从下载的数据集中复制图片到制作的数据集保存路径
    generate_mask: True                          # 是否生成mask图片集
    generate_txt: True                           # 是否生成训练、评估、测试用的图片路径列举文本文件
    generate_trimap: True                        # 是否生成trimap图片集
    ksize: 10                                    # kernel尺寸设置
    fixed_ksize: True                            # 是否生成trimap图片集时，设置kernel尺寸固定或者随机扰动
    generate_alpha: True                         # 是否生成alpha图片集
    generate_debug: True                         # 是否生成调试图片集
    generate_mean_std: True                      # 是否计算训练集和验证集的均值和方差

    # 下载的数据集中存在的错误文件，添加在此处，以便代码执行过程中过滤
    list_error_files: ['/cache/human_matting/matting/1803201916/._matting_00000000',
                        '/cache/human_matting/clip_img/1803241125/clip_00000000/._1803241125-00000005.jpg']
    ```

### 训练配置

在`config.yaml`下进行配置，训练有不同的阶段，可分别在`pre_train_t_net`、`pre_train_m_net`、`end_to_end`字段下进行配置

```text
# 训练配置
seed: 9527                          # 设置随机种子
rank: 0                             # 当前训练所使用的卡号（单卡时，为设置的值；多卡时，程序自动设置当前的卡号值）
group_size: 8                       # 分布式训练使用的总卡数
device_target: 'Ascend'             # 指定设备（当前仅支持'Ascend'）
saveIRFlag: False                   # 保存IR图
ckpt_version: ckpt_s2               # .ckpt文件保存版本

pre_train_t_net:                    # T-Net 训练配置
  rank: 0                           # 该参数会在程序运行时进行更新，来自上文的rank配置，以适配当前T-Net训练阶段的rank配置
  group_size: 8                     # 该参数会在程序运行时进行更新，来自上文的group_size配置，以适配当前T-Net训练阶段的group_size配置
  finetuning: True                  # 加载预训练模型
  nThreads: 4                       # 数据集加载的线程数量
  train_batch: 8                    # batch大小
  patch_size: 320                   # 图片resize大小，固定为320
  lr: 1e-3                          # 学习率
  nEpochs: 1000                     # 总epoch数
  save_epoch: 1                     # 每隔多少个epoch保存一次ckpt文件
  keep_checkpoint_max: '10'         # 保存ckpt文件的最大数量，支持'all', '0', '1', '2', ...
  train_phase: pre_train_t_net      # 当前训练阶段

pre_train_m_net:                    # M-Net 训练配置
  rank: 0
  group_size: 8
  finetuning: True
  nThreads: 1
  train_batch: 8
  patch_size: 320
  lr: 1e-4
  nEpochs: 200
  save_epoch: 1
  keep_checkpoint_max: '10'
  train_phase: pre_train_m_net

end_to_end:                         # 端到端训练配置
  rank: 0
  group_size: 8
  finetuning: True
  nThreads: 1
  train_batch: 8
  patch_size: 320
  lr: 1e-4
  nEpochs: 200
  save_epoch: 1
  keep_checkpoint_max: '10'
  train_phase: end_to_end
```

### 测试、推理、模型导出配置

在`config.yaml`中还可以进行以下配置

- 测试：在`test`字段下进行配置

- 推理：在`infer`字段下进行配置

- 模型导出：在`export`字段下进行配置

详细配置如下

```text
# 测试配置
test:
  device_target: 'Ascend'                                                        # 指定推理设备
  model: /cache/output/distribute/ckpt_s2/end_to_end/semantic_hm_best.ckpt       # 训练完成后的checkpoint文件路径
  test_pic_path: /cache/datasets                                                 # 制作完成后的数据集目录
  output_path: /cache/output/distribute/test_result                              # 测试结果保存目录
  size: 320

# 推理配置
infer:
  file_test_list: /cache/datasets/test/test.txt                                  # 数据集制作完成后生成的text.txt文件
  size: 320                                                                      # 图片resize尺寸，固定为320

# 模型导出配置
export:
  ckpt_file: /cache/output/distribute/ckpt_s2/end_to_end/semantic_hm_best.ckpt   # 训练完成后的checkpoint文件路径
  file_name: shm_export                                                          # 模型导出文件名
  file_format: 'MINDIR'                                                          # 指定模型导出格式
  device_target: 'Ascend'                                                        # 指定设备
```

## 数据集制作过程

### 数据集制作

- 下载数据集

    [Matting Human Datasets](<https://github.com/aisegmentcn/matting_human_datasets>)

- 配置`config.yaml`

    配置`generate_data`字段即可

- 执行命令

    ```bassh
    python3 generate_datasets.py --yaml_path=../config.yaml
    ```

- 输出

    - 输出目录结构：

    ```text
    ├── /cache/datasets
        ├── train                        // 训练集：用于训练
            ├── alpha                    // alpha图片集
            ├── clip_img                 // 从下载的数据集的clip_img目录抽取
            ├── mask                     // 生成的mask图片集
            ├── matting                  // 从下载的数据集的matting目录抽取
            ├── trimap                   // 三色图片集
            ├── train.txt                // 列举训练使用的图片列表
        ├── eval                         // 验证集：用于训练完一个epoch后，进行实时评估
            ├── alpha
            ├── clip_img
            ├── mask
            ├── matting
            ├── trimap
            ├── eval.txt                  // 列举验证使用的图片列表
        ├── test                          // 测试集：用于推理测试
            ├── alpha
            ├── clip_img
            ├── mask
            ├── matting
            ├── trimap
            ├── test.txt                  // 列举测试使用的图片列表
    ```

    训练时使用**train**进行训练，每个epoch训练结束时用**eval**进行验证（`T-Net训练阶段`无验证，`M-Net训练阶段（训练时默认不开启该阶段）`和`End-to-End训练阶段`会进行验证），推理时使用**test**目录下的数据。

    - 输出打印日志：

    除了生成训练集、验证集和测试集外，还会打印如下内容

    ```text
    Namespace(yaml_path='../config.yaml')
    Copying source files to train dir...
    Copying source files to eval dir...
    Copying source files to test dir...
    Generate mask ...
    Generate datasets txt ...
    Generate trimap ...
    Generate alpha ...
    Copying train mask into alpha...
    Copying eval mask into alpha...
    Copying test mask into alpha...
    Generate datasets_debug ...
    Generate datasets_debug txt ...
    Generate train and eval datasets mean/std ...
    Total images: 27540
    mean_clip:  [0.40077, 0.43385, 0.49808] [102, 110, 127]
    std_clip:  [0.24744, 0.24859, 0.26404] [63, 63, 67]
    mean_trimap:  [0.56147, 0.56147, 0.56147] [143, 143, 143]
    std_trimap:  [0.47574, 0.47574, 0.47574] [121, 121, 121]
    ```

## 获取初始化权重文件过程

### 获取初始化权重

1. 从github获取torch版源代码： https://github.com/lizhengwei1992/Semantic_Human_Matting

2. 将`src`文件夹下的`get_init_weight.py`文件拷贝到`torch版源代码的根目录`下

    ```text
    ├── shm_original_code         // torch版源代码
        ├── data
        │   ├── data.py
        │   ├── gen_trimap.py
        │   ├── gen_trimap.sh
        │   ├── knn_matting.py
        │   ├── knn_matting.sh
        ├── model
        │   ├── M_Net.py
        │   ├── network.py
        │   ├── N_Net.py
        ├── network.png
        ├── README.md
        ├── test_camera.py
        ├── test_camera.sh
        ├── train.py
        ├── train.sh
        ├── get_init_weight.py    // 将文件拷贝到此处
    ```

3. 执行以下命令

    ```text
    python3 get_init_weight.py
    ```

4. 输出

    执行命令后，会在`torch版源代码根目录`生成`init_weight.ckpt`初始化权重文件，可将**初始化权重文件**拷贝到该**项目根目录**下。

## 训练过程

### 训练

1. 配置`config.yaml`

    设置`pre_train_t_net`、`pre_train_m_net`、`train_phase`字段以及其他字段，详情参考[脚本参数](#脚本参数)下的`训练配置`

2. 执行以下命令

    ```text
    python3 train.py --yaml_path=[YAML_PATH] --data_url=[DATASETS] --train_url=[OUTPUT] --init_weight=[INIT_WEIGHT][OPTIONAL] > train.log 2>&1 &
    # example:  python3 train.py --yaml_path=./config.yaml --data_url=/cache/datasets --train_url=/cache/output --init_weight=./init_weight.ckpt > train.log 2>&1 &
    ```

    上述python命令将在后台运行，您可以通过`train.log`文件查看结果。

    训练过程中会输出以下损失值、速度（秒）等信息：

    ```text
    train epoch: 553 step: 1, loss: 0.022731708, speed: 0.14632129669189453
    train epoch: 553 step: 2, loss: 0.016247142, speed: 0.22849583625793457
    train epoch: 553 step: 3, loss: 0.015720012, speed: 0.19010353088378906
    ...
    ```

    模型检查点保存在`/cache/output/single/ckpt_s2(yaml配置文件中的ckpt_version选项)`目录下，最终推理模型为
    `/cache/output/single/ckpt_s2/end_to_end/semantic_hm_best.ckpt`

3. 训练输出目录结构

    ```text
    ├── /cache/output/[single或distribute]/ckpt_s2/
        ├── pre_train_t_net                       // T-Net训练阶段模型文件及日志保存目录
        │   ├── log_best.txt                      // 未使用
        │   ├── log_latest.txt                   // T-Net训练阶段loss等信息日志
        │   ├── semantic_hm_latest_1.ckpt        // 保存的epoch对应checkpoint文件
        │   ├── semantic_hm_latest_2.ckpt
        │   ├── semantic_hm_latest_3.ckpt
        │   ├── ···
        ├── pre_train_m_net                       // M-Net训练阶段模型文件及日志保存目录，训练时若添加了M-Net训练，则会生成此目录
        │   ├── log_best.txt                      // 相对于前面epoch出现最好精度时的loss等信息日志
        │   ├── log_latest.txt                   // M-Net训练阶段loss等信息日志
        │   ├── semantic_hm_best.ckpt             // 每个epoch最好精度出现时，会覆盖该checkpoint文件
        │   ├── semantic_hm_latest_1.ckpt        // 保存的epoch对应checkpoint文件
        │   ├── semantic_hm_latest_2.ckpt
        │   ├── semantic_hm_latest_3.ckpt
        │   ├── ···
        ├── end_to_end                            // End-to-End训练阶段模型文件及日志保存目录
        │   ├── log_best.txt
        │   ├── log_latest.txt
        │   ├── semantic_hm_best.ckpt
        │   ├── semantic_hm_latest_1.ckpt
        │   ├── semantic_hm_latest_2.ckpt
        │   ├── semantic_hm_latest_3.ckpt
        │   ├── ···
        ├── log_best.txt                          // 所有训练阶段，相对于前面epoch出现最好精度时的loss等信息日志
        ├── log_latest.txt                        // 所有训练阶段loss等信息日志
    ```

4. 注意

    `End-to-End`训练阶段（或者`M-Net`训练阶段）输出日志中的`Sad指标`为图片尺寸为`patch_size`（yaml配置的选项）时直接用预测的alpha和groundtruth计算`Sad指标`，而推理中的`Sad指标`，会将尺寸resize回图片原尺寸再进行计算

### 分布式训练

1. 生成分布式训练json配置文件

    对于分布式训练，需要提前创建JSON格式的hccl配置文件。可将文件命名为`hccl_8p.json`，并存放在当前工程的根目录下。
    请遵循以下链接中的说明：<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

2. 配置`config.yaml`

    设置`pre_train_t_net`、`pre_train_m_net`、`train_phase`字段以及其他字段，详情参考[脚本参数](#脚本参数)下的`训练配置`

3. 执行以下命令

    ```bash
    bash run_train.sh [RANK_TABLE_FILE] [YAML_PATH] [DATASETS] [OUTPUT] [INIT_WEIGHT][OPTIONAL]
    # example: bash run_train.sh ../hccl_8p.json ../config.yaml /cache/datasets /cache/output ../init_weight.ckpt
    ```

4. 输出

   上述shell脚本将在后台运行分布训练。训练的输出`OUTPUT/distribute/..`和上一节训练的输出`OUTPUT/single/..`类似

   - 查看日志

   您可以通过`tail -f device0/train.log`命令查看结果。也可直接打开`device[0~7]/train.log`查看

   - 模型保存位置

   模型检查点保存在`/cache/output/distribute/ckpt_s2(yaml配置文件中的ckpt_version选项)`目录下，最终推理模型为
   `/cache/output/distribute/ckpt_s2/end_to_end/semantic_hm_best.ckpt`

## 评估过程

### 评估

1. 配置`config.yaml`

    设置`test`字段的相关参数

2. 执行以下命令

    ```bash
    python3 eval.py --yaml_path=[YAML_PATH] > eval.log 2>&1 &
    # example: python3 eval.py --yaml_path=./config.yaml > eval.log 2>&1 &
    ```

    或者

    ```bash
    bash run_eval.sh [DEVICE_ID][OPTIONAL]
    # example: bash run_eval.sh
    ```

3. 输出

    上述python命令将在后台运行，您可以通过`eval.log`文件查看结果。测试数据集的准确性如下：

    ```text
    # grep "ave_sad: " ./eval.log
    ave_sad: 5.4309
    ```

## 导出过程

### 导出

在进行推理之前我们需要先导出模型。`mindir`可以在任意环境上导出。`batch_size`只支持1。

1. 配置`config.yaml`
    设置`export`字段的相关参数

2. 执行以下命令

    ```text
    python3 export.py --config_path=[CONFIG_PATH]
    # example: python3 export.py --config_path=./config.yaml
    ```

3. 输出
    执行命令后会在当前目录下生成转换后的模型文件`shm_export.mindir`

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

1. 配置`config.yaml`

    设置`infer`字段下面的参数。修改的项包括`file_test_list`（测试集文件列表, .txt文件绝对路径）和`size`（输入网络的图片尺寸）。

2. 执行以下命令

```bash
bash run_infer_cpp.sh [MINDIR_PATH] [DEVICE_TYPE] [DEVICE_ID]
# [MINDIR_PATH]：上个步骤yaml配置文件中的导出模型文件路径
# example: bash run_infer_cpp.sh ../shm_export.mindir Ascend 0
```

3. 输出

    推理的结果保存在当前目录下

    ```text
    ├── scripts
        ├── preprocess_Result                                  // 前处理输出目录
        │   ├── clip_data                                      // 原图片保存目录
        │   │   ├── matting_0000_1803280628-00000477.jpg       // 原图片——命名规则：matting_[图片编号4位]_[原数据集图片名称]
        │   │   ├── matting_0001_1803280628-00000478.jpg
        │   │   ├── ···
        │   ├── img_data                                       // 原图片预处理成能够输入网络的bin数据目录
        │   │   ├── matting_0000_1803280628-00000477.bin
        │   │   ├── matting_0001_1803280628-00000478.bin
        │   │   ├── ···
        │   ├── label                                          // label目录
        │   │   ├── matting_0000_1803280628-00000477.png
        │   │   ├── matting_0001_1803280628-00000478.png
        │   │   ├── ···
        ├── result_Files                                       // 模型推理输出目录
        │   │   ├── matting_0000_1803280628-00000477_0.bin     // 模型输出第一个值：trimap
        │   │   ├── matting_0000_1803280628-00000477_1.bin     // 模型输出第二个值：alpha
        │   │   ├── matting_0001_1803280628-00000478_0.bin
        │   │   ├── matting_0001_1803280628-00000478_1.bin
        │   │   ├── ···
        ├── postprocess_Result                                 // 后处理输出目录，可在此目录主观查看模型推理效果
        │   │   ├── matting_0000_1803280628-00000477.jpg
        │   │   ├── matting_0001_1803280628-00000478.jpg
        │   │   ├── ···
        ├── time_Result                                        // 推理执行耗时结果保存目录
        │   │   ├── test_perform_static.txt
        ├── infer.log                                          // 模型推理过程日志
        ├── acc.log                                            // 精度输出日志
    ```

在`time_Result`目录下的`test_perform_static.txt`文件中会记录推理的耗时结果

```bach
# grep "time" ./time_Result/test_perform_static.txt
NN inference cost average time: 102.869 ms of infer_count 6901
```

在`acc.log`日志文件中可以找到类似以下的结果。

```bash
# grep "ave sad: " ./acc.log
Total images: 6901, total sad: 38133.463921038725, ave sad: 5.525788135203409
```

# 模型描述

## 性能

以下性能是在加载初始化权重文件`init_weight.ckpt`下得到

### 训练性能

| 参数          | Ascend                                                                                                                                           |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| 模型版本      | Semantic Human Matting V1                                                                                                                        |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8                                                                                         |
| 上传日期      | 2022-01-10                                                                                                                                       |
| MindSpore版本 | 1.6.0                                                                                                                                            |
| 数据集        | human matting dataset                                                                                                                            |
| 训练参数      | T-Net: epoch=1000, steps=320, batch\_size = 8, lr=1e-3, nThreads=1; <br> End-to-End: epoch=200, steps=320, batch\_size = 8, lr=1e-4, nThreads=1; |
| 优化器        | Adam                                                                                                                                             |
| 损失函数      | Softmax交叉熵，绝对误差                                                                                                                          |
| 输出          | T-Net：概率， End-to-End: Sad指标                                                                                                                |
| 速度          | 8卡： T-Net:  453.35 ms/step；End-to-End: 693.10 ms/step                                                                                         |
| 总时长        | 8卡：52h37m12s                                                                                                                                   |
| 微调检查点    | 12.03M (.ckpt文件)                                                                                                                               |
| 推理模型      | 16.56M(.mindir文件)                                                                                                                              |
| 脚本          |                                                                                                                                                  |

### 评估性能

| 参数           | Ascend                    |
|----------------|---------------------------|
| 模型版本       | Semantic Human Matting V1 |
| 资源           | Ascend 910                |
| 上传日期       | 2022-01-10                |
| MindSpore 版本 | 1.6.0                     |
| 数据集         | human matting dataset     |
| batch_size     | 8                         |
| 输出           | matting图的sad指标        |
| 准确性         | 8卡：5.4309               |
| 推理模型       | 16.56M (.mindir文件)      |

# 随机情况说明

- 在`dataset.py`中，我们设置了`create_dataset`函数内的种子，同时还使用了`train.py`中的随机种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
