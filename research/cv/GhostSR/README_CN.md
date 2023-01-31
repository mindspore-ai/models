# 目录

<!-- TOC -->

- [目录](#目录)
- [GhostSR 描述](#GhostSR 描述)
- [环境配置/推理/导出](#环境配置/推理/导出)
- [数据集](#数据集)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [模型评估](#模型评估)
    - [评估性能](#评估性能)
        - [DIV2K上的评估2倍超分辨率重建的EDSR](#DIV2K上的评估2倍超分辨率重建的GhostSR_EDSR)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GhostSR 描述

GhostSR 是2022年提出的轻量级单图超分辨重建网络。它通过引入shift operation 来生成 ghost
features，大幅减少参数量、flops和推理延迟的同时几乎性能无损。

论文：[GhostSR: Learning Ghost Features for Efficient Image Super-Resolution](https://arxiv.org/abs/2101.08525)

# 环境配置/推理/导出

本代码修改自 [EDSR(MindSpore)](https://gitee.com/mindspore/models/tree/master/official/cv/EDSR),
环境配置/推理/导出等操作可参考EDSR

# 数据集

使用的数据集：[DIV2K](<https://data.vision.ee.ethz.ch/cvl/DIV2K/>)

- 数据集大小：7.11G，共1000组（HR,LRx2,LRx3,LRx4）有效彩色图像
    - 训练集：6.01G，共800组图像
    - 验证集：783.68M，共100组图像
    - 测试集：349.53M，共100组图像(无HR图)
- 数据格式：PNG图片文件文件
    - 注：数据将在src/dataset.py中处理。
- 数据目录树：官网下载数据后，解压压缩包，训练和验证所需的数据目录结构如下：

```shell
├─DIV2K_train_HR
│  ├─0001.png
│  ├─...
│  └─0800.png
├─DIV2K_train_LR_bicubic
│  ├─X2
│  │  ├─0001x2.png
│  │  ├─...
│  │  └─0800x2.png
│  ├─X3
│  │  ├─0001x3.png
│  │  ├─...
│  │  └─0800x3.png
│  └─X4
│     ├─0001x4.png
│     ├─...
│     └─0800x4.png
├─DIV2K_valid_LR_bicubic
│  ├─0801.png
│  ├─...
│  └─0900.png
└─DIV2K_valid_LR_bicubic
   ├─X2
   │  ├─0801x2.png
   │  ├─...
   │  └─0900x2.png
   ├─X3
   │  ├─0801x3.png
   │  ├─...
   │  └─0900x3.png
   └─X4
      ├─0801x4.png
      ├─...
      └─0900x4.png
```

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估。对于分布式训练，需要提前创建JSON格式的hccl配置文件。请遵循以下链接中的说明：
<https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- GPU环境运行单卡评估DIV2K

  ```python
  # 运行评估示例(EDSR_mindspore(x2) in the paper)
  python eval.py --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained ./ckpt/EDSR_GhostSR_x2.ckpt > train.log 2>&1 &
  ```

- GPU环境运行单卡评估benchmark

  ```python
  # 运行评估示例(EDSR_mindspore(x2) in the paper)
  python eval.py --config_path benchmark_config.yaml --scale 2 --data_path [benchmark path] --output_path [path to save sr] --pre_trained ./ckpt/EDSR_GhostSR_x2.ckpt > train.log 2>&1 &
  ```

# 脚本说明

## 脚本及样例代码

```text
├── model_zoo
    ├── README.md                       // 所有模型相关说明
    ├── EDSR
        ├── README_CN.md                // EDSR说明
        ├── model_utils                 // 上云的工具脚本
        ├── DIV2K_config.yaml           // EDSR参数
        ├── ckpt
        │   └── EDSR_GhostSR_x2.ckpt    // EDSR_GhostSR 2倍超分辨率模型权重
        ├── GhostSR                     // GhostSR 网络架构
        │   ├── EDSR_mindspore          // EDSR_GhostSR 网络架构
        │   └── unsupported_model       // mindspore 中未原生支持的算子
        ├── scripts
        │   ├── run_train.sh            // 分布式到Ascend的shell脚本
        │   ├── run_eval.sh             // Ascend评估的shell脚本
        │   ├── run_infer_310.sh        // Ascend-310推理shell脚本
        │   └── run_eval_onnx.sh        // 用于ONNX评估的shell脚本
        ├── src
        │   ├── dataset.py              // 创建数据集
        │   ├── edsr.py                 // edsr网络架构
        │   ├── config.py               // 参数配置
        │   ├── metric.py               // 评估指标
        │   ├── utils.py                // train.py/eval.py公用的代码段
        ├── train.py                    // 训练脚本
        ├── eval.py                     // 评估脚本
        ├── eval_onnx.py                // ONNX评估脚本
        ├── export.py                   // 将checkpoint文件导出到onnx/air/mindir
        ├── preprocess.py               // Ascend-310推理的数据预处理脚本
        ├── ascend310_infer
        │   ├── src                     // 实现Ascend-310推理源代码
        │   ├── inc                     // 实现Ascend-310推理源代码
        │   ├── build.sh                // 构建Ascend-310推理程序的shell脚本
        │   ├── CMakeLists.txt          // 构建Ascend-310推理程序的CMakeLists
        ├── postprocess.py              // Ascend-310推理的数据后处理脚本
```

## 脚本参数

在DIV2K_config.yaml中可以同时配置训练参数和评估参数。benchmark_config.yaml中的同名参数是一样的定义。

- 可以使用以下语句可以打印配置说明

  ```python
  python train.py --config_path DIV2K_config.yaml --help
  ```

- 可以直接查看DIV2K_config.yaml内的配置说明，说明如下

  ```yaml
  enable_modelarts: "在云道运行则需要配置为True, default: False"

  data_url: "云道数据路径"
  train_url: "云道代码路径"
  checkpoint_url: "云道保存的路径"

  data_path: "运行机器的数据路径，由脚本从云道数据路径下载，default: /cache/data"
  output_path: "运行机器的输出路径，由脚本从本地上传至checkpoint_url，default: /cache/train"
  device_target: "可选['Ascend']，default: Ascend"

  amp_level: "可选['O0', 'O2', 'O3', 'auto']，default: O3"
  loss_scale: "除了O0外，其他混合精度时会做loss放缩，default: 1000.0"
  keep_checkpoint_max: "最多保存多少个ckpt， defalue: 60"
  save_epoch_frq: "每隔多少epoch保存ckpt一次， defalue: 100"
  ckpt_save_dir: "保存的本地相对路径，根目录是output_path， defalue: ./ckpt/"
  epoch_size: "训练多少个epoch， defalue: 6000"

  eval_epoch_frq: "训练时每隔多少epoch执行一次验证，defalue: 20"
  self_ensemble: "验证时执行self_ensemble，仅在eval.py中使用， defalue: True"
  save_sr: "验证时保存sr和hr图片，仅在eval.py中使用， defalue: True"

  opt_type: "优化器类型,可选['Adam']，defalue: Adam"
  weight_decay: "优化器权重衰减参数，defalue: 0.0"

  learning_rate: "学习率，defalue: 0.0001"
  milestones: "学习率衰减的epoch节点列表，defalue: [4000]"
  gamma: "学习率衰减率，defalue: 0.5"

  dataset_name: "数据集名称，defalue: DIV2K"
  lr_type: "lr图的退化方式，可选['bicubic', 'unknown']，defalue: bicubic"
  batch_size: "为了保证效果，建议8卡用2，单卡用16，defalue: 2"
  patch_size: "训练时候的裁剪HR图大小，LR图会依据scale调整裁剪大小，defalue: 192"
  scale: "模型的超分辨重建的尺度，可选[2,3,4], defalue: 4"
  dataset_sink_mode: "训练使用数据下沉模式，defalue: True"
  need_unzip_in_modelarts: "从s3下载数据后加压数据，defalue: False"
  need_unzip_files: "需要解压的数据列表, need_unzip_in_modelarts=True时才起作用"

  pre_trained: "加载预训练模型，x2/x3/x4倍可以相互加载，可选[[s3绝对地址], [output_path下相对地址], [本地机器绝对地址], '']，defalue: ''"
  rgb_range: "图片像素的范围，defalue: 255"
  rgb_mean: "图片RGB均值，defalue: [0.4488, 0.4371, 0.4040]"
  rgb_std: "图片RGB方差，defalue: [1.0, 1.0, 1.0]"
  n_colors: "RGB图片3通道，defalue: 3"
  n_feats: "每个卷积层的输出特征数量，defalue: 256"
  kernel_size: "卷积核大小，defalue: 3"
  n_resblocks: "resblocks数量，defalue: 32"
  res_scale: "res的分支的系数，defalue: 0.1"
  ```

# 模型评估

## 性能

### DIV2K上的评估2倍/3倍/4倍超分辨率重建的EDSR

| 参数           | Ascend |
|--------------|---|
| 模型版本         | EDSR-GhostSR(x2) |
| MindSpore版本  | 1.9.0 |
| 数据集          | DIV2K, 100张图像 |
| self_ensemble | True |
| batch_size   | 1 |
| 输出           | 超分辨率重建RGB图 |
| Set5 psnr    | 38.101 db |
| Set14 psnr   | 33.856 db |
| B100 psnr    | 32.288 db |
| Urban100 psnr | 32.793 db |
| DIV2K psnr   | 34.8748 db |
| 推理模型         | 83.3 MB (.ckpt文件) |

# 随机情况说明

在train.py，eval.py中，我们设置了mindspore.common.set_seed(2021)种子。

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
