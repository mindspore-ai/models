# 目录

<!-- TOC -->

- [目录](#目录)
- [ERFNet描述](#erfnet描述)
    - [概述](#概述)
    - [论文](#论文)
    - [关于精度](#关于精度)
- [环境](#环境)
- [数据集](#数据集)
- [脚本说明](#脚本说明)
- [训练](#训练)
    - [单卡训练](#单卡训练)
    - [多卡训练](#多卡训练)
- [验证](#验证)
    - [验证单个ckpt](#验证单个ckpt)
- [推理](#推理)
    - [使用ckpt文件推理](#使用ckpt文件推理)
    - [310推理](#310推理)

<!-- /TOC -->

# ERFNet描述

## 概述

ERFNet可以看作是对ResNet结构的又一改变，ERFNet提出了Factorized Residual Layers，内部全部使用1D的cov(非对称卷积)，以此来降低参数量，提高速度。同时ERFNet也是对ENet的改进，在模型结构上删除了encode中的层和decode层之间的long-range链接，同时所有的downsampling模块都是一组并行的max pooling和conv。

使用mindpsore复现ERFNet[[论文]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)。
这个项目迁移于原作者对ERFNet的Pytorch实现[[HERE](https://github.com/Eromera/erfnet_pytorch)]。

## 论文

1. [论文](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)：E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo."ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation"

2. [论文](https://arxiv.org/abs/1606.02147)：A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello."ENet: A deep neural network architecture for real-time semantic segmentation."

## 关于精度

| (Val IOU/Test IOU) | [erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch) | [论文](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf) |
|-|-|-|
| **512 x 1024** |  **72.1/69.8** | * |
| **1024 x 2048** | * | **70.0/68.0** |

[erfnet_pytorch](https://github.com/Eromera/erfnet_pytorch)是作者对erfnet的pytroch实现，
上表显示了其readme中称能达到的结果和论文中声称的结果。

测试和训练时图片的输入大小尺寸会影响精度，cityscapes数据集中的图片尺寸全部是2048x1024。论文和pytorch的具体实现， 对于图片尺寸的处理也有所不同。

论文中声称对图片和标签进行2倍下采样(1024x512)再进行训练，测试时在1024x512下进行推断，然后对prediction进行插值到2048x1024再和label计算IOU。在pytorch的实现中，训练和测试均在下采样后的1024x512下进行。实测Pytorch实现在val上能达到70.7%的IOU。

# 环境

Ascend

# 数据集

[**The Cityscapes dataset**](https://www.cityscapes-dataset.com/):

在官网直接下载的标签文件, 像素被分为30多类, 在训练时我们需要将其归纳到20类, 所以对其需要进行处理.

## 处理数据集

在官网上下载数据集：gtFine_trainvaltest.zip (241MB) [md5]、gtCoarse.zip (1.3GB) [md5]、gtCoarse.zip (1.3GB) [md5]

Cityscapes数据集提供了34种分类（默认为19种分类），如果想要任意分类，需要使用自带的[**工具**](https://github.com/mcordts/cityscapesScripts)。cityscapesscripts需要和gtCoarse、gtFine、leftImg8bit在同一目录下。

cityscapesscripts/helpers/labels.py：这个python文件工具作用是映射我们所需训练的物体到一个具体颜色数值。最多可分34类，若增减其中需要分割物体可用此工具。label字典中trainID这一列，凡是255的表示均没有加入训练分类。修改trainID和不感兴趣的ignoreInEval改为True。

cityscapesscripts/preparation/creatTrainldLabellmgs.py：当用上一个工具定义好训练集识别物体数量后，需要用这个文件重新生成训练集。

## 处理好的数据集

为了方便可以直接下载已经处理好的数据, 链接：https://pan.baidu.com/s/1jH9GUDX4grcEoDNLsWPKGw. 提取码：aChQ.

涉及的文件目录如下:

```path
├── gtFine.zip
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
```

## 生成训练数据的mindrecord文件

运行脚本：

```shell
python src/build_mrdata.py \
--dataset_path /path/to/cityscapes/ \
--subset train \
--output_name train.mindrecord
```

脚本会在/path/to/cityscapes/数据集根目录下，找到训练集，在output_name指出的路径下生成mindrecord文件，
然后再把mindrecord文件（train.mindrecord、train.mindrecord.db）移动到项目根目录下的data文件夹下，来让脚本中的相对路径能够寻找到

- 测试或验证数据集没有使用mindrecord文件，而是通过src/dataset.py中的cityscapes函数读取

# 脚本说明

```sh
├── ascend310_infer
│   ├── inc
│   │   └── utils.h                           // utils头文件
│   └── src
│       ├── CMakeLists.txt                    // cmakelist
│       ├── main.cc                           // 推理代码
│       ├── build.sh                          // 运行脚本
│       └── utils.cc                          // utils实现
├── eval.py                                   // 测试脚本
├── export.py                                 // 生成模型文件脚本
├── README_CN.md                              // 描述文件
├── requirements.txt                          // python环境依赖
├── scripts
│   ├── run_infer_310.sh                          // 310推理脚本
│   ├── run_eval.sh                               // 推理脚本
│   ├── run_distribute_train_gpu.sh               // gpu多卡训练脚本
│   ├── run_distribute_train.sh                   // 多卡训练脚本
│   └── run_standalone_train.sh                   // 单卡训练脚本
├── src
│   ├── build_mrdata.py                           // 生成mindrecord数据集
│   ├── config.py                                 // 配置参数脚本
│   ├── dataset.py                                // 数据集脚本
│   ├── infer.py                                  // 推断脚本
│   ├── iouEval.py                                // metric计算脚本
│   ├── model.py                                  // 模型脚本
│   ├── eval310.py                                // 310推理脚本
│   ├── show.py                                   // 结果可视化脚本
│   └── util.py                                   // 工具函数脚本
└── train.py                                      // 训练脚本
```

# 训练

训练之前需要生成mindrecord数据文件并放到项目根目录的data文件夹下，然后启动脚本。

## 单卡训练

如果你要使用单卡进行训练，进入项目根目录

- Ascend处理器环境运行

  ```shell
  # 训练示例
  nohup bash scripts/run_standalone_train.sh Ascend /home/name/cityscapes 0 &
  ```

- GPU处理器环境运行

  ```shell
  # 训练示例
  nohup bash scripts/run_standalone_train.sh GPU /home/name/cityscapes 0 &
  ```

在项目根目录下会生成standalone_train文件夹，./standalone_train/log_stage*.txt即为程序log文件

其中/home/name/cityscapes指数据集的位置，其后的0指定device_id.

## 多卡训练

例如，你要使用4卡进行训练，进入项目根目录

- Ascend处理器环境运行

  ```shell
  # 训练示例
  nohup bash scripts/run_distribute_train.sh /home/name/cityscapes 4 0,1,2,3 /home/name/rank_table_4pcs.json &
  ```

在项目根目录下会生成distribute_train文件夹，./distribute_train/log0/log.txt即为程序log文件

- GPU处理器环境运行

  ```shell
  # 训练示例
  nohup bash scripts/run_distribute_train_gpu.sh /home/name/cityscapes 4 0,1,2,3 &
  ```

在项目根目录下会生成distribute_train_gpu文件夹，./distribute_train_gpu/log_stage*.txt即为程序log文件

其中/home/name/cityscapes指数据集的位置，其后的4指rank_size, 再后的0,1,2,3制定了设备的编号，/home/name/rank_table_4pcs.json指并行训练配置文件的位置。其他数目的设备并行训练也类似。

# 验证

训练之后，需要调用验证脚本，对不同的ckpt文件，会加上后缀_metrics.txt，其中包含测试精度。

## 验证ckpt

- Ascend处理器环境运行

  ```shell
  # 验证示例
  nohup bash scripts/run_eval.sh Ascend /home/name/cityscapes distribute_train/checkpoint 0 &
  ```

- GPU处理器环境运行

  ```shell
  # 验证示例
  nohup bash scripts/run_eval.sh GPU /home/name/cityscapes distribute_train_gpu/checkpoint 0 &
  ```

/home/name/cityscapes为数据集根目录，distribute_train/checkpoint为ckpt文件路径，其后的0指定device_id.。

验证完毕后，会在ckpt文件同目录下生成后缀_metrics.txt文件，其中包含测试点数。

```txt
mean_iou 0.7090318296884867
mean_loss 0.296806449357143
iou_class tensor([0.9742, 0.8046, 0.9048, 0.4574, 0.5067, 0.6105, 0.6239, 0.7221, 0.9134,
        0.5903, 0.9352, 0.7633, 0.5624, 0.9231, 0.6211, 0.7897, 0.6471, 0.4148,
        0.7069], dtype=torch.float64)
```

# 推理

## 使用ckpt文件推理

- Ascend处理器环境运行

  ```shell
  # 推理示例
  python src/infer.py \
  --data_path /path/to/imgs \
  --model_path /path/to/ERFNet.ckpt /
  --output_path /output/path \
  --device_id 3
  ```

- GPU处理器环境运行

  ```shell
  # 推理示例
  python src/infer.py \
  --data_path /path/to/imgs \
  --model_path /path/to/ERFNet.ckpt /
  --output_path /output/path \
  --device_target='GPU'
  --device_id 3
  ```

脚本会读取/path/to/imgs下的图片，使用/path/to/ERFNet.ckpt模型进行推理，得到的可视化结果输出到/output/path下。

## 310推理

需要处理训练好的ckpt文件, 得到能在310上直接推理的mindir模型文件:

- Ascend处理器环境运行

  ```shell
  python export.py --model_path=/path/to/net.ckpt
  ```

- GPU处理器环境运行

  ```shell
  python export.py --model_path=/path/to/net.ckpt --device_target='GPU'
  ```

会在当前目录下得到ERFNet.mindir文件, 之后进入ascend310_infer文件夹,

```shell
cd ascend310_infer
bash scripts/run_infer_310.sh /path/to/net.mindir /path/to/images /path/to/result  /path/to/label 0
```

其中/path/to/images指验证集的图片, 由于原始数据集的路径cityscapes/leftImg8bit/val/的图片根据拍摄的城市进行了分类, 需要先将其归到一个文件夹下才能供推理.
例如

```shell
cp /path/to/cityscapes/leftImg8bit/val/frankfurt/* /path/to/images/
cp /path/to/cityscapes/leftImg8bit/val/lindau/* /path/to/images/
cp /path/to/cityscapes/leftImg8bit/val/munster/* /path/to/images/
```

验证集的ground truth, 同理也要归到/path/to/labels/下. 其余的参数/path/to/net.mindir指mindir文件的路径, /path/to/result推理结果的输出路径(文件夹需要提前创建好), 0指的是device_id

最终推理结果会输出在/res/result/文件夹下, 当前目录下会生成metric.txt, 其中包含精度.
