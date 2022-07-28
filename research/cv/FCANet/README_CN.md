# 目录

- [目录](#目录)
- [FCANet描述](#FCANet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本代码](#脚本代码)
    - [脚本参数](#脚本参数)
    - [准备过程](#准备过程)
    - [训练过程](#训练过程)
        - [启动](#启动)
        - [结果](#结果)
    - [评估过程](#评估过程)
        - [启动](#启动-1)
        - [结果](#结果-1)
- [模型说明](#模型说明)
    - [模型性能](#模型性能)
- [随机情况的描述](#随机情况的描述)
- [ModelZoo](#modelzoo)

<!-- /TOC -->

# FCANet描述

FCANet是一种基于初始点关注的交互分割网络，通过用户交互点击前背景点，不断修复，最终得到精细分割结果。(CVPR 2020)

[论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Interactive_Image_Segmentation_With_First_Click_Attention_CVPR_2020_paper.pdf)：Zheng Lin, Zhao Zhang, Lin-Zhuo Chen, Ming-Ming Cheng，Shao-Ping Lu，Interactive Image Segmentation with First Click Attention. (CVPR2020)

# 模型架构

FCANet总体网络架构如下：

[链接](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Interactive_Image_Segmentation_With_First_Click_Attention_CVPR_2020_paper.pdf)

# 数据集

使用的数据集：处理后的标准交互分割格式 (ISF)

Augmented PASCAL [ [GoogleDrive](https://drive.google.com/file/d/1sQgd_H6m9TGRcPVFJYzGK6u77pKuPDls) | [BaiduYun](https://pan.baidu.com/s/1xshbtKxp1glLyoEmQZGBlg) pwd: **o8vi** ]

GrabCut [ [GoogleDrive](https://drive.google.com/file/d/1CKzgFbk0guEBpewgpMUaWrM_-KSVSUyg/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1Sc3vcHrocYQr9PCvti1Heg) pwd: **2hi9** ]

Berkeley [ [GoogleDrive](https://drive.google.com/file/d/16GD6Ko3IohX8OsSHvemKG8zqY07TIm_i/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/16kAidalC5UWy9payMvlTRA) pwd: **4w5g** ]

我们也在  `(./dataset/)` 中提供了从原始数据集转变成该格式的代码

Augmented PASCAL [ 原始数据集 [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) | [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) ]

```shell
python generate_dataset_pascal_sbd.py --src_pascal_path [source pascal path] --src_sbd_path [source sbd path]
```

GrabCut [ 原始数据集 [GrabCut](https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip) ]

```shell
python generate_dataset_grabcut.py --src_grabcut_path [source grabcut path]
```

Berkeley [ 原始数据集 [Berkeley](https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip) ]

```shell
python generate_dataset_berkeley.py --src_berkeley_path [source berkeley path]
```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- python第三方库（requirements.txt）
    - opencv-python
    - scipy
    - tqdm

# 脚本说明

## 脚本代码

```text
├── FCANet
  ├── README_CN.md                             # 模型相关描述
  ├── dataset                                  # 下载后的数据集解压后放在这里
  │   ├──PASCAL_SBD  
  │   ├──Berkeley
  │   ├──GrabCut
  │   ├──generate_dataset_pascal_sbd.py        # 生成标准数据集 augmented pascal
  │   ├──generate_dataset_berkeley.py          # 生成标准数据集 berkeley
  │   └──generate_dataset_grabcut.py           # 生成标准数据集 grabcut
  ├── scripts
  │   ├──run_standalone_train.sh               # 用于单卡训练的shell脚本
  │   └──run_eval.sh                           # 用于评估的shell脚本
  ├── src
  │   ├──model                                 # 模型架构
  │   │   ├──fcanet.py                         # fcanet模型架构
  │   │   ├──res2net.py                        # res2net主干网络架构
  │   │   └──res2net_pretrained_mindspore.ckpt # 下载后的res2net预训练模型
  │   ├──config.py                             # 参数配置
  │   ├──dataloader_cut.py                     # 数据集导入
  │   ├──helpers.py                            # 辅助的函数
  │   ├──my_custom_transforms.py               # 训练时候的数据增强
  │   └──trainer.py                            # 训练器
  ├── train.py                                 # 训练脚本
  └── eval.py                                  # 评估脚本
```

## 脚本参数

模型训练和评估过程中使用的参数可以在config.py中设置:

```python
"dataset_path": "./dataset/",                                 # 数据集存放位置
"backbone_pretrained": "./src/model/res2net_pretrained.ckpt", # 预训练的res2net模型位置
"dataset_train": "PASCAL_SBD",                                # 训练数据集
"datasets_val": ["GrabCut", "Berkeley"],                      # 测试数据集
"epochs": 33,                                                 # 训练epoch数
"train_only_epochs": 32,                                      # 不进行validation的epoch数
"val_robot_interval": 1,                                      # validation的间隔数
"lr": 0.007,                                                  # 训练初始学习率
"batch_size": 8,                                              # 数据批次大小
"max_num": 0,                                                 # 训练的图像数量，0代表全部图片
"size": (384, 384),                                           # 训练图像尺寸
"device": "Ascend",                                           # 运行设备
"num_workers": 4,                                             # 数据生成线程数
"itis_pro": 0.7,                                              # 迭代训练概率
"max_point_num": 20,                                          # 评测时最大的点数
"record_point_num": 5,                                        # 评测时记录的点数
"pred_tsh": 0.5,                                              # 二值分割的阈值
"miou_target": [0.90, 0.90],                                  # 评测时对不同数据集datasets_val的目标mIoU
```

## 准备过程

- 下载数据集 ,创建文件夹并放在对应目录位置,如`(./dataset/)`，该路径可在config中`"dataset_path"`项修改 。
- 下载res2net101@imagenet预训练模型 [ [GoogleDrive](https://drive.google.com/file/d/1xmbYJOiYvYCp1i_gmif0R8F4Nbl0YupX/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1E9a6PkZ7w_qnOa3iU3ragQ) pwd: **1t4n** ] ,放在对应目录位置`(./src/model/res2net_pretrained.ckpt)`，该路径可在config中`"backbone_pretrained"`项修改 。预训练模型也可通过 [res2net 代码](https://gitee.com/mindspore/models/tree/master/research/cv/res2net)  训练得到。

## 训练过程

### 启动

您可以使用python或shell脚本进行训练。

```shell
# 训练示例
  python:
      Ascend单卡训练示例：DEVICE_ID=[DEVICE_ID] python train.py

  shell:
      Ascend单卡训练示例: bash ./scripts/run_standalone_train.sh [DEVICE_ID]
```

### 结果

ckpt文件将存储在生成的`./snapshot/` 路径下，训练日志将被记录到 `./train.log` 中。训练日志部分示例如下：

```shell
Epoch [000]=>    |-lr:0.0070000-|
Training :
Loss: 0.141: 100%|█████████████████████████████████████████████████████████████| 3229/3229 [37:54<00:00,  1.74it/s]
```

## 评估过程

### 启动

训练脚本最后会进行一次评估, 您可以使用python或shell脚本进行评估,将训练好的模型放在位置 `[PRETRAINED MODEL]` 中，若为训练时保存路径则为 `./snapshot/model-epoch-32.ckpt`  

```shell
# 评估示例
  python:
      DEVICE_ID=[DEVICE_ID] python eval.py -r [PRETRAINED MODEL]

  shell:
      bash ./scripts/run_eval.sh [DEVICE_ID] [PRETRAINED MODEL]
```

### 结果

可以在 `./eval.log` 查看评估结果。

```shell
Validation Robot: [GrabCut]
(point_num_target_avg : 2.76)
(pos_points_num_avg : 1.92) (neg_points_num_avg : 0.84)
(point_num_miou_avg : [0.    0.785 0.877 0.916 0.933 0.946])

Validation Robot: [Berkeley]
(point_num_target_avg : 4.85)
(pos_points_num_avg : 3.03) (neg_points_num_avg : 1.82)
(point_num_miou_avg : [0.    0.745 0.86  0.895 0.912 0.922])
```

# 模型说明

## 模型性能

| 参数                        | Ascend                                |
| -------------------------- | ------------------------------------- |
| 模型名称                    | FCANet                       |
| 模型版本                    | Res2Net版本             |
| 运行环境                    | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| 上传时间                    | 2021-12-18                         |
| 数据集                      | Augmented PASCAL, GrabCut, Berkeley |
| 训练参数                    | src/config.py                        |
| 优化器                      | SGD                           |
| 损失函数                    | CrossEntropyLoss     |
| 最终损失                    | 0.082                        |
| 平均交互点数 (NoC)       | GrabCut (NoC@90=2.76), Berkeley (NoC@90=4.85) |
| 训练总时间             | 24 h                             |
| 评估总时间                  | 10 min                             |
| 单step时间              | 0.8s/                   |
| 卡数                  | 1                            |
| 脚本                       | [链接](https://gitee.com/mindspore/models/tree/master/research/cv/FCANet) |

# 随机情况的描述

我们在 `trainer.py` 脚本中设置了随机种子。

# ModelZoo

请核对官方 [主页](https://gitee.com/mindspore/models)。