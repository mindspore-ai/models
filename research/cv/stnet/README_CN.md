# 目录

<!-- TOC -->

- [目录](#目录)
- [StNet描述](#StNet描述)
- [环境要求](#环境要求)
- [数据集](#数据集)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [导出过程](#导出过程)
    - [推理过程](#推理过程)
- [模型描述](#模型描述)
    - [训练性能](#训练性能)
    - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [模型结构情况说明](#模型结构情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# StNet描述

StNet是兼顾局部时空联系以及全局时空联系的视频时空联合建模网络框架,其将视频中连续N帧图像级联成一个3N通道的超图,然后用2D卷积对超图进行局部时空联系的建模。为了建立全局时空关联,StNet中引入了对多个局部时空特征图进行时域卷积的模块,并采用时序Xception模块对视频特征序列进一步建模和挖掘出蕴含的时序依赖。

[论文地址](https://arxiv.org/abs/1811.01549)："StNet: Local and Global Spatial-Temporal Modeling for Action Recognition."

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 数据集

[**Kinetics400 dataset**](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics):

考虑到从油管下载的视频有部分已失效或者无法使用，经过筛选后的数据集标签提供如下
数据集标签链接：[[HERE]](https://pan.baidu.com/s/1e58v4nrwzfYT459EZ3L_PA). 提取码是：ms13.

下载后可以得到以下目录:

  ```sh
  └── data
      ├── train_mp4 ............................训练集
      └── val_mp4 ............................. 验证集
  ```

- mp4文件预处理

为提高数据读取速度，提前将mp4文件解帧并打pickle包，dataloader从视频的pkl文件中读取数据（该方法耗费更多存储空间）。pkl文件里打包的内容为(video-id,[frame1, frame2,...,frameN],label)。

在 data目录下创建目录train_pkl和val_pkl

  ```bash
  cd $Code_Root/data

  mkdir train_pkl && mkdir val_pkl
  ```

进入$Code_Root/src目录，使用video2pkl.py脚本进行数据转化。首先需要下载train和validation数据集的文件列表。

首先生成预处理需要的数据集标签文件

  ```bash
  python generate_label.py kinetics-400_train.csv kinetics400_label.txt
  ```

然后执行如下程序(该脚本依赖ffmpeg库，请预先安装ffmpeg)：

  ```bash
  python video2pkl.py kinetics-400_train.csv $Source_dir $Target_dir  8 #以8个进程为例
  ```

对于train数据，

  ```bash
  Source_dir = $Code_Root/data/train_mp4

  Target_dir = $Code_Root/data/train_pkl
  ```

对于val数据，

  ```bash
  Source_dir = $Code_Root/data/val_mp4

  Target_dir = $Code_Root/data/val_pkl
  ```

这样即可将mp4文件解码并保存为pkl文件。

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估。对于分布式训练，需要提前创建JSON格式的hccl配置文件。请遵循以下链接中的说明：
 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>

- Ascend处理器环境运行

  ```python
  # 训练时添加预训练好的resnet50参数，设置保存参数和summary位置
  pre_res50:Code_Root/data/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76.ckpt
  checkpoint_path summary_dir
  # 线下训练时
  run_online = False
  ```

  ```python
  # 运行训练示例
  python train.py --device_id=0 --dataset_path='' --run_distribute=0 --resume=''(可选)

  # 运行分布式训练示例
  bash run_distribute_train.sh [run_distribute][DATASET_PATH][rank_table_PATH][PRETRAINED_CKPT_PATH]（可选）

  # 运行评估示例
  python eval.py --device_id=0 --dataset_path='' --run_distribute=0 --resume=''
  或者
  bash run_eval.sh [device_id] [DATASET_PATH] [CHECKPOINT_PATH]
  ```

对于分布式训练，需要提前创建JSON格式的hccl配置文件。

请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 上训练(如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

  ```python
  # (1) 选择上传代码到 S3 桶
  #     选择代码目录/s3_path_to_code/StNet/
  #     选择启动文件/s3_path_to_code/StNet/train.py
  # (2) 在config.py设置参数
  #     run_online = True
  #     data_url = [S3 桶中数据集的位置]
  #     local_data_url = [云上数据集的位置]
  #     pre_url = [S3 桶中预训练resnet50、resume、train和val标签的位置]
  #     pre_res50_art_load_path = [云上预训练resnet50的位置]
  #     best_acc_art_load_path = [云上预训练模型的位置] 或 [不设置]
  #     load_path = [云上预训练resnet50、resume、train和val标签的位置]
  #     train_url = [S3桶中结果输出的位置]
  #     output_path = [云上结果输出的位置]
  #     local_train_list = [云上训练集标签的位置]
  #     local_val_list = [云上验证集标签的位置]
  #     [其他参数] = [参数值]
  # (3) 上传Kinetics-400数据集到 S3 桶上,由于处理过后的数据集过大,推荐将训练集拆分成16个压缩文件，验证集拆分成2个压缩文件, 配置"训练数据集"路径
  # (4) 在网页上设置"训练输出文件路径"、"作业日志路径"等
  # (5) 选择8卡机器，创建训练作业
  ```

# 脚本说明

## 脚本及样例代码

```bash
├── scripts
    ├── run_distribute_train.sh                   // 多卡训练脚本
    └── run_eval.sh                               // 验证脚本
├── src
    ├── model_utils
        └── moxing_adapter.py                     // model_art传输文件
    ├── eval_callback.py                          // 验证回传脚本
    ├── config.py                                 // 配置参数脚本
    ├── dataset.py                                // 数据集脚本
    ├── Stnet_Res_model.py                        // 模型脚本
    ├── generate_label.py                         // 预处理生成标签脚本
    └── video2pkl.py                              // 预处理视频转pkl格式脚本
├── README_CN.md                                  // 描述文件
├── eval.py                                       // 测试脚本
└── train.py                                      // 训练脚本
```

## 脚本参数

- 在config.py中可以同时配置训练参数和评估参数。

  ```python
  'batch_size': 16,                                   # 输入张量的批次大小
  'num_epochs': 60,                                   # 此值仅适用于训练；应用于推理时固定为1
  'class_num': 400,                                   # 数据集类数
  'T': 7,                                             # 片段数量
  'N':5,                                              # 片段长度
  'mode': 'GRAPH',                                    # 训练模式
  'resume':None,                                      # 预训练模型
  'pre_res50': "data/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76.ckpt"  # resnet50预训练模型参数
  'momentum': 0.9,                                    # 动量
  'lr': 1e-2,                                         # 初始学习率
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 单卡训练

- Ascend处理器环境运行

  ```bash
  python train.py --device_id=0 --dataset_path='' --run_distribute=0 --resume=''
  ```

  从头开始训练，需要加载在ImageNet上训练的ResNet50权重作为初始化参数，请下载此[模型参数](https://download.mindspore.cn/model_zoo/r1.2/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76/)，将参数保存在data/pretrained目录下面.
  可下载已发布模型,通过`--resume`指定权重存放路径进行finetune等开发

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash run_distribute_train.sh  [run_distribute][DATASET_PATH][rank_table_PATH][PRETRAINED_CKPT_PATH]（可选）
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。

## 评估过程

### 评估

- 在Ascend环境运行时评估Kinetics-400数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/stnet/best_acc.ckpt”。

  ```bash
  python eval.py --device_id=0 --dataset_path='' --run_distribute=0 --resume=''
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" eval.log
  accuracy:{'acc':0.69}
  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“username/stnet/train_parallel0/best_acc.ckpt”。测试数据集的准确性如下：

  ```bash
  # grep "accuracy:" dist.eval.log
  accuracy:{'acc':0.69}
  ```

## 导出过程

### 导出

在导出之前需要修改数据集对应的配置文件config.py
需要修改的配置项为 batch_size 和 ckpt_file.

  ```shell
  python export.py --resume [CONFIG_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
  ```

## 推理过程

### 推理

在还行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

- 在昇腾310上使用Kinetics-400数据集进行推理

  在执行下面的命令之前，我们需要先修改confi.py的配置文件。修改的项包括batch_size。

  推理的结果保存在当前目录下，在acc.log日志文件中可以找到类似以下的结果。

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS]  [DEVICE_ID]
  after allreduce eval: top1_correct=9252, tot=10000, acc=92.52%
  ```

需要四个参数：

-MINDIR_PATH:MINDIR文件的绝对路径

-DATA_PATH:未处理过的eval数据集路径，若NEED_PREPROCESS为N，此参数可以填为“”

-NEED_PREPROCESS:是否要将eval数据集进行处理

-DEVICE_ID:310上使用的芯片号

# 模型描述

## 性能

### 训练性能

#### Kinetics-400上训练StNet(8卡)

| Parameters          | Ascend                                           |
| ------------------- | ------------------------------------------------ |
| Model Version       | StNet                                             |
| Resource            | Ascend 910；CPU：2.60GHz, 192 cores; RAM: 755 GB |
| uploaded Date       | 2021-12-17                                       |
| MindSpore Version   | 1.3.0                                            |
| Dataset             | Kinetics-400                                            |
| Training Parameters | See config.py for details                        |
| Optimizer           | Momentum                                             |
| Loss Function       | CrossEntropySmooth                            |
| Speed               | 921 ms / step（0卡）                                |
| Total time          | 33h 59 min 48s (0卡)                                         |
| Top1         | 69.16%                                           |
| Parameters          | 301                                         |

### 评估性能

#### Kinetics-400上评估

the performance of Top1 is 69.16%.

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# 模型结构情况说明

在TemporalXception中由于MindSpore1.3版本对于分组卷积不支持，故将其注释，若版本为1.5，可尝试

# ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。