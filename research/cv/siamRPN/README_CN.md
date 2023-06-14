# 目录

- [目录](#目录)
- [SiamRPN描述](#概述)
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
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
            - [910评估](#910评估)
            - [GPU评估](#gpu评估)
            - [310评估·](#310评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 概述

Siam-RPN提出了一种基于RPN的孪生网络结构。由孪生子网络和RPN网络组成，它抛弃了传统的多尺度测试和在线跟踪，从而使得跟踪速度非常快。在VOT2015, VOT2016和VOT2017上取得了领先的性能，并且速度能都达到160fps。

[论文](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)：Bo Li,Junjie Yan,Wei Wu,Zheng Zhu,Xiaolin Hu, High Performance Visual Tracking with Siamese Region Proposal Network[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.

# 模型架构

此网络由Siamese Network和Region Proposal Network两部分组成。前者用来提取特征，后者用来产生候选区域。其中，RPN子网络由两个分支组成，一个是用来区分目标和背景的分类分支，另外一个是微调候选区域的回归分支。整个网络实现了端到端的训练。

# 数据集

：[VID-youtube-bb](https://pan.baidu.com/s/1QnQEM_jtc3alX8RyZ3i4-g) [VOT2015](https://www.votchallenge.net/vot2015/dataset.html)  [VOT2016](https://www.votchallenge.net/vot2016/dataset.html)

- 百度网盘密码：myq4
- 数据集大小：143.8G，共600多万图像
    - 训练集：141G，共600多万图像
    - 测试集：2.8G，共60个视频
- 数据格式：RGB

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend/GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.2/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.2/index.html#operator_api)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：
首先我们需要先下载训练时使用的[预训练模型](https://drive.google.com/drive/folders/1HJOvl_irX3KFbtfj88_FVLtukMI1GTCR)

- Ascend处理器环境运行

  ```python
  #模型转换
  python src/pth2ckpt.py --model_path /path/to/pth

  # 运行训练示例
  bash scripts/run.sh DATA_PATH DEVICE_ID

  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh /path/dataset /path/rank_table

  # 运行评估示例
  bash scripts/run_eval.sh 0 /path/dataset /path/ckpt/siamRPN-50_1417.ckpt eval.json

  ```

- GPU处理器环境运行

  ```python
  # 运行训练示例
  bash scripts/run_gpu.sh 0

  # 运行分布式训练示例
  bash scripts/run_distribute_train_gpu.sh  device_num device_list

  # 运行评估示例
  bash scripts/run_eval_gpu.sh 0 /path/dataset /path/ckpt/siamRPN-50_1417.ckpt eval.json

  ```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── research
        ├── cv
            ├── siamRPN
                ├── README_CN.md            // SiamRPN相关说明
                ├── ascend_310_infer        // 实现310推理源代码
                ├── scripts
                │    ├──run.sh              // 训练脚本
                |    |──run_distribute_train.sh //本地Ascend多卡训练脚本
                |    |──run_eval.sh         //910评估脚本
                |    |──run_eval_gpu.sh     // GPU评估脚本
                |    |──run_distribute_train_gpu.sh      // 本地GPU多卡训练脚本
                |    |──run_infer_310.sh    //310推理评估脚本
                |    |──run_gpu.sh          //GPU单卡训练脚本
                |    |──run_evalonnx.sh          //onnx推理数据集脚本
                ├── src
                │    ├── data_loader.py      // 数据集加载处理脚本
                │    ├── net.py              //  siamRPN架构
                │    ├── loss.py             //  损失函数
                │    ├── util.py             //  工具脚本
                │    ├── tracker.py
                │    ├── generate_anchors.py
                │    ├── tracker.py
                │    ├── evaluation.py
                │    ├── config.py          // 参数配置
                ├── ytb_vid_filter         //训练集(需要自己下载)
                │    ├── --0bLFuriZ4
                │    ├── --4VWx_0Sc4
                │    ├── ······
                │    ├── ······
                │    └── meta_data.pkl
                ├── vot2015                //测试集(需要自己下载)
                │    ├── bag
                │    ├── ball1
                │    ├── ······
                │    ├── ······
                │    └── list.txt
                ├── vot2016                //测试集(需要自己下载)
                │    ├── bag
                │    ├── ball1
                │    ├── ······
                │    ├── ······
                │    └── list.txt
                ├── train.py               // 训练脚本
                ├── evalonnx.py                // onnx评估脚本
                ├── eval.py                // ckpt评估脚本
                ├── export.py                // 将checkpoint文件导出到onnx或mindir
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 训练相关参数。

  ```python
    checkpoint_path = r'/home/data/ckpt'        # 模型检查点保存目录
    pretrain_model = 'alexnet.ckpt'             # 预训练模型名称
    train_path = r'/home/data/ytb_vid_filter'   # 训练数据集路径
    cur_epoch = 0                               #当前训练周期
    max_epoches = 50                            #训练最大周期
    batch_size = 32                             #每次训练样本数

    start_lr = 3e-2                             #初始训练学习率
    end_lr = 1e-7                               #结束学习率
    momentum = 0.9                              #动量
    weight_decay = 0.0005                       # 权重衰减值
    check = True                                #是否加载模型
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --device_id=0 --device_target="Ascend"> train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash
  # grep "loss is " train.log
  epoch:1 step:390, loss is 1.4842823
  epcoh:2 step:390, loss is 1.0897788
  ...
  ```

  模型检查点保存在当前目录下。

- GPU处理器环境运行

  在运行train.py文件前，需要手动配置src/config.py文件中的pretrain_model参数、train_path参数和checkpoint_path参数，pretrain_model参数代表预训练权重模型路径，train_path参数代表训练集存放的位置，checkpoint_path参数代表存放生成得到的训练模型的位置。

  ```bash
  python train.py --device_id=0 --device_target="GPU"> train.log 2>&1 &
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

### Ascend分布式训练

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练 VID-youtube-bb 数据集

      ```python
      # (1) 在网页上设置 "is_cloudtrain=True"
      #     在网页上设置 "is_parallel=True"
      #     在网页上设置 "unzip_mode=1"(原始的数据集设置为0，tar压缩文件设置为1)
      #     在网页上设置 "data_url=/cache/data/ytb_vid_filter/"
      #     在网页上设置 其他参数
      # (2) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (3) 在网页上设置你的代码路径为 "/path/siamRPN"
      # (4) 在网页上设置启动文件为 "train.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

#### GPU分布式训练

  ```bash
  cd  SiamRPN      //进入到SiamRPN文件根目录

  bash scripts/run_distribute_train_gpu.sh DEVICE_NUM DEVICE_ LIST //运行脚本

  # DEVICE_NUM表示显卡数量
  # DEVICE_LIST: GPU处理器的id，需用户指定，例如“0,1,2,3”
  ```

## 评估过程

### 评估

#### 910评估

- 评估过程如下，需要vot数据集对应video的图片放于对应文件夹的color文件夹下，标签groundtruth.txt放于该目录下。

```bash
# 使用Ascend
  python eval.py --device_id=0 --dataset_path=/path/dataset --checkpoint_path=/path/ckpt/siamRPN-xx_xxxx.ckpt --filename=eval.json --device_target="Ascend"&> evallog &
```

- 上述python命令在后台运行，可通过`evallog`文件查看评估进程，结束后可通过`eval.json`文件查看评估结果。评估结果如下：

```bash
{... "all_videos": {"accuracy": 0.5809545709441025, "robustness": 0.33422978326730364, "eao": 0.3102655908013835}}
```

#### GPU评估

- 评估过程如下，需要vot数据集对应video的图片放于对应文件夹的color文件夹下，标签groundtruth.txt放于该目录下。

```bash
# 使用gpu
  python eval.py --device_id=0 --dataset_path=/path/dataset --checkpoint_path=/path/ckpt/siamRPN-xx_xxxx.ckpt --filename=eval.json --device_target="GPU"&> evallog &
```

- 上述python命令在后台运行，可通过`evallog`文件查看评估进程，结束后可通过`eval.json`文件查看评估结果。评估结果如下：

```bash
{... "all_videos": {"accuracy": 0.5826686315079969, "robustness": 0.2982987648566767, "eao": 0.3289693903290864}}
```

#### 310评估

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

- 评估过程如下，需要vot数据集对应video的图片放于对应文件夹的color文件夹下，标签groundtruth.txt放于该目录下，并到script目录。

```bash
# 使用数据集
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATA_NAME] [DEVICE_ID]
```

查看评估结果命令如下：

```bash
cat acc.log
```

# 模型描述

## 性能

### 训练性能

| 参数           | siamRPN(Ascend)                                  | siamRPN(GPU) |
| -------------------------- | ---------------------------------------------- | --------- |
| 模型版本                | siamRPN                                          | siamRPN |
| 资源                   | Ascend 910；CPU：2.60GHz，192核；内存：755 GB    | RTX3090 |
| 上传日期              | 2021-07-22                                           |   |
| MindSpore版本        | 1.2.0-alpha                                     |   |
| 数据集                |VID-youtube-bb                                     | VID-youtube-bb|
| 训练参数  |epoch=50, steps=1417, batch_size = 32                      | epoch=50, steps=1417, batch_size = 32  |
| 优化器                  | SGD                                               | SGD  |
| 损失函数 | 自定义损失函数 | 自定义损失函数 |
| 输出              | 目标框                                                |目标框  |
| 损失             |100~0.05                                          | 100~0.05     |
| 速度 | 8卡：625毫秒/步 | 8卡：296毫秒/步  |
| 总时长 | 8卡：12.3小时 | 8卡： 5.8小时|
| 调优检查点 |    247.58MB（.ckpt 文件）               | 247.44MB（.ckpt 文件）|
| 脚本                | [siamRPN脚本](https://gitee.com/mindspore/models/tree/r2.0/research/cv/siamRPN) | [siamRPN脚本](https://gitee.com/mindspore/models/tree/r2.0/research/cv/siamRPN) |

### 评估性能

| 参数  | siamRPN(Ascend)        | siamRPN(Ascend)     | siamRPN(GPU)         | siamRPN(GPU)                   |
| ------------------- | --------------------------- | --------------------------- |--------------------------- | --------------------------- |
| 模型版本      | simaRPN               | simaRPN          |simaRPN                       | simaRPN                       |
| 资源        | Ascend 910           | Ascend 910       |GPU         | GPU                       |
| 上传日期              | 2021-07-22         | 2021-07-22         |     2021-12-7      |         2021-12-7             |
| MindSpore版本   | 1.2.0-alpha                 | 1.2.0-alpha       |      1.5.0   |   1.5.0   |
| 数据集 | vot2015，60个video | vot2016，60个video |vot2015，60个video          | vot2016，60个video            |
| batch_size          |   1                |   1               |1           | 1                |
| 输出 | 目标框 | 目标框 |目标框             | 目标框        |
| 准确率 | 单卡：accuracy：0.58,robustness：0.33,eao:0.31; | 单卡：accuracy：0.56,robustness：0.39,eao:0.28;|单卡：accuracy：0.5826,robustness：0.298,eao:0.329;       | 单卡：accuracy：0.5538,robustness：0.345,eao:0.295;                  |

# 随机情况说明

在train.py中，我们设置了随机种子。

# ONNX推理

```bash
# 生成onnx文件
  python export.py --ckpt_file=/path/ckpt/siamRPN-xx_xxxx.ckpt
```

# onnx推理,根据vot2015数据集和vot2016数据集分别选择对应的onnx推理代码

# 例如选择vot2015数据集推理

```bash
python evalonnx.py --checkpoint_path=/path/siamrpn.onnx
或者 sh run_evalonnx.sh [dataset_path] [model_path] [filename]
#vot2016同理
```

-结果保存在filename当中，例如

```bash
{"all_videos": {"accuracy": 0.5890433443656077, "robustness": 0.3868562106735027, "eao": 0.30735406482761557}}
}
```

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
