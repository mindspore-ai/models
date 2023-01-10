# 目录

- [目录](#目录)
- [PWCnet描述](#pwcnet描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [预训练](#预训练)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [运行示例](#运行示例)
        - [训练](#训练)
        - [评估](#评估)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo主页](#modelzoo主页)

# [PWCnet描述](#目录)

PWC-Net是根据简单并且完备的原则设计的。

[论文](https://arxiv.org/pdf/1709.02371.pdf)：Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"

# [模型架构](#目录)

PWCnet使用金字塔处理、图像扭转和使用代价体积。

# [数据集](#目录)

使用的训练集：[FlyingChairs](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip)
使用的评估数据集：[MPI-Sintel](https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip)）

我们使用大约22872张图像对以及相应的流场作为训练集，23个序列作为评估数据集，您也可以使用自己的数据集或其他开源数据集。

- 训练集的目录结构如下：

    ```bash
    .
    └─training
    ├── 00001_img1.ppm        //img1文件
    ├── 00001_img2.ppm        //img2文件
    ├── 00001_flow.flo       //flo文件
    │    ...
    ...
    ```

- 评估数据集的目录结构如下：

    ```bash
    .
    └─training
    ├── albedo
    ├── clean
    ├── final
        ├── alley_1
        ├── frame_0001.png
        ├── frame_0002.png
        ├── frame_0003.png
        ├── ....
        ├── frame_0050.png
        ├── ....
        ├── ....
    ├── flow
        ├── alley_1
        ├── frame_0001.flo
        ├── frame_0002.flo
        ├── ....
        ├── frame_0049.flo
    ├── flow_viz
    ├── invalid
    ├── occlusions
    ├── occlusions1
    ├── occlusions-clean
    │    ...
    ...
    ```

# [预训练](#目录)

- 下载预训练模型

    ```bash
    mkdir ./pretrained_model
    # 下载PWCNet预训练文件
    wget -O ./pretrained_model/pwcnet-pretrained.pth https://github.com/visinf/irr/blob/master/saved_check_point/pwcnet/PWCNet/checkpoint_best.ckpt
    ```

- 转换预训练模型（从PyTorch到MindSpore，必须同时安装Pytorch和MindSpore。）

    ```bash
    # 将PyTorch预训练模型文件转换为MindSpore文件。
    bash scripts/run_ckpt_convert.sh [PYTORCH_FILE_PATH] [MINDSPORE_FILE_PATH]
    ```

# [环境要求](#目录)

- 硬件
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

代码结构如下：

```bash
.
└─ PWCnet
  ├─ README.md
  ├─ model_utils
    ├─ __init__.py                          # 模型初始化文件
    ├─ config.py                            # 解析参数
    ├─ device_adapter.py                    # ModelArts设备适配器
    ├─ local_adapter.py                     # 本地适配器
    └─ moxing_adapter.py                    # ModelArts的装饰器
  ├─ scripts
    ├─ run_standalone_train.sh              # 在Ascend上进行单机训练（单卡）
    ├─ run_distribute_train.sh              # 在Ascend上进行分布式训练（8卡）
    ├─ run_eval.sh                          # 在Ascend上进行评估
    ├─ run_ckpt_convert.sh                  # 将PyTorch的CKPT文件转换为GPU上的PICKLE文件
  ├─ src
    ├─ sintel.py                            # 预处理用于评估的数据集
    ├─ common.py                            # 处理数据集
    ├─ transforms.py                        # 处理数据集
    ├─ flyingchairs.py                      # 预处理用于训练的数据集
    ├─ pwcnet_model.py                      # 主干网络
    ├─ pwc_modules.py                       # 主干网络
    ├─ log.py                               # 日志函数
    ├─ loss.py                              # 损失函数
    └─ lr_generator.py                      # 生成学习率
    ├─ utils
        ├─ ckpt_convert.py                  # 将PyTorch的CKPT文件转换为PICKLE文件
  ├─ default_config.yaml                    # 默认设置
  ├─ train.py                               # 训练脚本
  ├─ eval.py                                # 评估脚本
```

## [运行示例](#目录)

### 训练

- 单机训练模式（推荐）

    ```bash
    Ascend
    bash scripts/run_standalone_train.sh [TRAIN_LABEL_FILE] [EVAL_DIR] [DEVCIE_ID] [PRETRAINED_BACKBONE]
    ```

    例如，在Ascend上训练：

    ```bash
    bash scripts/run_standalone_train.sh ./data/FlyingChairs/ ./data/MPI-Sintel/ 0 ./pretrain.ckpt
    ```

- 分布式训练模式

    ```bash
    Ascend
    bash scripts/run_distribute_train.sh [TRAIN_LABEL_FILE] [EVAL_DIR] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

在**./output/[TIME]/[TIME].log**或者**./device0/train.log**中，得到执行每个步骤后的损失值：

```bash
epoch[0], iter[0],  0.01 imgs/sec   Loss 639.3672 639.3672
epoch[0], iter[20], 0.38 imgs/sec   Loss 68.1912 179.4218
epoch[0], iter[40], 25.73 imgs/sec   Loss 33.6643 50.4679
INFO:epoch[0], iter[80], 26.97 imgs/sec Loss 18.4107 23.8088

...
epoch[9], iter[55460], 27.81 imgs/sec   Loss 3.4625 2.4753
epoch[9], iter[55480], 29.96 imgs/sec   Loss 1.9749 2.3819
epoch[9], iter[55500], 29.11 imgs/sec   Loss 2.8970 2.7417
epoch[9], iter[55520], 27.98 imgs/sec   Loss 4.3935 2.9107
epoch[9], iter[55540], 27.83 imgs/sec   Loss 2.7637 2.3774
epoch[9], iter[55560], 28.15 imgs/sec   Loss 1.8070 2.4678
```

### 评估

```bash
Ascend

bash scripts/run_eval.sh [EVAL_DIR] [DEVCIE_ID] [PRETRAINED_BACKBONE]
```

例如，在Ascend上评估：

```bash
bash scripts/run_eval.sh ./data/MPI-Sintel/ 0 ./0-1_10000.ckpt
```

在**./device0/eval.log**中得到如下结果：

```bash
EPE: 6.9049
```

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend
| -------------------------- | ----------------------------------------------------------
| 模型版本             | V1
| 资源                  | Ascend 910；CPU 2.60GHz，192核；内存755G；EulerOS 2.8
| 上传日期             | 01/04/2022
| MindSpore版本         | 1.5.0
| 数据集                   | 22872个图像对
| 训练参数       | epoch=10, batch_size=4, momentum=0.9, lr=0.0001
| 优化器                 | Adam
| 损失函数             | MultiScaleEPE_PWC
| 输出                   | EPE
| 总时长                | 单卡：2.5小时；8卡：0.6小时

### 评估性能

| 参数         | Ascend
| ------------------- | -----------------------------
| 模型版本      | V1
| 资源           | Ascend 910；EulerOS 2.8
| 上传日期      | 01/04/2022
| MindSpore版本  | 1.5.0
| 数据集            | 133
| batch_size         | 1
| EPE                | 6.9049

# [ModelZoo主页](#目录)

请查看官方[主页](https://gitee.com/mindspore/models)。
