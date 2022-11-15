# 目录

<!-- TOC -->

- [目录](#目录)
- [APDrawingGAN描述](#apdrawinggan描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [使用的数据集](#使用的数据集)
    - [数据组织](#数据组织)
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
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
    - [ONNX模型导出及评估](#onnx模型导出及评估)
        - [ONNX模型导出](#onnx模型导出)
        - [ONNX模型评估](#onnx模型评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# APDrawingGAN描述

APDrawingGAN指的是APDrawingGAN: Generating Artistic Portrait Drawings from Face Photos with Hierarchical GANs，该网络的特点是可以生成非真实感的抽象艺术肖像画，既能捕捉到照片特征又和真实照片观感完全不同。

[论文](http://openaccess.thecvf.com/content_CVPR_2019/html/Yi_APDrawingGAN_Generating_Artistic_Portrait_Drawings_From_Face_Photos_With_Hierarchical_CVPR_2019_paper.html)：Yi R, Liu Y J, Lai Y K, et al. Apdrawinggan: Generating artistic portrait drawings from face photos with hierarchical gans[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 10743-10752.

[作者主页](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm)

# 模型架构

APDrawingGAN为G和D提出了分层的网络结构，包括一个全局网络和六个局部网络。六个局部网络分别负责左眼、右眼、鼻子、嘴、头发和背景。通过融合网络将全局网络和局部网络的输出进行整合。

# 数据集

## 使用的数据集

- 数据集大小：约204MB，共490张彩色图像
    - [训练集](http://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingDB.zip)：167MB，420张图像
    - [测试集](https://github.com/yiranran/APDrawingGAN/tree/master/dataset)：11.1MB，39张图像
- 数据格式：RGB图像

## 数据组织

将数据集dataset解压到任意路径，文件夹结构如下：

```bash
├── dataset
│   ├── data
│   │   ├── train
│   │   └── test
│   ├── landmark
│   │   └── ALL
│   └── mask
│       └── ALL
```

测试集文件夹结构如下：  

```bash
├── test_dataset
│   ├── data
│   │   └── test_single
│   ├── landmark
│   │   └── ALL
│   └── mask
│       └── ALL
```

auxiliary.ckpt文件获取：从 https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingGAN-Models2.zip 下载后解压得到auxiliary中的四个pth，通过src/utils/convert_apdrawinggan.py脚本转化为ckpt文件。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 运行

  ```bash
  # 运行训练示例
  bash scripts/run_train.sh [DATA_PATH] [LM_PATH] [BG_PATH] [CKPT_PATH] [AUXILIARY_PATH] [DEVICE_ID] [EPOCH] [SAVA_EPOCH_FREQ] [DEVICE_TARGET]
  # 例如：
  bash scripts/run_train.sh dataset/data/train dataset/landmark/ALL dataset/mask/ALL checkpoint auxiliary/auxiliary.ckpt 0 300 25 GPU

  # 运行分布式训练示例Ascend
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH] [LM_PATH] [BG_PATH] [CKPT_PATH] [AUXILIARY_PATH] [EPOCH] [SAVA_EPOCH_FREQ]
  # 例如：
  bash scripts/run_distribute_train.sh hccl_8p.json dataset/data/train dataset/landmark/ALL dataset/mask/ALL checkpoint auxiliary/auxiliary.ckpt 300 25

  # 运行分布式训练示例GPU
  bash scripts/run_train_distribute_GPU.sh [DATA_PATH] [LM_PATH] [BG_PATH] [CKPT_PATH] [AUXILIARY_PATH] [EPOCH] [SAVA_EPOCH_FREQ]
  # 例如：
  bash scripts/run_train_distribute_GPU.sh /dataset/train/data /dataset/train/landmark/ALL dataset/train/mask/ALL checkpoint /auxiliary/pretrain_APDGAN.ckpt 300 25"

  # 运行评估示例
  bash scripts/run_eval.sh [DATA_PATH] [LM_PATH] [BG_PATH] [RESULT_PATH] [MODEL_PATH] [DEVICE_TARGET]
  # 例如：
  bash scripts/run_eval.sh test_dataset/data/test_single/ test_dataset/landmark/ALL test_dataset/mask/ALL test_result checkpoint/netG_300.ckpt GPU
  ```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用8卡训练

      ```python
      # (1) 在网页上设置 "isModelarts=True"
      #     在网页上设置 "run_distribute=True"
      # (2) 上传你的数据集到obs桶上，设置文件夹名为data
      # (3) 把auxiliary.ckpt文件和config_train.yaml文件放到data文件夹下
      # (4) 在网页上设置你的代码路径为 "/ap-drawing-db/code/"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡验证

      ```python
      # (1) 在网页上设置 "isModelarts=True"
      #     在网页上设置 "model_path=netG_300.ckpt"
      # (2) 上传你的测试数据集到obs桶上，设置文件夹名为test_data
      # (3) 把netG_300.ckpt文件和config_eval_and_export.yaml文件放到test_data文件夹下
      # (4) 在网页上设置你的代码路径为 "/ap-drawing-db/code/"
      # (5) 在网页上设置启动文件为 "eval.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业
      ```

    - 在 ModelArts 上使用单卡导出mindir文件

      ```python
      # (1) 在网页上设置 "isModelarts=True"
      #     在网页上设置 "model_path=netG_300.ckpt"
      # (2) 把netG_300.ckpt文件和config_eval_and_export.yaml文件放到test_data文件夹下
      # (3) 在网页上设置你的代码路径为 "/ap-drawing-db/code/"
      # (4) 在网页上设置启动文件为 "export.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

# 脚本说明

## 脚本及样例代码

```bash
.
├── APDrawingGAN
    ├─ README_CN.md                        # 模型相关说明
    ├─ config_eval_and_export.yaml         # 评估和导出超参数设置
    ├─ config_train.yaml                   # 训练超参数设置
    ├─ eval.py                             # 评估脚本
    ├─ eval_onnx.py                        # ONNX模型评估脚本
    ├─ train.py                            # 训练脚本
    ├─ export.py                           # 模型导出脚本
    ├─ export_onnx.py                      # ONNX模型导出脚本
    ├─ auxiliary                           # 预训练权重值
    │  └─ auxiliary.ckpt                   # 预训练模型ckpt
    ├─ dataset
    │  ├─ data                             # 训练图像
    │  │  ├─ train                         # 训练图像
    │  │  └─ test                          # 测试样例
    │  ├─ landmark                         # 图像的脸部特征数据
    │  │  └─ ALL
    │  └─ mask                             # 背景轮廓图像
    │     └─ ALL
    ├─ test_dataset
    │  ├─ data                             # 测试图像
    │  │  └─ test_single                   # 测试图像
    │  ├─ landmark                         # 图像的脸部特征数据
    │  │  └─ ALL
    │  └─ mask                             # 背景轮廓图像
    │     └─ ALL
    ├─ scripts
    │  ├─ run_eval.sh                      # 启动评估
    │  ├─ run_eval_onnx.sh                 # 启动对导出的onnx模型的评估
    │  ├─ run_distribute_train.sh          # 启动多卡训练
    │  ├─ run_train.sh                     # 启动单卡训练
    │  └─ run_infer_310.sh                 # 实现310推理源代码
    ├─ ascend310_infer
    └─ src
       ├─ data                             # 数据处理
       │  ├─ aligned_dataset.py            # 生成训练数据集
       │  ├─ base_dataset.py               # 数据集基类
       │  ├─ base_dataloader.py            # 数据加载基类
       │  ├─ single_dataloader.py          # 加载测试数据
       │  ├─ single_dataset.py             # 生成测试数据集
       │  └─ __init__.py
       ├─models
       │  ├─ APDrawingGAN.py               # APDrawingGAN模型
       │  ├─ APDrawingGAN_D.py             # 判别器模型
       │  ├─ APDrawingGAN_G.py             # 生成器模型
       │  ├─ APDrawingGAN_WithLossCellD.py # 判别器损失函数
       │  └─ APDrawingGAN_WithLossCellG.py # 生成器损失函数
       ├─networks
       │  ├─ controller.py                 # 生成指定网络
       │  ├─ layer_func.py                 # 生成指定的标准化层
       │  ├─ networks_block.py             # 一些网络模块
       │  ├─ networks_D.py                 # 判别器网络
       │  ├─ networks_G.py                 # 生成器网络
       │  ├─ networks_loss.py              # GAN损失函数
       │  └─ netwroks_init.py              # 网络初始化
       ├─utils
       │  ├─ tools.py                      # modelarts读入和保存数据
       │  └─ convert_apdrawinggan.py       # 预训练权重值转换
       └─option
          ├─ options.py                    # 训练阶段参数
          ├─ config.py                     # 配置文件
          └─ options_test.py               # 评估阶段参数
```

## 脚本参数

在config_train.yaml中可以配置训练参数和评估参数。

  ```python
  'pre_trained':'True'
  'lr_'=0.0002          # 学习速率
  'niter'=300           # 轮次数
  'lambda_l1'=100       # 损失权重值
  'lambda_local'=25     # 局部损失权重值
  'lambda_chamfer'=0.1  # DT1损失权重值
  'lambda_chamfer2'=0.1 # DT2损失权重值
  'beta1'=0.5           # Adam优化器的beta1
  ```

更多配置细节请参考config_train.yaml。

## 训练过程

### 训练

- GPU处理器环境运行

  ```bash
  bash scripts/run_train.sh dataset/data/train dataset/landmark/ALL dataset/mask/ALL checkpoint auxiliary/auxiliary.ckpt 0 300 25 GPU
  ```

  或

  ```bash
  python train.py --device_id=0 --device_target=GPU --dataroot=dataset/data/train/ --lm_dir=dataset/landmark/ALL/ --bg_dir=dataset/mask/ALL/ --auxiliary_dir=auxiliary/auxiliary.ckpt --ckpt_dir=checkpoint --niter=300 --save_epoch_freq=25 --use_local --discriminator_local --no_flip --no_dropout  --pretrain --isTrain
  ```

- Ascend处理器环境运行

  ```bash
  bash scripts/run_train.sh dataset/data/train dataset/landmark/ALL dataset/mask/ALL checkpoint auxiliary/auxiliary.ckpt 0 300 25 Ascend
  ```

  或

  ```bash
  python train.py --device_id=0 --device_target=Ascend --dataroot=dataset/data/train/ --lm_dir=dataset/landmark/ALL/ --bg_dir=dataset/mask/ALL/ --auxiliary_dir=auxiliary/auxiliary.ckpt --ckpt_dir=checkpoint --niter=300 --save_epoch_freq=25 --use_local --discriminator_local --no_flip --no_dropout  --pretrain --isTrain
  ```

  ```bash
  用法：train.py [--dataroot DATAROOT] [--ckpt_dir CKPT_DIR][--device_target GPU]
                [--auxiliary_dir AUXILIARY_DIR]
                [--mindrecord_dir MINDRECORD_DIR] [--lm_dir LM_DIR]
                [--bg_dir BG_DIR][--batch_size BATCH_SIZE]
                [--device_id DEVICE_ID][--use_local] [--no_flip]
                [--save_epoch_freq SAVE_EPOCH_FREQ]
                [--niter NITER][--discriminator_local]
                [--no_dropout]
                [--num_parallel_workers NUM_PARALLEL_WORKERS]
                [--isTrain]
                [--run_distribute BOOLEAN] [--device_id DEVICE_ID]

  选项：
    --dataroot                        图片数据路径
    --device_target                   运行的处理器,根据需要进行修改
    --mindrecord_dir                  数据下沉的数据文件
    --lm_dir                          脸部特征点路径
    --bg_dir                          背景图片路径
    --auxiliary_dir                   预训练模型的保存路径
    --ckpt_dir                        训练模型的保存路径
    --batch_size                      批大小
    --device_id                       device_id
    --use_local                       使用局部生成器
    --discriminator_local             使用局部鉴别器
    --niter                           训练的epoch数
    --save_epoch_freq                 保存的频率
    --no_flip                         不翻转图像
    --no_dropout                      不使用dropout
    --isTrain                         训练
    --run_distribute                  多卡并行训练
    --num_parallel_workers            并行工作数量
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认`./checkpoint/`脚本文件夹下找到检查点文件。

### 分布式训练

- GPU处理器环境运行

    ```bash
    bash scripts/run_train_distribute_GPU.sh dataset/data/train dataset/landmark/ALL dataset/mask/ALL checkpoint auxiliary/pretrain_APDGAN.ckpt 300 25
    ```

- Ascend处理器环境运行

  ```bash
  bash scripts/run_distribute_train.sh hccl_8p_01_127.0.0.1.json dataset/data/train dataset/landmark/ALL dataset/mask/ALL checkpoint auxiliary/auxiliary.ckpt 300 25
  ```

  上述shell脚本将在后台运行分布式训练。您可以通过train_parallel[X]/train.log文件查看结果。

## 评估过程

### 评估

- 评估dataset数据集
  运行评估脚本，对用户指定的测试数据集进行测试，测试数据集包括测试图片、测试图片的背景轮廓图片及测试图片的脸部特征点数据，最终会给每张测试图片生成对应的肖像画图片。在运行以下命令前，请检查各数据源的路径是否正确。

  ```bash
  python eval.py  --dataroot=test_dataset/data/test_single
                  --device_target=GPU
                  --lm_dir=test_dataset/landmark/ALL
                  --bg_dir=test_dataset/mask/ALL
                  --results_dir=test_dataset/result
                  --model_path=checkpoint/netG_300.ckpt > eval.log 2>&1 &
  ```

  或者，

  ```bash
  bash scripts/run_eval.sh test_dataset/data/test_single test_dataset/landmark/ALL test_dataset/mask/ALL test_dataset/result checkpoint/netG_300.ckpt GPU
  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看评估过程。测试数据生成的对应肖像画图片保存在指定的结果目录中，例如：上述指令指定的 test_dataset/result目录中。

## 导出过程

### 导出

  ```bash
  python export.py --model_path=checkpoint/netG_300.ckpt
  ```

执行完后会在当前路径生成infer_model.mindir文件。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在进行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。

- 在昇腾310上进行推理

  进入scripts目录，执行下面的命令，开始推理。

  ```bash
  # Ascend310 inference
  bash run_infer_310.sh [GEN_MINDIR_PATH] [DATA_PATH] [LM_PATH] [BG_PATH] [NEED_PREPROCESS] [DEVICE_ID]
  # 例如：
  bash run_infer_310.sh infer_model_tran_test.mindir ../test_dataset/data/test_single/ ../test_dataset/landmark/ALL/ ../test_dataset/mask/ALL/ y 0
  ```

  推理的结果保存在当前目录result_Files下，在time_Result/test_perform_static.txt文件中可以看到推理性能。

  ```bash
  NN inference cost average time: 168.92 ms of infer_count 39
  ```

## ONNX模型导出及评估

### ONNX模型导出

因为在该模型评估时，对于每张图片的处理，网络需要根据对应图片的center信息进行set_pad设置，所以需要对于评估集中每张图片对应的网络分别导出一个onnx模型文件，进行该图片的onnx推理。
所以下述onnx导出指令将会生成一系列的onnx文件。

  ```bash
  python export_onnx.py  --model_path=checkpoint/netG_300.ckpt
                         --dataroot=test_dataset/data/test_single
                         --lm_dir=test_dataset/landmark/ALL
                         --bg_dir=test_dataset/mask/ALL
  ```

执行完后，会在当前路径生成一系列以每张图片的center信息命名的onnx模型文件，提供给onnx模型评估使用。

### ONNX模型评估

  运行评估脚本，对用户指定的测试数据集进行测试，测试数据集包括测试图片、测试图片的背景轮廓图片及测试图片的脸部特征点数据，最终会给每张测试图片生成对应的肖像画图片。在运行以下命令前，请检查各数据源的路径是否正确。

  ```bash
  python eval_onnx.py  --dataroot=test_dataset/data/test_single
                       --lm_dir=test_dataset/landmark/ALL
                       --bg_dir=test_dataset/mask/ALL
                       --results_dir=test_dataset/result
                       --onnx_path=./ > eval_onnx.log 2>&1 &
  ```

  或者，

  ```bash
  bash scripts/run_eval_onnx.sh [DATA_PATH] [LM_PATH] [BG_PATH] [RESULT_PATH] [ONNX_PATH]
  # example: bash scripts/run_eval_onnx.sh test_dataset/data/test_single/ test_dataset/landmark/ALL test_dataset/mask/ALL result_onnx onnx_file/
  ```

  注意，其中[ONNX_PATH]指的是前一步导出的一系列onnx模型文件的目录。

  上述python命令将在后台运行，您可以通过eval_onnx.log文件查看评估过程。测试数据生成的对应肖像画图片保存在指定的结果目录中，例如：上述指令指定的 result_onnx 目录中。

# 模型描述

## 性能

### 训练性能

| 参数                      | Ascend               |GPU               |
| --------------------------| ---------------------- | ---------------------- |
| 模型版本                  | APDrawingGAN           |APDrawingGAN           |
| 资源                      | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |GPU(Tesla V100-SXM2 32G)；CPU：3.0GHz 36cores ；RAM：0.5T|
| 上传日期                  | 2021-12-09      |2022-03-07      |
| MindSpore版本             | 1.3.0              |1.5.0              |
| 数据集                    |          APDrawingDB     |APDrawingDB     |
| 训练参数                  | epoch=300, lr=0.0002, bata1=0.5             |epoch=300, lr=0.0002, bata1=0.5             |
| 优化器                    | Adam                                                    |Adam                   |
| 损失函数                  | Distance transformer loss & L1 loss                                    |Distance transformer loss & L1 loss    |
| 输出                      | 图片                                                         |图片  |
| 损失                      |GANLoss,L1Loss,localLoss,DTLoss|GANLoss,L1Loss,localLoss,DTLoss|
| 速度                      | 单卡：357毫秒/步;  8卡：380毫秒/步                      |单卡: 442毫秒/步;8卡：506毫秒/步|
| 总时长                    | 单卡：750分钟;  8卡：120分钟                      |单卡：920分钟 ; 8卡：140分钟 |
| 微调检查点                | 243.37MB (.ckpt文件)                             |273.43MB (.ckpt文件)|
| 推理模型                  | 250.53M(.mindir) |
| 脚本                      | [APDrawingGAN脚本](https://gitee.com/yang-mengYM/models_1/tree/master/research/cv/APDrawingGAN) |

### 评估性能

| 参数          | Ascend                      |GPU       |
| ------------------- | --------------------------- |--------------------------- |
| 模型版本       | APDrawingGAN                |APDrawingGAN  |
| 资源            |  Ascend 910；系统 Euler2.8                  |GPU(Tesla V100-SXM2 32G)；CPU：3.0GHz 36cores ；RAM：0.5T |
| 上传日期       | 2021-12-09 |2022-03-07      |
| MindSpore 版本   | 1.3.0                       |1.5.0|
| 数据集             | APDrawingDB |APDrawingDB |
| batch_size          | 1                         |1   |
| 输出             | 图片      | 图片      |
| 推理模型 | 250.53M(.mindir) |

# 随机情况说明

使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。