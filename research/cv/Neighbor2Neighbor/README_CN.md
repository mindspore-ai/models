# Neighbor2Neighbor

<!-- TOC -->

- [Neighbor2Neighbor](#Neighbor2Neighbor)
- [Neighbor2Neighbor介绍](#Neighbor2Neighbor介绍)
- [模型结构](#模型结构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## Neighbor2Neighbor介绍

​    Neighbor2Neighbor:  Self-Supervised Denoising from Single Noisy Images. 是一种仅需要含噪图像即可训练任意降噪网络的方法。本方法是Noise2Noise的扩展，通过理论分析将Noise2Noise推广到了单张含噪图像和相似含噪图像这两个场景，并通过设计采样器的方式从单张含噪图像构造出相似含噪图像。随后通过引入正则项的方式解决了采样过程中相似含噪图像采样位置不同而导致的图像过于平滑的问题。本方法是一种训练策略，可以训练任意降噪网络而无需改造网络结构、无需估计噪声参数，也无需对输出图像进行复杂的后处理。

​    在 RGB 域合成数据上，对于 Gaussian 和 Poisson 噪声，每种噪声分别尝试了固定噪声水平和动态噪声水平两种情况。结果表明，在多个测试集上，本方法在性能上比使用配对数据训练的方法（N2C）低 0.3dB 左右，超越了现有的自监督降噪方法。在动态噪声水平的场景下，本方法显著超越其他自监督方法，甚至与自监督 + 后处理的 Laine19 不相上下，这更进一步说明了本方法的有效性。

[论文](https://arxiv.org/abs/2101.02824)：Huang T , Li S , Jia X , et al. Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images[J]. 2021.

## 模型结构

本方法不受限于具体模型结构。

训练策略上，从单张含噪图像通过采样器构造出两张子图 ，通过这两个子图构造重建损失函数；之后对原图进行推理降噪，得到的降噪图像再通过同样的采样过程生成两张子图，最后计算正则项。训练好的网络可直接用于图像降噪，无需进行后处理。

对于采样器，我们设计了近邻采样，即将图像划分成![[公式]](https://www.zhihu.com/equation?tex=2+%5Ctimes+2)的单元，在每个单元的四个像素中随机选择两个近邻的像素分别划分到两个子图中，这样构造出来两张"相似但不相同"的子图，我们称他们为"Neighbor"。

## 数据集

- 训练数据集：从ImageNet验证集中选取的尺寸在256x256px和 512x512px之间的共 44328 张图片。数据提取脚本：

  https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/main/dataset_tool.py

- 验证数据集：

  数据集路径：https://github.com/TaoHuang2018/Neighbor2Neighbor/tree/main/validation

  KODAK:   24张图片；

  BSD300：  100张图片；

  SET14：  12张图片；

## 环境要求

- 硬件（Ascend/ModelArts/GPU）
    - 准备Ascend或ModelArts或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

通过官方网站安装 MindSpore 后，您可以按照如下步骤进行训练和评估：

```bash
###参数配置请修改 default_config.yaml 文件

#启动训练：
#通过 python 命令行运行Ascend单卡训练脚本。
python train.py --is_distributed=0 --device_target=Ascend > train_ascend_log.txt 2>&1 &
#通过 python 命令行运行GPU单卡训练脚本。
python train.py --is_distributed=0 --device_target=GPU > train_gpu_log.txt 2>&1 &
#通过 bash 命令启动Ascend单卡训练。
bash ./scripts/run_train_ascend.sh device_id
e.g. bash ./scripts/run_train_ascend.sh 0
#通过 bash 命令启动GPU单卡训练。
bash ./scripts/run_train_gpu.sh device_id
e.g. bash ./scripts/run_train_gpu.sh 0

#Ascend多卡训练。
bash ./scripts/run_distribute_train_ascend.sh rank_size rank_start_id rank_table_file
e.g. bash ./scripts/run_distribute_train_ascend.sh 8 0 /data/hccl_8p.json
#GPU多卡训练。
bash ./scripts/run_distribute_train_gpu.sh

#启动推理
# default_config.yaml 文件中的 pretrain_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
# 通过 python 命令行运行Ascend推理脚本。
python eval.py --device_target=Ascend > eval_ascend_log.txt 2>&1 &
# 通过 python 命令行运行GPU推理脚本。
python eval.py --device_target=GPU > eval_gpu_log.txt 2>&1 &
#通过 bash 命令启动Ascend推理。
bash ./scripts/run_eval_ascend.sh device_id
e.g. bash ./scripts/run_eval_ascend.sh 0
#通过 bash 命令启动GPU推理。
bash ./scripts/run_eval_gpu.sh device_id
e.g. bash ./scripts/run_eval_gpu.sh 0
```

Ascend训练：生成[RANK_TABLE_FILE](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

## 脚本说明

### 脚本及样例代码

```tex
├── model_zoo
    ├── README.md                            // 所有模型的说明文件
    ├── Neighbor2Neighbor
        ├── README_CN.md                     // Neighbor2Neighbor 的说明文件
        ├── scripts
        │   ├──run_distribute_train_ascend.sh // Ascend 8卡训练脚本
        │   ├──run_distribute_train_gpu.sh   // GPU 8卡训练脚本
        │   ├──run_eval_ascend.sh            // Ascend 推理启动脚本
        │   ├──run_eval_gpu.sh               // GPU 推理启动脚本
        │   ├──run_train_ascend.sh           // Ascend 单卡训练启动脚本
        │   ├──run_train_gpu.sh              // GPU 单卡训练启动脚本
        ├── src
        │   ├──config.py                     // 配置加载文件
        │   ├──dataset.py                    // 数据集处理
        │   ├──models.py                     // 模型结构
        │   ├──logger.py                     // 日志打印文件
        │   ├──util.py                       // 工具类
        ├── default_config.yaml              // 默认配置信息，包括训练、推理、模型冻结等
        ├── train.py                         // 训练脚本
        ├── eval.py                          // 推理脚本
        ├── export.py                        // 将权重文件冻结为 MINDIR 等格式的脚本
```

### 脚本参数

```tex
模型训练、推理、冻结等操作及模型部署环境的参数均在 default_config.yaml 文件中进行配置。
关键参数默认如下：
noisetype: "gauss25"
n_feature: 48
n_channel: 3
lr: 3e-4
gamma: 0.5
epoch: 100
batch_size: 4
patchsize: 256
increase_ratio: 2.0
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```bash
  ###参数配置请修改 default_config.yaml 文件
  #通过 python 命令行运行Ascend单卡训练脚本。
  python train.py --is_distributed=0 --device_target=Ascend > train_ascend_log.txt 2>&1 &

  #通过 bash 命令启动Ascend单卡训练。
  bash ./scripts/run_train_ascend.sh device_id
  e.g. bash ./scripts/run_train_ascend.sh 0

  #Ascend多卡训练
  bash ./scripts/run_distribute_train_ascend.sh rank_size rank_start_id rank_table_file
  e.g. bash ./scripts/run_distribute_train_ascend.sh 8 0 /data/hccl_8p.json
  #Ascend多卡训练将会在代码根目录创建ascend_work_space文件夹，并在该工作目录下独立运行、保存相关训练信息。
  ```

  训练完成后，您可以在 output_path 参数指定的目录下找到保存的权重文件，训练过程中的部分 loss 收敛情况如下（8卡并行）：

  ```tex
  2021-11-01 19:33:16,671:INFO:epoch[0], iter[1330], loss:3368.929517, 366.12 imgs/sec, lr:0.0003
  2021-11-01 19:33:17,561:INFO:epoch[0], iter[1340], loss:3396.297974, 359.39 imgs/sec, lr:0.0003
  2021-11-01 19:33:18,432:INFO:epoch[0], iter[1350], loss:3541.782886, 367.64 imgs/sec, lr:0.0003
  2021-11-01 19:33:19,339:INFO:epoch[0], iter[1360], loss:3227.241211, 352.93 imgs/sec, lr:0.0003
  2021-11-01 19:33:20,214:INFO:epoch[0], iter[1370], loss:2993.920679, 365.87 imgs/sec, lr:0.0003
  2021-11-01 19:33:21,074:INFO:epoch[0], iter[1380], loss:3103.255444, 372.08 imgs/sec, lr:0.0003
  2021-11-01 19:33:22,941:INFO:load test weights from /opt/npu/data/luxuff/Neighbor2Neighbor/code/device0/output/unet_gauss25_b4e100r02_2021-11-01_time_19_28_32/ckpt_0/rank_0-1_1385.ckpt
  2021-11-01 19:33:23,169:INFO:loaded test weights from /opt/npu/data/luxuff/Neighbor2Neighbor/code/device0/output/unet_gauss25_b4e100r02_2021-11-01_time_19_28_32/ckpt_0/rank_0-1_1385.ckpt
  2021-11-01 19:35:06,667:INFO:Result in:/opt/npu/data/luxuff/Neighbor2Neighbor/test_dataset/Kodak
  2021-11-01 19:35:06,668:INFO:Before denoise: Average PSNR_b = 20.1734, SSIM_b = 0.3161;After denoise: Average PSNR = 28.5173, SSIM = 0.7326
  2021-11-01 19:35:06,668:INFO:testing finished....
  2021-11-01 19:35:06,668:INFO:time cost:101.5878632068634 seconds!
  2021-11-01 19:35:06,847:INFO:epoch[1], iter[0], loss:3164.208740, 257.58 imgs/sec, lr:0.0003
  2021-11-01 19:35:07,912:INFO:epoch[1], iter[10], loss:3439.687671, 300.52 imgs/sec, lr:0.0003
  2021-11-01 19:35:08,876:INFO:epoch[1], iter[20], loss:3154.986890, 332.05 imgs/sec, lr:0.0003
  2021-11-01 19:35:09,944:INFO:epoch[1], iter[30], loss:3273.177441, 299.91 imgs/sec, lr:0.0003
  2021-11-01 19:35:10,873:INFO:epoch[1], iter[40], loss:3339.457080, 344.40 imgs/sec, lr:0.0003
  2021-11-01 19:35:11,785:INFO:epoch[1], iter[50], loss:3451.508179, 350.94 imgs/sec, lr:0.0003
  2021-11-01 19:35:13,539:INFO:epoch[1], iter[60], loss:3344.975000, 343.29 imgs/sec, lr:0.0003
  2021-11-01 19:35:14,561:INFO:epoch[1], iter[70], loss:3131.437231, 313.12 imgs/sec, lr:0.0003
  2021-11-01 19:35:15,567:INFO:epoch[1], iter[80], loss:3136.780591, 318.17 imgs/sec, lr:0.0003
  2021-11-01 19:35:16,530:INFO:epoch[1], iter[90], loss:3601.045679, 332.69 imgs/sec, lr:0.0003
  2021-11-01 19:35:17,760:INFO:epoch[1], iter[100], loss:3244.493335, 312.30 imgs/sec, lr:0.0003
  2021-11-01 19:35:18,813:INFO:epoch[1], iter[110], loss:3406.463281, 303.87 imgs/sec, lr:0.0003
  2021-11-01 19:35:19,846:INFO:epoch[1], iter[120], loss:3017.756201, 310.16 imgs/sec, lr:0.0003
  2021-11-01 19:35:20,827:INFO:epoch[1], iter[130], loss:3185.839136, 331.49 imgs/sec, lr:0.0003
  2021-11-01 19:35:21,770:INFO:epoch[1], iter[140], loss:3179.780005, 339.76 imgs/sec, lr:0.0003
  2021-11-01 19:35:22,705:INFO:epoch[1], iter[150], loss:3281.860107, 342.37 imgs/sec, lr:0.0003
  2021-11-01 19:35:23,640:INFO:epoch[1], iter[160], loss:3148.518530, 342.49 imgs/sec, lr:0.0003
  2021-11-01 19:35:24,608:INFO:epoch[1], iter[170], loss:3570.691528, 330.59 imgs/sec, lr:0.0003
  2021-11-01 19:35:25,561:INFO:epoch[1], iter[180], loss:3377.788354, 336.09 imgs/sec, lr:0.0003
  2021-11-01 19:35:26,827:INFO:epoch[1], iter[190], loss:3527.461914, 327.03 imgs/sec, lr:0.0003
  2021-11-01 19:35:27,764:INFO:epoch[1], iter[200], loss:3316.355737, 341.83 imgs/sec, lr:0.0003
  2021-11-01 19:35:28,663:INFO:epoch[1], iter[210], loss:3899.881787, 356.10 imgs/sec, lr:0.0003
  2021-11-01 19:35:29,681:INFO:epoch[1], iter[220], loss:3490.284790, 314.26 imgs/sec, lr:0.0003
  2021-11-01 19:35:30,627:INFO:epoch[1], iter[230], loss:3348.329663, 338.70 imgs/sec, lr:0.0003
  2021-11-01 19:35:31,546:INFO:epoch[1], iter[240], loss:3355.376270, 348.10 imgs/sec, lr:0.0003
  ...
  ```

- GPU处理器环境运行

  ```bash
  ###参数配置请修改 default_config.yaml 文件
  #通过 python 命令行运行GPU单卡训练脚本。
  python train.py --is_distributed=0 --device_target=GPU > train_gpu_log.txt 2>&1 &

  #通过 bash 命令启动GPU单卡训练。
  bash ./scripts/run_train_gpu.sh device_id
  e.g. bash ./scripts/run_train_gpu.sh 0

  #GPU多卡训练。
  bash ./scripts/run_distribute_train_gpu.sh
  #GPU多卡训练将会在代码根目录创建gpu_work_space文件夹，并在该工作目录下独立运行、保存相关训练信息。
  ```

  训练完成后，您可以在 output_path 参数指定的目录下找到保存的权重文件(GPU多卡运行时所有代码会先复制一份到device目录，然后在这个目录下运行)，训练过程中的部分 loss 收敛情况如下（8卡并行）：

  ```tex
  ......
  2021-11-15 10:18:35,002:INFO:epoch[97], iter[1310], loss:7053.938965, 245.62 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:36,348:INFO:epoch[97], iter[1320], loss:7147.390430, 237.73 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:37,659:INFO:epoch[97], iter[1330], loss:6888.020801, 244.29 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:38,997:INFO:epoch[97], iter[1340], loss:7328.122461, 239.12 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:40,318:INFO:epoch[97], iter[1350], loss:6857.961377, 242.36 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:41,638:INFO:epoch[97], iter[1360], loss:7225.858057, 242.51 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:42,954:INFO:epoch[97], iter[1370], loss:6789.295264, 243.31 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:44,232:INFO:epoch[97], iter[1380], loss:6954.170557, 250.46 imgs/sec, lr:1.875e-05
  2021-11-15 10:18:44,836:INFO:Start to test on /data1/n2n/test_dataset/Kodak
  2021-11-15 10:18:52,336:INFO:Result in:/data1/n2n/test_dataset/Kodak
  2021-11-15 10:18:52,337:INFO:Before denoise: Average PSNR_b = 20.1729, SSIM_b = 0.3191;
  2021-11-15 10:18:52,337:INFO:After denoise: Average PSNR = 32.0001, SSIM = 0.8773
  2021-11-15 10:18:52,337:INFO:testing finished....
  2021-11-15 10:18:52,337:INFO:time cost:7.50101113319397 seconds!
  2021-11-15 10:18:52,337:INFO:Start to test on /data1/n2n/test_dataset/Set14
  2021-11-15 10:18:55,170:INFO:Result in:/data1/n2n/test_dataset/Set14
  2021-11-15 10:18:55,170:INFO:Before denoise: Average PSNR_b = 20.1706, SSIM_b = 0.3813;
  2021-11-15 10:18:55,170:INFO:After denoise: Average PSNR = 30.9913, SSIM = 0.8619
  2021-11-15 10:18:55,170:INFO:testing finished....
  2021-11-15 10:18:55,170:INFO:time cost:2.833129644393921 seconds!
  2021-11-15 10:18:55,171:INFO:Start to test on /data1/n2n/test_dataset/BSD300
  2021-11-15 10:19:04,709:INFO:Result in:/data1/n2n/test_dataset/BSD300
  2021-11-15 10:19:04,709:INFO:Before denoise: Average PSNR_b = 20.1711, SSIM_b = 0.3869;
  2021-11-15 10:19:04,709:INFO:After denoise: Average PSNR = 30.8586, SSIM = 0.8762
  2021-11-15 10:19:04,709:INFO:testing finished....
  2021-11-15 10:19:04,709:INFO:time cost:9.538610458374023 seconds!
  2021-11-15 10:19:04,721:INFO:Update newly best ckpt! best_value: 35.53895035799344
  2021-11-15 10:19:04,871:INFO:epoch[98], iter[0], loss:6756.173145, 250.80 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:06,429:INFO:epoch[98], iter[10], loss:6989.332422, 205.43 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:07,993:INFO:epoch[98], iter[20], loss:7170.686377, 204.80 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:09,551:INFO:epoch[98], iter[30], loss:7216.792285, 205.57 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:11,139:INFO:epoch[98], iter[40], loss:6964.292969, 201.65 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:12,672:INFO:epoch[98], iter[50], loss:6972.157861, 208.82 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:14,234:INFO:epoch[98], iter[60], loss:7075.860840, 204.91 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:15,784:INFO:epoch[98], iter[70], loss:6929.559766, 206.69 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:17,372:INFO:epoch[98], iter[80], loss:7098.844824, 201.72 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:18,933:INFO:epoch[98], iter[90], loss:7027.414697, 205.11 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:20,490:INFO:epoch[98], iter[100], loss:7030.924365, 205.87 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:22,047:INFO:epoch[98], iter[110], loss:6919.385596, 205.67 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:23,604:INFO:epoch[98], iter[120], loss:7414.254639, 205.54 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:25,147:INFO:epoch[98], iter[130], loss:7066.815283, 207.57 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:26,663:INFO:epoch[98], iter[140], loss:7203.811914, 211.23 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:28,229:INFO:epoch[98], iter[150], loss:7073.466602, 204.46 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:29,836:INFO:epoch[98], iter[160], loss:6908.576367, 199.24 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:31,421:INFO:epoch[98], iter[170], loss:7336.978125, 202.00 imgs/sec, lr:1.875e-05
  2021-11-15 10:19:32,976:INFO:epoch[98], iter[180], loss:7385.383936, 205.86 imgs/sec, lr:1.875e-05
  ......
  ```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于推理评估的权重文件路径是否正确。

- Ascend处理器环境运行

  ```bash
  ### 参数配置请修改 default_config.yaml 文件
  #  default_config.yaml 文件中的 pretrain_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
  #  test_dir 指测试数据集文件夹的根目录，而不应该是图片的根目录，即同时测试多个数据集
  # 通过 python 命令行运行Ascend推理脚本。
  python eval.py --device_target=Ascend > eval_ascend_log.txt 2>&1 &
  #通过 bash 命令启动Ascend推理。
  bash ./scripts/run_eval_ascend.sh device_id
  e.g. bash ./scripts/run_eval_ascend.sh 0
  ```

  运行完成后，您可以在 output_path 指定的目录下找到推理运行日志和去噪前后的图片。

- GPU处理器环境运行

  ```bash
  ### 参数配置请修改 default_config.yaml 文件
  #  default_config.yaml 文件中的 pretrain_path 指 ckpt 所在目录，为了兼容 modelarts，将其拆分为了 “路径” 与 “文件名”
  #  test_dir 指测试数据集文件夹的根目录，而不应该是图片的根目录，即同时测试多个数据集
  # 通过 python 命令行运行GPU推理脚本。
  python eval.py --device_target=GPU > eval_gpu_log.txt 2>&1 &
  #通过 bash 命令启动GPU推理。
  bash ./scripts/run_eval_gpu.sh device_id
  e.g. bash ./scripts/run_eval_gpu.sh 0
  ```

  运行完成后，您可以在 output_path 指定的目录下找到推理运行日志和去噪前后的图片。

### 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

#### 导出MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --dataset [DATASET]--file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数ckpt_file为必填项，
`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。
`DATASET` 使用的推理数据集名称，默认为`Kodak`，可在`Kodak`或者`BSD300`中选择。

#### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。
使用配置文件默认的export_batch_size导出MINDIR文件

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [INPUT_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件路径
- `DATASET` 使用的推理数据集名称，默认为`Kodak`，可在`Kodak`或者`BSD300`中选择
- `INPUT_PATH` 推理数据集路径
- `DEVICE_ID` 可选，默认值为0。

#### 结果

推理结果保存在脚本执行的当前路径，你可以在acc.log中看到以下精度计算结果。

```bash
# Kodak
Before denoise: Average PSNR_b = 20.1734, SSIM_b = 0.3186;
After denoise: Average PSNR = 32.1237, SSIM_b = 0.8788;
```

```bash
# BSD300
Before denoise: Average PSNR_b = 20.1734, SSIM_b = 0.3865;
After denoise: Average PSNR = 30.9367, SSIM_b = 0.8766;
```

## 模型描述

### 性能

#### 评估性能

Validation for Neighbor2Neighbor

| Parameters                 | Ascend                                                       | GPU                                                          |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores; Memory, 755G             | GeForce RTX 3090*8，Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz；Memory：256G |
| uploaded Date              | 11/1/2021 (month/day/year)                                   | 11/16/2021 (month/day/year)                                  |
| MindSpore Version          | 1.3.0                                                       | 1.3.0                                                       |
| Dataset                    | Kodak                                                        | Kodak                                                        |
| Training Parameters        | noisetype=gauss25, n_feature=48, n_channel=3, lr=3e-4, gamma=0.5, epoch=100, batch_size=4, patchsize=256, increase_ratio=2.0 | noisetype=gauss25, n_feature=48, n_channel=3, lr=3e-4, gamma=0.5, epoch=100, batch_size=4, patchsize=256, increase_ratio=2.0 |
| Optimizer                  | Adam                                                         | Adam                                                         |
| Loss Function              | 均方差的和(自定义loss，见models.py中的UNetWithLossCell)      | 均方差的和(自定义loss，见models.py中的UNetWithLossCell)      |
| outputs                    | image without noise                                          | image without noise                                          |
| Loss                       | 2994.17 ~ 6868.48                                            | 2918.00 ~ 7415.26                                            |
| Accuracy                   | PSNR = 32.1240, SSIM = 0.8863                                | PSNR = 32.0975, SSIM = 0.8795                                |
| Total time                 | 8p：3h50m (without validation)                               | 8p：5h54m (without validation)                               |
| Checkpoint for Fine tuning | 8p: 18.60MB(.ckpt file)                                      | 8p: 14.90MB(.ckpt file)                                      |
| Scripts                    | [Neighbor2Neighbor脚本](https://gitee.com/mindspore/models/tree/master/research/cv/Neighbor2Neighbor) | [Neighbor2Neighbor脚本](https://gitee.com/mindspore/models/tree/master/research/cv/Neighbor2Neighbor) |

## 随机情况说明

train.py 和 eval.py 中设置了随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
