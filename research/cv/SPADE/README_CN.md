# 目录

<!-- TOC -->

- [目录](#目录)
- [SPADE描述](#ADNet描述)
- [模型架构](#模型架构)
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
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## SPADE描述

SPADE是NVIDIA在2019年提出的语义图像合成算法,其论文发表在CVPR2019上,该算法提出了空间自适应归一化方法，用于在给定输入语义布局的情况下合成照片级真实感图像.

[论文](https://arxiv.org/abs/1903.07291)：Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. "Semantic Image Synthesis with Spatially-Adaptive Normalization.g". *Presented at CVPR 2019*.

## 模型架构

SPADE模型采用生成式对抗网络（GAN）作为网络主干。其中，生成器由一系列论文提出的 SPADE ResBlks 模块和最近邻上采样组成。判别器的体系结构遵循pix2pixHD方法中使用的体系结构，该方法使用多尺度设计的InstanceNorm，两者唯一的区别是SPADE将谱归一化（spectral norm）应用于所有的卷积层.

## 数据集

使用的数据集：[ADE20K]</br>
官网链接 </br>
ADE20K http://groups.csail.mit.edu/vision/datasets/ADE20K/

## 环境要求

- 硬件（GPU）
    - 准备GPU处理器搭建硬件环境
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后（模型在MindSpore1.5版本上经过检验），您可以按照如下步骤进行训练和评估：</br>

1. 训练开始前您得先安装依赖项和创建必要的文件夹：

    ```
    pip install -r requirements.txt
    mkdir vgg inception checkpoints
    ```

2. 然后下载pytorch版的inception预训练模型：https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth 到inception文件夹中，并执行以下命令来获取mindspore的vgg和inception预训练模型

    ```python
   # 转换vgg.pth，会在vgg目录下生成一个vgg19.ckpt
    python src/util/vgg19_pth2ckpt.py
    # 转换inception.pth,需提供inception.pth的路径 [inception_path]，会在inception目录下生成一个inception_pid.ckpt
    python src/util/inception_pth2ckpt.py inception/pt_inception-2015-12-05-6726825d.pth
    ```

3. 您可以按照如下步骤进行训练和评估：

    ```python
    # GPU下多卡训练
    bash ./scripts/run_distribute_train.sh vgg/vgg19.ckpt [data_root] [load_epoch]
    # GPU下单卡训练
    bash ./scripts/run_train.sh vgg/vgg19.ckpt [data_root] [load_epoch]
    # GPU下评估
    bash ./scripts/run_eval.sh [device_id] [eval_epoch] [load_ckpt] [DATA_ROOT]
    ```

其中 [data_root] 表示数据集的地址；[load_epoch] 表示需要加载哪个epoch的ckpt来训练，若从头开始训练，设为0即可；[device_id]表示评估所用的GPU设备id；[eval_epoch]表示需要评估的checkpoint的epoch数；[load_ckpt]表示该checkpoint所在地址。

具体参数细节可以参考`scripts/`目录下的可执行文件

## 脚本说明

### 脚本及样例代码

```text
├── SPADE
    ├── scripts
    │   ├──run_distribute_train.sh          // 在GPU中多卡训练
    │   ├──run_train.sh                     // 在GPU中单卡训练
    │   ├──run_eval.sh                      // 在GPU中单卡测试
    ├── src             //源码
    │   │   ├── data
    │   │   │   ├──__init__.py
    │   │   │   ├──ade20k_dataset.py
    │   │   │   ├──base_dataset.py
    │   │   │   ├──cityscapes_dataset.py
    │   │   │   ├──coco_dataset.py
    │   │   │   ├──custom_dataset.py
    │   │   │   ├──facades_dataset.py
    │   │   │   ├──image_folder.py
    │   │   │   ├──pix2pix_dataset.py
    │   │   ├── models
    │   │   │   ├──__init__.py
    │   │   │   ├──architecture.py
    │   │   │   ├──cells.py                //loss网络wrapper  
    │   │   │   ├──inception.py            //FID推理网络结构
    │   │   │   ├──init_Parameter.py       //参数初始化
    │   │   │   ├──loss.py                 //损失函数定义  
    │   │   │   ├──netD.py                 //判别器网络结构
    │   │   │   ├──netG.py                 //生成器网络结构
    │   │   │   ├──normalization.py        //自定义正则化
    │   │   │   ├──spectral_norm.py        //系谱归一化实现
    │   │   │   ├──vgg.py                  //损失函数网络结构
    │   │   ├── options
    │   │   │   ├──__init__.py
    │   │   │   ├──base_options.py
    │   │   │   ├──test_options.py
    │   │   │   ├──train_options.py
    │   │   ├── utils
    │   │   │   ├──adam.py                 //自定义优化器
    │   │   │   ├──coco.py
    │   │   │   ├──eval_fid.py             //fid精度测量
    │   │   │   ├──inception_pth2ckpt.py   //得到inception_fid.ckpt
    │   │   │   ├──instancenorm.py
    │   │   │   ├──lr_schedule.py          //自定义学习率策略
    │   │   │   ├──util.py                 //图片处理工具
    │   │   │   ├──vgg19_pth2ckpt.py       //得到vgg19.ckpt
    │   │   │   ├──visualizer.py
    ├── README_CN.md                       // SPADE相关说明
    ├── train.py                           // 训练入口
    ├── test.py                            // 评估入口
```

### 脚本参数

```text
共有参数
--batchSize:                     # 输入的batch大小
--dataroot：                     # 数据集根目录
--id:                            # 单卡执行时使用的物理卡号
--distribute:                    # 多卡运行
--checkpoints_dir                # checkpoints保存目录
--norm_G                         # 生成器中使用instance normalization 还是 batch normalization
--checkpoints_dir                # checkpoints保存目录

spade_train.py
--vgg_ckpt_path：                # vgg ckpt路径
--which_epoch                    # 选择加载哪个epoch的ckpt进行评估
--decay_epoch:                   # 学习率变化的起始epoch
--total_epoch:                   # 学习率变化的最终epoch
--now_epoch                      # 选择加载哪个epoch的ckpt继续训练，若为0，则不加载ckpt，重新开始训练
--beta1：                        # adam优化器beta1的值
--beta2：                        # adam优化器beta2的值
--G_lr：                         # 生成器起始学习率
--D_lr：                         # 判别器起始学习率

spade_run.py
--results_dir：                  # 运行结果保存的路径
--fid_eval_ckpt_dir：            # inception的ckpt文件保存路径
--which_epoch：                  # 评估的ckpt的epoch数
--ckpt_dir:                      # 需要评估的ckpt文件路径
```

更多配置细节请参考`src/options/`目录下的脚本

### 训练过程

#### 训练

- GPU处理器多卡环境运行

  ```python
  python train.py --distribute True --vgg_ckpt_path ./vgg/vgg19.ckpt --dataroot ADEChallengeData2016 --now_epoch 0
  # 或执行脚本
  bash ./scripts/run_distribute_train.sh [vgg_ckpt_path] [data_root] [load_epoch]
  ```

  经过训练后，损失值如下：

  ```text
  [189/200][101/315]: Loss_D: 0.845605 Loss_G: 26.058798
  [189/200][101/315]: Loss_D: 1.171079 Loss_G: 18.928753
  [189/200][101/315]: Loss_D: 0.983772 Loss_G: 24.714558
  [189/200][101/315]: Loss_D: 0.970536 Loss_G: 22.845257
  [189/200][101/315]: Loss_D: 1.284949 Loss_G: 25.437853
  [189/200][101/315]: Loss_D: 1.140993 Loss_G: 25.671362
  [189/200][101/315]: Loss_D: 1.350708 Loss_G: 21.344685
  [189/200][101/315]: Loss_D: 1.232908 Loss_G: 27.253286
  [189/200][101/315]: Loss_D: 0.993034 Loss_G: 24.286970
  ```

### 评估过程

#### 评估

- GPU处理器环境运行推理

  ```python
  bash ./scripts/run_eval.sh 0 200 ./checkpoints/netG_epoch_200.ckpt ./ADEChallengeData2016
  ```

-
  实际测试220epoch的FID精度为35.381

## 模型描述

### 性能

#### 评估性能

| 参数 | ModelArts
| -------------------------- | -----------------------------------------------------------
| 资源 | GPU RTX3090 * 8；CPU 3.36GHz, 64核；内存：270G
| 上传日期 | 2022-06-18
| MindSpore版本 | 1.5.0
| 数据集 | ADE20K
| 训练参数 | epoch=200, batch_size=8, D_lr=0.0004，G_lr=0.0001
| 损失函数 | L1Loss，vgg
| 损失 | g_loss:20左右, d_loss:1左右
| 速度 | 921毫秒/步
| 总时间 | 129小时
| 微调检查点 | 生成器: 368.62 MB （.ckpt文件） 判别器：22.34 MB （.ckpt文件）

## 随机情况说明

在base_dataset.py中，我们设置了“get_params”函数内的随机种子。

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)