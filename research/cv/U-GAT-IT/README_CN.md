[TOC]

# U-GAT-IT描述

U-GAT-IT提出了一种新的无监督的图像到图像的翻译方法。该方法以端到端的方式引入了一个新的注意力模块和一个新的可学习的归一化函数。注意力模块引导模型关注区分源域和目标域的重要区域 ，实现形状的变化。此外，适应性层-实例归一化（Adaptive Layer-Instance Normalization）帮助模型灵活地控制形状和纹理的变化。该模型能够实现真实人脸自拍图像到卡通人脸图像的转换。

[论文](https://openreview.net/attachment?id=BJlZ5ySKPH&name=original_pdf)：Kim J, Kim M, Kang H, et al. U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation[C]//International Conference on Learning Representations. 2019.

# 模型架构

两个生成器和两个判别器。

# 数据集

使用的数据集：[selfie2anime](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view)

- Dataset size：400M，7,000 256*256 JPG images
    - Train：6800 images (3400 selfie images and 3400 anime images respectively)
    - Test：200 images (100 selfie images and 100 anime images respectively)

```text

└─dataset
  └─selfie2anime
    └─trainA
    └─trainB
    └─testA
    └─testB
```

- 数据集解压缩后，将/dataset目录放至../U-GAT-IT/目录下

# 特性

# 环境要求

- 硬件（Ascend/GPU/CPU）
    - 使用Ascend/GPU/CPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 脚本说明

## 脚本及样例代码

```bash
├──U-GAT-IT
    ├── README_CN.md           # README
    ├── requirements.txt       # required modules
    ├── scripts                # shell script
        ├─run_standalone_train_910.sh     # training in standalone mode
        ├─run_distributed_train_910.sh    # training in parallel mode
        ├─run_eval_910.sh      # evaluation
        └─run_infer_310.sh     # 310 inference
    ├── src
        ├─dataset
            └─dataset.py       # data preprocess and loader
        ├─modelarts_utils      # utils for modelarts platform
        ├─models
            ├─cell.py          # trainers
            ├─networks.py      # generator and discriminator networks define
            └─UGATIT.py        # training and testing pipline
        ├─utils
            ├─args.py          # parse args
            └─tools.py         # utils for U-GAT-IT
        └─default_config.yaml  # configs for training and testing
    ├── ascend310_infer        # 310 inference
        ├─ src
            ├─ main.cc         # 310 inference
            └─ utils.cc        # utils for 310 inference
        ├─ inc
            └─ utils.h         # head file
        ├─ CMakeLists.txt      # compile
        ├─ build.sh            # script of main.cc
        └─ fusion_switch.cfg   # Use BatchNorm2d instead of InstanceNorm2d
    ├── train.py               # train launch file
    ├── eval.py                # eval launch file
    ├── export.py              # export checkpoints into mindir model
    ├── preprocess.py          # preprocess for 310 inference
    └── postprocess.py         # translate the result of 310 inference into jpg images
```

# 脚本参数

在../U-GAT-IT/src/defalt_config.yaml中可以同时配置训练参数和评估参数。

- 关键参数：

```python
    '--distributed', type=str2bool, default=False                                              # 是否是分布式训练
    '--output_path', type=str, default="results"                                               # 中间结果、模型保存路径
    '--phase', type=str, default='train', help='[train / test]'                                # 训练还是评估
    '--dataset', type=str, default='selfie2anime', help='dataset_name'                         # 数据集名字
    '--iteration', type=int, default=1000000, help='The number of training iterations'         # 总迭代次数
    '--batch_size', type=int, default=1, help='The size of batch size'                         # 批大小
    '--print_freq', type=int, default=1000, help='The number of image print freq'              # 中间结果保存频率
    '--save_freq', type=int, default=100000, help='The number of model save freq'              # 中间模型保存频率
    '--decay_flag', type=str2bool, default=True, help='The decay_flag'                         # 是否使用学习率衰减

    '--loss_scale', type=float, default=1024.0, help='The loss scale'                          # 为保证梯度精度，loss的放大倍数
    '--lr', type=float, default=0.0001, help='The learning rate'                               # 初始学习率
    '--weight_decay', type=float, default=0.0001, help='The weight decay'                      # 权重衰减值
    '--adv_weight', type=int, default=1, help='Weight for GAN'                                 # 对抗损失的权重
    '--cycle_weight', type=int, default=10, help='Weight for Cycle'                            # 循环一致性损失的权重
    '--identity_weight', type=int, default=10, help='Weight for Identity'                      # identity损失的权重
    '--cam_weight', type=int, default=1000, help='Weight for CAM'                              # 注意力模块损失权重
    '--ch', type=int, default=64, help='base channel number per layer'                         # 每层的通道数基数
    '--n_res', type=int, default=4, help='The number of resblock'                              # resblock的个数
    '--n_dis', type=int, default=6, help='The number of discriminator layer'                   # 判别器的层数
    '--img_size', type=int, default=256, help='The size of image'                              # 输入图像大小
    '--img_ch', type=int, default=3, help='The size of image channel'                          # 输入图像的通道数

    '--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU']            # 训练平台
    '--device_id', type=str, default='7', help='Set target device id to run'                   # 卡号
    '--resume', type=str2bool, default=False                                                   # 是否加载现有模型继续训练
    '--save_graphs', type=str2bool, default=False, help='Whether or not to save the graph'     # 是否保存训练图
    '--graph_path', type=str, default='graph_path', help='Directory name to save the graph'    # 图的存放路径

    '--genA2B_ckpt', type=str, default='./results/selfie2anime_genA2B_params_latest.ckpt'      # 310推理专用，生成器模型的储存路径
    '--MINDIR_outdir', type=str, default='./mindir_result'                                     # 310推理专用，mindir文件的导出路径
    "--export_file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR"      # 310推理专用，导出文件格式

    '--bifile_inputdir', type=str, default='./bifile_in'                                       # 310推理专用，二进制图像文件的输入路径
    '--bifile_outputdir', type=str, default='./bifile_out'                                     # 310推理专用，二进制图像文件的导出路径
    '--eval_outputdir', type=str, default='./infer_output_img'                                 # 310推理专用，后处理的JPG文件导出路径
```

# 训练和评估

进入目录../U-GAT-IT/，通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估（解压后的数据集/dataset需放至../U-GAT-IT/目录下）：

- Ascend 910处理器环境训练

```bash
# 单卡训练
bash ./scripts/run_standalone_train_910.sh [DEVICE_ID]

# 多卡训练
bash ./scripts/run_distributed_train_910.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START]
```

示例：

  ```bash
# 单卡训练
bash ./scripts/run_standalone_train_910.sh 0

# Ascend多卡训练（8P）
bash ./scripts/run_distributed_train_910.sh ./rank_table_8pcs.json 8 0
  ```

- Ascend 910处理器环境评估

```bash
# 评估
bash ./scripts/run_eva_910.sh [DEVICE_ID]
```

示例：

  ```bash
# 评估
bash ./scripts/run_eval_910.sh 0
  ```

- 在华为云ModelArts上训练

在ModelArts上训练需要提前配置好../U-GAT-IT/src/default_config.yaml文件的参数。请阅读华为云ModelArts的官方文档[modelarts](https://support.huaweicloud.com/modelarts/), 然后按照下面的提示进行训练：

```text
# 在 ModelArts 上使用单卡训练
# (1) 执行a或者b
#       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
#          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
#          在 default_config.yaml 文件中设置 "ckpt_path='/cache/train'"
#          在 default_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_path='/cache/data'"
#          在网页上设置 "ckpt_path='/cache/train'"
#          在网页上设置 其他参数
# (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
# (4) 上传原始数据集到 S3 桶上。
# (5) 在网页上设置你的代码路径为 "/path/U-GAT-IT"
# (6) 在网页上设置启动文件为 "train.py"
# (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (8) 创建训练作业
#
# 在 ModelArts 上使用八卡训练
# (1) 执行a或者b
#       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
#          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
#          在 default_config.yaml 文件中设置 "ckpt_path='/cache/train'"
#          在 default_config.yaml 文件中设置 "distributed=True"
#          在 default_config.yaml 文件中设置 其他参数
#       b. 在网页上设置 "enable_modelarts=True"
#          在网页上设置 "data_path='/cache/data'"
#          在网页上设置 "ckpt_path='/cache/train'"
#          在网页上设置 "distributed=True"
#          在网页上设置 其他参数
# (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
# (4) 上传原始数据集到 S3 桶上。
# (5) 在网页上设置你的代码路径为 "/path/U-GAT-IT"
# (6) 在网页上设置启动文件为 "train.py"
# (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
# (8) 创建训练作业
```

- 训练结果示例

```text
...
epoch 1:[ 1220/10200] time per iter: 0.6664
d_loss: 3.2190554
g_loss: 2914.113
epoch 1:[ 1221/10200] time per iter: 0.6666
d_loss: 3.6988006
g_loss: 115.64786
epoch 1:[ 1222/10200] time per iter: 0.6654
d_loss: 3.3407924
g_loss: 313.15454
epoch 1:[ 1223/10200] time per iter: 0.6641
d_loss: 2.4665039
g_loss: 163.84
epoch 1:[ 1224/10200] time per iter: 0.6639
d_loss: 4.269173
g_loss: 87.08121
epoch 1:[ 1225/10200] time per iter: 0.6648
d_loss: 2.621345
g_loss: 100.11629
...
 [*] Training finished!
```

- 评估结果示例

```text
Dataset load finished
build_model cost time: 104.3066
these params are not loaded:  {'genA2B': []}
 [*] Load SUCCESS
[WARNING] SESSION(6660,fffe647f41e0,python):2021-12-27-16:10:12.128.336 [mindspore/ccsrc/backend/session/ascend_session.cc:1806] SelectKernel] There are 23 node/nodes used reduce precision to selected the kernel!
 [*] Test finished!
```

# 310推理

在训练得到生成器的ckpt文件以后，可以将ckpt文件通过export.py导出为MINDIR格式文件并在Ascend 310上进行推理：

- 生成.mindir文件

  ```bash
  python export.py --genA2B_ckpt ./results/selfie2anime_genA2B_params_latest.ckpt --MINDIR_outdir ./mindir_result
  ```

- 执行310推理脚本

  ```bash
  bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]

  示例：

  bash scripts/run_infer_310.sh ./mindir_result/UGATIT_AtoB_graph.mindir ./dataset/selfie2anime/testA/ 0

  ```

# 模型描述

## 性能

### 评估性能

#### selfie2anime上的U-GAT-IT

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型名称              | U-GAT-IT                                                  |
| 资源                  | Ascend: 8 * Ascend-910(32GB) \| ARM: 192核 2048GB \| CentOS 4.18.0-193.el8.aarch64 |
| 上传日期              | 2021-12-23                                             |
| MindSpore版本         | 1.5.0                                                       |
| 数据集                | selfie2anime                                           |
| 训练参数              | epoch=100, batch_size = 1, lr=0.0001 |
| 优化器                | Adam                                                        |
| 损失函数              | 自定义损失函数                                                |
| 输出 | 图像 |
| 速度 | 640ms/step |
| 检查点                | 1.04GB, .ckpt文件                                      |

# 随机情况说明

 在main.py中设置了随机种子为1

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
