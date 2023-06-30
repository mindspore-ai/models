# 目录

# 模型名称

> Segformer

Segformer是由香港大学谢恩泽等人提出的一个简单，高效而强大的语义分割框架，此模型将transformer与轻量级多层感知(MLP)解码器统一起来。
SegFormer的两个特点:1)SegFormer包括一个新颖的层次结构编码器，输出多尺度特征。它不需要位置编码，从而避免了位置编码的插值，避免了当测试分辨率与训练不同时，会导致性能下降问题。2) SegFormer避免了复杂的解码器。

## 论文

[论文链接](https://arxiv.org/abs/2105.15203)

## 模型架构

Segformer的总体网络架构如下：
[链接](https://arxiv.org/abs/2105.15203)

## 数据集

使用的数据集：[Cityscapes](https://www.cityscapes-dataset.com/downloads/) 需要注册账号后下载

- 数据集大小：共19个类、5000个1024*2048彩色图像
    - 训练集：2975个图像
    - 评估集：500个图像
    - 测试集：1525个图像
- 数据格式：RGB
- 目录结构如下：

```txt
segformer
├── src
├── scripts
├── config
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```

注意：需要将`**polygons.json`转换为`**labelTrainIds.png`文件后才能用于训练，可以使用convert_dataset.py脚本进行转换，脚本基于[cityscapesscripts](https://github.com/mcordts/cityscapesScripts) 开发

```shell
python tools/convert_dataset.py data/cityscapes --nproc 8
```

## 特性

采用 [混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html) 的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

## 环境要求

- 硬件(Ascend)
    - 准备Ascend搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速入门

> 在训练之前，您需要先转换数据集格式、预训练模型和生成组网信息文件

- 转换数据集格式
    - 如果您使用Cityspaces数据集，在训练之前需要转换数据集格式
    - 您可以使用convert_dataset.py脚本，将`**polygons.json`转换为`**labelTrainIds.png`
    - 其中，data/cityscapes为数据集路径，nproc为并行执行的进程数

```shell
python tools/convert_dataset.py data/cityscapes --nproc 8
```

- 转换预训练模型
    - 您可以使用convert_model.py脚本，将论文作者提供的pytorch预训练模型转换为mindspore预训练模型
    - 下载论文作者提供的预训练模型([google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing)|[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw))
    - 其中，pt_model_path为pytorch模型路径，ms_model_path为mindspore模型保存路径

```shell
python tools/convert_model.py --pt_model_path=./pretrained/mit_b0.pth --ms_model_path=./pretrained/ms_pretrained_b0.ckpt
```

- 生成组网信息文件
    - 如果使用分布式训练，需要使用脚本[hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) 脚本生成组网信息文件：

```shell  
# 参数[0,8)表示生成0~7号卡的组网信息文件
python hccl_tools.py --device_num "[0,8)"
```

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE]

# 单机训练
用法：bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE]

# 运行评估示例
用法：bash run_eval.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH]
```

注意，如果运行上述快速启动命令，数据集和预训练模型需要放在默认路径，参考如下：

```txt
segformer
├── src
├── scripts
├── config
├── data                    # 数据集目录
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
├── pretrained              # 预训练模型目录
```

## 脚本说明

### 脚本和样例代码

```log
└──segformer
   ├── config                                    # 配置文件目录
       ├── segformer.b0.512x1024.city.yaml       # mit_b0规格配置文件
       ├── segformer.b0.1024x1024.city.yaml
       ├── segformer.b1.1024x1024.city.yaml
       ├── segformer.b2.1024x1024.city.yaml
       ├── segformer.b3.1024x1024.city.yaml
       ├── segformer.b4.1024x1024.city.yaml
       ├── segformer.b5.1024x1024.city.yaml
       ├── segformer.base.yaml                   # 公共参数配置文件
   ├── scripts                                   # 脚本目录
       ├── run_distribute_train.sh               # 运行多卡训练脚本
       ├── run_eval.sh                           # 运行评估脚本
       ├── run_infer.sh                          # 运行推理脚本
       ├── run_standalone_train.sh               # 运行单卡训练脚本
   ├── src
       ├── model_utils
           ├── config.py                         # 配置文件脚本
       ├── base_dataset.py                       # 数据集处理脚本
       ├── dataset.py                            # Cityscapes数据集处理脚本
       ├── loss.py                               # 定义loss函数
       ├── mix_transformer.py                    # 定义backbone，支持b0到b5
       ├── optimizers.py                         # 定义优化器
       ├── segformer.py                          # 定义整体网络
       ├── segformer_head.py                     # 定义head网络
   ├── tools
       ├── convert_dataset.py                    # 转换数据集格式脚本
       ├── convert_model.py                      # 转换预训练模型脚本
   ├── eval.py                                   # 评估脚本
   ├── infer.py                                  # 推理脚本
   ├── train.py                                  # 训练脚本
   ├── README.md
   ├── requirement.txt                           # 第三方依赖
```

### 脚本参数

```shell
run_distribute: False                                         # 是否是分布式训练
data_path: "./data/cityscapes/"                               # 数据集路径
load_ckpt: True                                               # 是否加载预训练模型
pretrained_ckpt_path: "./pretrained/ms_pretrained_b0.ckpt"    # 预训练模型存放路径
save_checkpoint: True                                         # 是否保存模型
save_checkpoint_epoch_interval: 1                             # 保存模型的epoch间隔
save_best_ckpt: True                                          # 是否保存最好的模型
checkpoint_path: "./checkpoint/"                              # 保存模型的路径
train_log_interval: 100                                       # 打印训练日志的step间隔

epoch_size: 200                                               # 训练epoch大小
batch_size: 2                                                 # 输入张量的批次大小
backbone: "mit_b0"                                            # backbone网络，支持mit_b0到mit_b5
class_num: 19                                                 # 数据集类别数
dataset_num_parallel_workers: 4                               # 并行处理数据集线程数
momentum: 0.9                                                 # 动量优化器
lr: 0.0001                                                    # 学习率
weight_decay: 0.01                                            # 权重衰减
base_size: [512, 1024]                                        # 基本图片大小，原始图片会先resize为此大小
crop_size: [512, 1024]                                        # 裁剪大小，图片经过裁剪后输入的网络
img_norm_mean: [123.675, 116.28, 103.53]                      # 图片正则化的均值
img_norm_std: [58.395, 57.12, 57.375]                         # 图片正则化的方差

run_eval: True                                                # 是否在训练中评估模型
eval_start_epoch: 0                                           # 开始评估模型的epoch数
eval_interval: 1                                              # 评估模型的epoch间隔数
eval_ckpt_path: ""                                            # 评估模型的存放路径
eval_log_interval: 100                                        # 评估模型日志打印的step间隔数

infer_copy_original_img: True                                 # 执行推理时是否复制原图片到推理结果目录
infer_save_gray_img: True                                     # 执行推理时是否保存灰度图
infer_save_color_img: True                                    # 执行推理时是否保存彩色图
infer_save_overlap_img: True                                  # 执行推理时是否保存彩色图与原图片的重叠图片
infer_log_interval: 100                                       # 推理日志打印step间隔数
infer_ckpt_path: ""                                           # 推理用的模型存放路径
infer_output_path: "./infer_result/"                          # 推理结果存放路径
```

## 训练过程

> 在训练之前，您需要先转换数据集格式和转换预训练模型

- 转换数据集格式
    - 如果您使用Cityspaces数据集，在训练之前需要转换数据集格式
    - 您可以使用convert_dataset.py脚本，将`**polygons.json`转换为`**labelTrainIds.png`
    - 其中，data/cityscapes为数据集路径，nproc为并行执行的进程数

```shell
python tools/convert_dataset.py data/cityscapes --nproc 8
```

- 转换预训练模型
    - 您可以使用convert_model.py脚本，将论文作者提供的pytorch预训练模型转换为mindspore预训练模型
    - 下载论文作者提供的预训练模型([google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing)|[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw))
    - 其中，pt_model_path为pytorch模型路径，ms_model_path为mindspore模型保存路径

```shell
python tools/convert_model.py --pt_model_path=./pretrained/mit_b0.pth --ms_model_path=./pretrained/ms_pretrained_b0.ckpt
```

### 训练

在昇腾上使用单卡训练运行下面的命令

```shell
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE]
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional)
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional)
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional) [RUN_EVAL](optional)
For example: bash run_standalone_train.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/data/cityscapes/"
```

也可以直接运行python脚本

```shell
python train.py --config_path=[CONFIG_FILE] --pretrained_ckpt_path=[PRETRAINED_CKPT_PATH](optional) --data_path=[DATASET_PATH](optional) --run_eval=[RUN_EVAL](optional)
```

```log
参数说明：
DEVICE_ID: 必选，执行训练的卡编号。如果通过python启动训练，可以执行linux命令export DEVICE_ID=X指定卡号，X为卡编号
CONFIG_FILE： 必选，配置文件路径
DATASET_PATH： 可选，数据集路径，默认data/cityscapes
PRETRAINED_CKPT_PATH： 可选，预训练模型的路径，默认pretrained/pretrained/ms_pretrained_b0.ckpt
RUN_EVAL: 可选，是否在训练时评估模型，默认True
```

训练过程日志保存在train.log文件种

```log
Epoch 12/200, step:100/1487, loss:0.08269882, overflow:False, loss_scale:32768.0, step cost:182ms
Epoch 12/200, step:200/1487, loss:0.08838322, overflow:False, loss_scale:32768.0, step cost:183ms
Epoch 12/200, step:300/1487, loss:0.06864901, overflow:False, loss_scale:32768.0, step cost:190ms
Epoch 12/200, step:400/1487, loss:0.10473227, overflow:False, loss_scale:32768.0, step cost:185ms
Epoch 12/200, step:500/1487, loss:0.07113585, overflow:False, loss_scale:32768.0, step cost:186ms
Epoch 12/200, step:600/1487, loss:0.12277897, overflow:False, loss_scale:32768.0, step cost:193ms
Epoch 12/200, step:700/1487, loss:0.07687371, overflow:False, loss_scale:32768.0, step cost:189ms
Epoch 12/200, step:800/1487, loss:0.052616842, overflow:False, loss_scale:65536.0, step cost:184ms
Epoch 12/200, step:900/1487, loss:0.057932653, overflow:False, loss_scale:32768.0, step cost:184ms
Epoch 12/200, step:1000/1487, loss:0.24258433, overflow:False, loss_scale:32768.0, step cost:183ms
Epoch 12/200, step:1100/1487, loss:0.04909046, overflow:False, loss_scale:32768.0, step cost:189ms
Epoch 12/200, step:1200/1487, loss:0.07944703, overflow:False, loss_scale:32768.0, step cost:184ms
Epoch 12/200, step:1300/1487, loss:0.08049801, overflow:False, loss_scale:32768.0, step cost:185ms
Epoch 12/200, step:1400/1487, loss:0.065799646, overflow:False, loss_scale:32768.0, step cost:174ms
eval dataset size:500
eval image 100/500 done, step cost: 593ms
eval image 200/500 done, step cost: 586ms
eval image 300/500 done, step cost: 585ms
eval image 400/500 done, step cost: 649ms
eval image 500/500 done, step cost: 580ms
====================== Evaluation Result ======================
===> class: road, IoU: 0.9600858858787784
===> class: sidewalk, IoU: 0.715323303638259
===> class: building, IoU: 0.8837955897151115
===> class: wall, IoU: 0.33827261107225753
===> class: fence, IoU: 0.41190368390165694
===> class: pole, IoU: 0.5130745390096472
===> class: traffic light, IoU: 0.53816132073092
===> class: traffic sign, IoU: 0.6545785767944123
===> class: vegetation, IoU: 0.9031885848185216
===> class: terrain, IoU: 0.5344057148054628
===> class: sky, IoU: 0.9240028381718857
===> class: person, IoU: 0.6519006647756983
===> class: rider, IoU: 0.3307419025377471
===> class: car, IoU: 0.8848571647533297
===> class: truck, IoU: 0.265923872794352
===> class: bus, IoU: 0.39119923446452287
===> class: train, IoU: 0.31510354339228125
===> class: motorcycle, IoU: 0.2662360862992992
===> class: bicycle, IoU: 0.6395381347665564
===============================================================
===> mIoU: 0.5853838553853, ckpt: segformer_mit_b1_12.ckpt
===============================================================
```

模型结果默认保存在checkpoint文件夹中

```shell
# ls checkpoint/
-r-------- 1 root root 176884411 Mar 31 15:28 segformer_mit_b1_1.ckpt
-r-------- 1 root root 176884411 Mar 31 15:40 segformer_mit_b1_2.ckpt
-r-------- 1 root root 176884411 Mar 31 15:50 segformer_mit_b1_3.ckpt
-r-------- 1 root root 176884411 Mar 31 16:00 segformer_mit_b1_4.ckpt
-r-------- 1 root root 176884411 Mar 31 16:09 segformer_mit_b1_5.ckpt
-r-------- 1 root root 176884411 Mar 31 16:19 segformer_mit_b1_6.ckpt
-r-------- 1 root root 176884411 Mar 31 16:29 segformer_mit_b1_7.ckpt
-r-------- 1 root root 176884411 Mar 31 16:39 segformer_mit_b1_8.ckpt
-r-------- 1 root root 176884411 Mar 31 16:49 segformer_mit_b1_9.ckpt
-r-------- 1 root root 176884411 Mar 31 17:13 segformer_mit_b1_best.ckpt
```

### 分布式训练

在分布式训练之前，需要使用脚本[hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) 脚本生成组网信息文件：

```shell
# 参数[0,8)表示生成0~7号卡的组网信息文件
python hccl_tools.py --device_num "[0,8)"
```

在昇腾上使用分布式训练运行下面的命令：

```shell
bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE]
bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional)
bash run_standalone_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional)
bash run_standalone_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional) [RUN_EVAL](optional)
For example: bash run_distribute_train.sh 8 /path/hccl_8p.json /segformer/config/segformer.b0.512x1024.city.yaml /segformer/data/cityscapes/"
```

```log
参数说明：
DEVICE_NUM: 必选，执行训练的卡数量。
RANK_TABLE_FILE： 必选，组网信息文件路径。
CONFIG_FILE： 必选，配置文件路径
DATASET_PATH： 可选，数据集路径，默认data/cityscapes
PRETRAINED_CKPT_PATH： 可选，预训练模型的路径，默认pretrained/pretrained/ms_pretrained_b0.ckpt
RUN_EVAL: 可选，是否在训练时评估模型，默认True
```

## 评估

### 评估过程

执行如下脚本进行评估：

```shell
bash run_eval.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH]
For example: bash run_eval.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/checkpoint/segformer_mit_b1_10.ckpt /segformer/data/cityscapes/
```

也可以直接运行python脚本：

```shell
python eval.py --config_path=[CONFIG_FILE] --eval_ckpt_path=[CKPT_PATH] --data_path=[DATASET_PATH]
```

```log
参数说明：
DEVICE_ID: 必选，执行评估的卡编号。如果通过python启动训练，可以执行linux命令export DEVICE_ID=X指定卡号，X为卡编号
CONFIG_FILE： 必选，配置文件路径
CKPT_PATH： 必选，想要评估的模型文件路径
DATASET_PATH： 必选，数据集路径
```

### 评估结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
eval dataset size:500
eval image 100/500 done, step cost: 550ms
eval image 200/500 done, step cost: 572ms
eval image 300/500 done, step cost: 569ms
eval image 400/500 done, step cost: 571ms
eval image 500/500 done, step cost: 572ms
====================== Evaluation Result ======================
===> class: road, IoU: 0.9660652483316745
===> class: sidewalk, IoU: 0.7492563854504648
===> class: building, IoU: 0.8866242816349342
===> class: wall, IoU: 0.31656804931621124
===> class: fence, IoU: 0.3411113046083727
===> class: pole, IoU: 0.5398745184914306
===> class: traffic light, IoU: 0.5906606518622559
===> class: traffic sign, IoU: 0.6890628404136455
===> class: vegetation, IoU: 0.9099730088219271
===> class: terrain, IoU: 0.5330567771230307
===> class: sky, IoU: 0.9325007178024083
===> class: person, IoU: 0.7382945383844727
===> class: rider, IoU: 0.45634221493539334
===> class: car, IoU: 0.9014416024726345
===> class: truck, IoU: 0.3700774842437022
===> class: bus, IoU: 0.3706747870677765
===> class: train, IoU: 0.34248168670673496
===> class: motorcycle, IoU: 0.4306732371022434
===> class: bicycle, IoU: 0.7107199409993931
===============================================================
===> mIoU: 0.6197610145141426, ckpt: segformer_mit_b1_22.ckpt
===============================================================
all eval process done, cost:353s
```

## 导出

### 导出过程

导出mindir或air模型，执行如下命令：

```shell
python export.py --config_path [CONFIG_PATH] --export_ckpt_path [EXPORT_CKPT_PATH] --export_format [EXPORT_FORMAT]
```

其中，参数config_path、export_ckpt_path为必填项，export_format 必须在 ["AIR", "MINDIR"]中选择，默认为AIR。

### 导出结果

导出的模型文件名称以配置文件名称开头，导出类型结尾。

```shell
-r-------- 1 root root 59788012 Apr  3 15:47 segformer.b1.1024x1024.city.air
-r-------- 1 root root 59185783 Apr  3 15:49 segformer.b1.1024x1024.city.mindir
```

## 推理

### 推理过程

> 提供推理脚本

```bash
bash run_infer.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH]
bash run_infer.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH] [OUTPUT_PATH](optional)
For example: bash run_infer.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/checkpoint/segformer_mit_b1_10.ckpt /segformer/data/cityscapes/
```

也可以直接运行python脚本

```shell
python infer.py --config_path=[CONFIG_FILE] --infer_ckpt_path=[CKPT_PATH] --data_path=[DATASET_PATH] --infer_output_path=[OUTPUT_PATH](optional)
```

```log
参数说明：
DEVICE_ID: 必选，执行推理的卡编号。如果通过python脚本启动训练，可以执行linux命令export DEVICE_ID=X指定卡号，X为卡编号
CONFIG_FILE： 必选，配置文件路径
CKPT_PATH： 必选，执行推理的模型文件路径
DATASET_PATH： 必选，数据集路径
OUTPUT_PATH： 可选，推理结果保存路径，默认infer_result
```

### 推理结果

例如：上述python命令将在后台运行，您可以通过infer.log文件查看结果。结果如下：

```log
get image size:398, infer result will save to /segformer/infer_result
infer 100/398 done, step cost:1958ms
infer 200/398 done, step cost:1898ms
infer 300/398 done, step cost:1894ms
all infer process done, cost:819s
```

推理结果默认保存在infer_result文件夹中

```shell
# ll infer_result/
-rw------- 1 root root   26318 Mar 31 17:35 munich_000005_000019_leftImg8bit_color.png    # 根据灰度图片生成的彩色图片
-rw------- 1 root root   21199 Mar 31 17:35 munich_000005_000019_leftImg8bit_gray.png     # 推导出的灰度图片
-rw------- 1 root root 1778554 Mar 31 17:35 munich_000005_000019_leftImg8bit_overlap.png  # 彩色图片与原图片的重叠图片
-rw------- 1 root root 2279236 Mar 31 17:35 munich_000005_000019_leftImg8bit.png          # 原图片
```

## 性能

### 训练性能

| Parameters                 | Ascend 910                                                   |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | Segformer                                                    |
| Resource                   | Ascend 910; CPU 2.70GHz, 96cores; Memory 1510G; OS Euler2.7  |
| uploaded Date              | 03/04/2023 (month/day/year)                                  |
| MindSpore Version          | master                                                       |
| Dataset                    | Cityscapes                                                   |
| Training Parameters        | epoch=300, steps per epoch=186, batch_size = 16              |
| Optimizer                  | AdamWeightDecay                                              |
| Loss Function              | Softmax Cross Entropy                                        |
| Backbone                   | mit_b0                                                       |
| Loss                       | 0.03060256                                                   |
| Speed                      | 100 ms/step                                                  |
| Total time                 | 95 mins                                                      |
| Parameters (M)             | 3.8                                                          |
| Checkpoint for Fine tuning | 43M (.ckpt file)                                             |

## 精度

| Encoder Model Size    | image size      | mIoU    |
| --------------------- | --------------- | ------- |
| MiT-B0                | 512 * 1024      | 70.88   |
| MiT-B0                | 1024 * 1024     | 71.54   |
| MiT-B1                | 1024 * 1024     | 75.34   |
| MiT-B2                | 1024 * 1024     | 78.53   |
| MiT-B3                | 1024 * 1024     | 78.97   |
| MiT-B4                | 1024 * 1024     | 79.65   |
| MiT-B5                | 1024 * 1024     | 79.90   |

## 贡献指南

如果你想参与贡献昇思的工作当中，请阅读[昇思贡献指南](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING_CN.md)和[how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
