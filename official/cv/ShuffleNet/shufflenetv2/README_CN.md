# 目录

- [ShuffleNetV2描述](#shufflenetv2描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend 310上进行推理](#在ascend-310上进行推理)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [ModelZoo主页](#modelzoo主页)

# [ShuffleNetV2描述](#目录)

ShuffleNetV2在Ascend和GPU等平台上运行比以往更快、更准确。  
[论文](https://arxiv.org/pdf/1807.11164.pdf):Ma, N., Zhang, X., Zheng, H. T., & Sun, J. (2018). Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Proceedings of the European conference on computer vision (ECCV) (pp. 116-131).

# [模型架构](#目录)

ShuffleNetV2的整体网络架构如下：

[链接](https://arxiv.org/pdf/1807.11164.pdf)

# [数据集](#目录)

使用的数据集：[ImageNet](http://www.image-net.org/)

- 数据集大小：~125GB，120万张彩色图像，共1000类
    - 训练集：120GB，120万张图像
    - 测试集：5GB，50000张
- 数据格式：RGB
    - 注：数据将在**src/dataset.py**中处理

转换数据集：[flower_photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

- 数据集大小：221MB，3670张彩色图像，共5类
    - 训练集：177MB，2934张图像
    - 测试集：44MB，736张图像
- 数据格式：RGB
    - 注：数据将在**src/dataset.py**中处理

# [环境要求](#目录)

- 硬件
    - 使用Ascend，GPU或CPU搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

```text
+-- ShuffleNetV2
  +-- Readme.md     # ShuffleNetV2的相关说明
  +-- scripts
    +--run_distribute_train_for_ascebd.sh   # 在Ascend上运行shell脚本进行分布式训练
    +--run_distribute_train_for_gpu.sh      # 在GPU上运行shell脚本进行分布式训练
    +--run_eval_for_ascend.sh               # 在Ascend上评估shell脚本
    +--run_eval_for_gpu.sh                  # 在GPU上评估shell脚本
    +--run_standalone_train_for_gpu.sh      # 在GPU上评估shell脚本
  +-- src
    +--config.py                            # 参数配置
    +--CrossEntropySmooth.py                # GPU训练的损失函数
    +--dataset.py                           # 创建数据集
    +--loss.py                              # 网络的损失函数
    +--lr_generator.py                      # 设置学习率
    +--shufflenetv2.py                      # ShuffleNetV2网络
  +-- cpu_transfer.py                       # 转换脚本
  +-- dataset_split.py                      # 拆分数据集用于转换脚本
  +-- quick_start.py                        # 快速启动脚本
  +-- train.py                              # 训练脚本
  +-- eval.py                               # 评估脚本
```

## [训练过程](#目录)

### 使用方法

您可以使用python命令或运行shell脚本来进行训练。shell脚本的用法如下：

- 在Ascend上进行分布式训练：sh run_distribute_train_for_ascend.sh [RANK_TABLE_FILE] [DATASET_PATH]
- 在GPU上进行分布式训练：sh run_standalone_train_for_gpu.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
- 在GPU上进行单机训练：sh run_standalone_train_for_gpu.sh [DATASET_PATH]

### 启动

```bash
# 训练示例
  python：
      GPU: mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout python train.py --is_distributed=True --platform='GPU' --dataset_path='~/imagenet' > train.log 2>&1 &
      CPU: python cpu_transfer.py --checkpoint_input_path ./input_ckpt/shufflenetv2_top1acc69.63_top5acc88.72.ckpt --checkpoint_save_path ./save_ckpt/Graph_mode --train_dataset ./data/flower_photos_split/train --use_pynative_mode False --platform CPU
  shell:
      GPU: cd scripts & sh run_distribute_train_for_gpu.sh 8 0,1,2,3,4,5,6,7 ~/imagenet
```

### 结果

训练结果将存储在示例路径中。默认情况下，检查点文件将存储在`./checkpoint`中，训练日志将重定向到`./train/train.log`。

## [评估过程](#目录)

### 使用方法

您可以使用python命令或运行shell脚本进行评估。shell脚本的用法如下：

- Ascend: bash run_eval_for_ascend.sh [DATASET_PATH] [CHECKPOINT]
- GPU: bash run_eval_for_gpu.sh [DATASET_PATH] [CHECKPOINT]

### 启动

```bash
# 推理示例
  python：
      Ascend: python eval.py --platform='Ascend' --dataset_path='~/imagenet' --checkpoint='checkpoint_file' > eval.log 2>&1 &
      GPU: CUDA_VISIBLE_DEVICES=0 python eval.py --platform='GPU' --dataset_path='~/imagenet/val/' --checkpoint='checkpoint_file'> eval.log 2>&1 &
      CPU: python eval.py --dataset_path ./data/flower_photos_split/eval --checkpoint_dir ./save_ckpt/Graph_mode --platform CPU --checkpoint ./save_ckpt/Graph_mode/shufflenetv2_1-154_18.ckpt --enable_checkpoint_dir True --use_pynative_mode False
  shell：
      Ascend: cd scripts & sh run_eval_for_ascend.sh '~/imagenet' 'checkpoint_file'
      GPU: cd scripts & sh run_eval_for_gpu.sh '~/imagenet' 'checkpoint_file'
```

### 结果

推理结果将存储在示例路径中，您可以在`eval.log`中找到结果。

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#目录)

本地导出MindIR模型

```shell
python export.py --device_target [PLATFORM] --ckpt_file [CKPT_FILE] --file_format [FILE_FORMAT] --file_name [OUTPUT_FILE_BASE_NAME]
```

必须先设置checkpoint_file_path参数。
`PLATFORM`：可选值为Ascend，GPU或CPU。
`FILE_FORMAT`：可选值为AIR，ONNX或MINDIR。

### 在Ascend 310上进行推理

在进行推理之前，必须通过`export.py`脚本导出MindIR文件。下方为使用MindIR模型进行推理的例子。
当前批处理大小只能设置为1。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `MINDIR_PATH`：MindIR模型的文件名。
- `DATASET_NAME`：ImageNet2012数据集。
- `DATASET_PATH`：ImageNet2012数据集中val的路径。
- `NEED_PREPROCESS`：值为y或n。
- `DEVICE_ID`：可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中找到类似如下结果。
Top1 acc：0.69608
Top5 acc：0.88726

### 转换后在CPU上进行推理

```python
# 在CPU上进行推理
python eval.py --dataset_path [eval dataset] --checkpoint_dir [ckpt dir for eavl ] --platform [CPU] --checkpoint [ckpt path for eval] --enable_checkpoint_dir [True/False]--use_pynative_mode [True/False]
```

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中找到类似如下结果。
Top1 acc：0.86
Top5 acc：1

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | Ascend 910                   | GPU                          |CPU（转换）                          |
| -------------------------- | ----------------------------- |-------------------------------|-------------------------------|
| 模型版本             | ShuffleNetV2                 | ShuffleNetV2                 | ShuffleNetV2                 |
| 资源                  | Ascend 910                   | NV SMX2 V100-32G             |Intel(R)Core(TM) i5-7200U CPU@2.50GHz(4 CPUs)|
| 上传日期             | 10/09/2021  | 09/24/2020  | 08/30/2022  |
| MindSpore版本         | 1.3.0                        | 1.0.0                        | 1.8                          |
| 数据集                   | ImageNet                     | ImageNet                     |Flower_photos                     |
| 训练参数       | src/config.py                | src/config.py                | src/config.py                |
| 优化器                 | 动量                     | 动量                     | 动量                     |
| 损失函数             | SoftmaxCrossEntropyWithLogits| CrossEntropySmooth           | CrossEntropySmooth           |
| 准确率                  | 69.59%（Top1）                 | 69.4%（Top1）                  | 86.4%（Top1）                  |
| 总时长                | 11.6小时8秒                   | 49小时8秒                     |15小时18分钟6秒                     |

### 推理性能

| 参数                | Ascend 910                   | GPU                          | CPU（转换）                          |
| -------------------------- | ----------------------------- |-------------------------------|-------------------------------|
| 资源                  | Ascend 910                   | NV SMX2 V100-32G             |Intel(R)Core(TM) i5-7200U CPU@2.50GHz(4 CPUs)             |
| 上传日期             | 10/09/2021  | 09/24/2020  | 08/30/2022  |
| MindSpore版本         | 1.3.0                        | 1.0.0                        | 1.8.0                        |
| 数据集                   | ImageNet                     | ImageNet                     |Flower_photos                     |
| batch_size                | 125                          | 128                          |128                    |
| 输出                   | 概率                  | 概率                  | 概率                  |
| 准确率                  | ac=69.59%（Top1）             | acc=69.4%（Top1）              | acc=86.4%（Top1）              |

# [ModelZoo主页](#目录)

请查看官方[主页](https://gitee.com/mindspore/models)。
