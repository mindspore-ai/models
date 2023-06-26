# 目录

- [目录](#目录)
    - [Unet描述](#unet描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
            - [在GPU上训练](#在gpu上训练)
            - [在Ascend上训练](#在ascend上训练)
            - [在CPU上训练](#在cpu上训练)
        - [分布式训练](#分布式训练)
            - [在GPU上进行分布式训练（8卡）](#在gpu上进行分布式训练8卡)
            - [在Ascend上进行分布式训练](#在ascend上进行分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
            - [在GPU上评估](#在gpu上评估)
            - [在Ascend上评估](#在ascend上评估)
            - [在CPU上评估](#在cpu上评估)
        - [ONNX评估](#onnx评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在ascend-310上进行推理](#在ascend-310上进行推理)
        - [结果](#结果)
    - [模型说明](#模型说明)
        - [性能](#性能)
            - [评估性能](#评估性能)
            - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

## [Unet描述](#目录)

Unet3D模型被广泛用于医学的三维图像分割。Unet3D的网络架构与Unet相似，主要区别在于Unet3D使用Conv3D等方法，而Unet完全是2D架构。要了解更多关于Unet3D网络的信息，请参阅论文《Unet3D: Learning Dense Volumetric Segmentation from Sparse Annotation》。

## [模型架构](#目录)

Unet3D模型是在Unet(2D)基础上创建的，包括编码器和解码器。编码器用于分析整个图像，提取和分析特征，解码器用于生成分割块图。在这个模型中，我们还在基块中增加了残差块来改善网络。

## [数据集](#目录)

使用的数据集：[LUNA16](https://luna16.grand-challenge.org/)

- 说明：该数据用于从立体CT图像中自动检测结节的位置。数据来自LIDC-IDRI数据库的888次CT扫描。完整的数据集分为10个子集，应用于10倍交叉验证。所有子集都在zip压缩文件中。

- 数据及大小：887
    - 训练集：877张图像
    - 测试集：10张图像（按字典排序的第9个子集中的最后10张图像）
- 数据格式：ZIP
    - 注：数据将在**convert_nifti.py**中处理，在数据处理过程中将忽略其中一个数据。
- 数据目录结构

    ```text

    .
    └─LUNA16
    ├── train
    │   ├── image         // 包含877个图像文件
    |   ├── seg           // 包含877个分割文件
    ├── val
    │   ├── image         // 包含10个图像文件
    |   ├── seg           // 包含10个分割文件
    ```

## [环境要求](#目录)

- 硬件
    - 使用Ascend或GPU搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 选择要使用的网络和数据集

    ```shell

    将数据集转换为MIFTI格式
    python ./src/convert_nifti.py --data_path=/path/to/input_image/ --output_path=/path/to/output_image/

    ```

    请参考`default_config.yaml`，支持快速入门的参数配置。

- 在Ascend上运行

    ```python

    # 运行训练示例
    python train.py --data_path=/path/to/data/ > train.log 2>&1 &

    # 运行分布式训练示例
    bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH]

    # 运行评估示例
    python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ > eval.log 2>&1 &
    ```

- 在GPU上运行

    ```shell
    # 输入脚本目录
    cd scripts
    # 运行训练示例（FP32）
    bash ./run_standalone_train_gpu_fp32.sh [TRAINING_DATA_PATH]
    # 运行训练示例（FP16）
    bash ./run_standalone_train_gpu_fp16.sh [TRAINING_DATA_PATH]
    # 运行分布式训练示例（FP32）
    bash ./run_distribute_train_gpu_fp32.sh [TRAINING_DATA_PATH]
    # 运行分布式训练示例（FP16）
    bash ./run_distribute_train_gpu_fp16.sh [TRAINING_DATA_PATH]
    # 运行评估示例（FP32）
    bash ./run_standalone_eval_gpu_fp32.sh [VALIDATING_DATA_PATH] [CHECKPOINT_FILE_PATH]
    # 运行评估示例（FP16）
    bash ./run_standalone_eval_gpu_fp16.sh [VALIDATING_DATA_PATH] [CHECKPOINT_FILE_PATH]

    ```

- 在GPU上运行

    ```python
    # 运行训练示例
    python train.py --device_target=CPU --data_path=/path/to/data/ > train.log 2>&1 &

    # 运行评估示例
    python eval.py --device_target=CPU --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ > eval.log 2>&1 &
    ```

如果您想在Modelarts中运行，请查看[ModelArts](https://support.huaweicloud.com/modelarts/)的官方文档，您可以按照以下方式开始训练和评估：

```python
# 在ModelArts上运行分布式训练
# （1）执行a或b；
#       a.在yaml文件上设置"enable_modelarts=True"；
#          在yaml文件中设置您需要的其他参数。
#       b.在网页上添加"enable_modelarts=True"；
#          在网页上添加其他参数；
# （2）下载NiBabel，将pip-requirements.txt设置为代码目录；
# （3）在网页上设置代码目录为"/path/unet3d"；
# （4）在网页上设置启动文件为"train.py"；
# （5）在网页上设置"Dataset path"、"Output file path"和"Job log path"；
# （6）创建任务。

# 在ModelArts上运行评估
# （1）复制或上传训练好的模型到S3桶；
# （2）执行a或b；
#       a.在yaml文件上设置"enable_modelarts=True"；
#          在yaml文件上设置"checkpoint_file_path='/cache/checkpoint_path/model.ckpt'"；
#          在yaml文件上设置"checkpoint_url=/The path of checkpoint in S3/"；
#       b.在网页上添加"enable_modelarts=True"；
#          在网页上添加"checkpoint_file_path='/cache/checkpoint_path/model.ckpt'"；
#          在网页上添加"checkpoint_url=/The path of checkpoint in S3/"；
# （3）下载NiBabel，将pip-requirements.txt设置为代码目录；
# （4）在网页上设置代码目录为"/path/unet3d"；
# （5）在网页上设置启动文件为"eval.py"。
# （6）在网页上设置"Dataset path"、"Output file path"和"Output file path"；
# （7）创建任务。
```

## [脚本说明](#目录)

### [脚本及样例代码](#目录)

```text

.
└─Unet3D
  ├── README.md                                 // 关于Unet3D的说明
  ├── scripts
  │   ├──run_distribute_train.sh                // 在Ascend上进行分布式训练的shell脚本
  │   ├──run_standalone_train.sh                // 在Ascend上进行单机训练的shell脚本
  │   ├──run_standalone_eval.sh                 // 在Ascend上进行评估的shell脚本
  │   ├──run_distribute_train_gpu_fp32.sh       // 在GPU上进行分布式训练的shell脚本（FP32）
  │   ├──run_distribute_train_gpu_fp16.sh       // 在GPU上进行分布式训练的shell脚本（FP16）
  │   ├──run_standalone_train_gpu_fp32.sh       // 在GPU上进行单机训练的shell脚本（FP32）
  │   ├──run_standalone_train_gpu_fp16.sh       // 在GPU上进行单机训练的shell脚本（FP16）
  │   ├──run_standalone_eval_gpu_fp32.sh        // 在GPU上进行评估的shell脚本（FP32）
  │   ├──run_standalone_eval_gpu_fp16.sh        // 在GPU上进行评估的shell脚本（FP16）
  │   ├──run_eval_onnx.sh                // 用于onnx评估的shell脚本
  ├── src
  │   ├──dataset.py                             // 创建数据集
  │   ├──lr_schedule.py                         // 学习率调度器
  │   ├──transform.py                           // 处理数据集
  │   ├──convert_nifti.py                       // 转换数据集
  │   ├──loss.py                                // 损失函数
  │   ├──utils.py                               // 通用组件（回调函数）
  │   ├──unet3d_model.py                        // Unet3D模型
  │   ├──unet3d_parts.py                        // Unet3D部分
          ├── model_utils
          │   ├──config.py                      // 参数配置
          │   ├──device_adapter.py              // 设备适配器
          │   ├── local_adapter.py              // 本地适配器
          │   ├──moxing_adapter.py              // 装饰器
  ├── default_config.yaml                       // 参数设置
  ├── train.py                                  // 训练脚本
  ├── eval.py                                   // 评估脚本
  ├── eval_onnx.py                              // ONNX评估脚本
  ├── quick_start.py                            // 快速启动脚本

```

### [脚本参数](#目录)

训练和评估的参数都可以在config.py中设置

- Unet3D和LUNA16数据集的配置

    ```python

    'model': 'Unet3D',                  # 模型名称
    'lr': 0.0005,                       # 学习率
    'epochs': 10,                       # 单卡训练总轮次
    'batchsize': 1,                     # 训练批处理大小
    "warmup_step": 120,                 # 学习率生成器的warmp up步骤
    "warmup_ratio": 0.3,                # warmp up比率
    'num_classes': 4,                   # 数据集中的分类数
    'in_channels': 1,                   # 通道数
    'keep_checkpoint_max': 5,           # 仅保留最后一个keep_checkpoint_max检查点
    'loss_scale': 256.0,                # 损失缩放
    'roi_size': [224, 224, 96],         # 随机ROI大小
    'overlap': 0.25,                    # 重叠率
    'min_val': -500,                    # 最小间隔初始范围
    'max_val': 1000,                    # 最大间隔初始范围
    'upper_limit': 5                    # num_classes的上限
    'lower_limit': 3                    # num_classes的下限

    ```

## [训练过程](#目录)

### 训练

#### 在GPU上训练

```shell
# 输入脚本目录
cd scripts
# FP32
bash ./run_standalone_train_gpu_fp32.sh /path_prefix/LUNA16/train
# FP16
bash ./run_standalone_train_gpu_fp16.sh /path_prefix/LUNA16/train

```

上述python命令将在后台运行，您可以通过`train.log`文件查看结果。

默认情况下，您可以在训练后从**train_fp[32|16]/output/ckpt_0/ folder**文件夹中获得检查点文件。

#### 在Ascend上训练

```shell
python train.py --data_path=/path/to/data/ > train.log 2>&1 &

```

上述python命令将在后台运行，您可以通过`train.log`文件查看结果。

训练结束后，您可在默认输出文件夹下找到CKPT文件。得到如下损失值：

```shell

epoch: 1 step: 878, loss is 0.55011123
epoch time: 1443410.353 ms, per step time: 1688.199 ms
epoch: 2 step: 878, loss is 0.58278626
epoch time: 1172136.839 ms, per step time: 1370.920 ms
epoch: 3 step: 878, loss is 0.43625978
epoch time: 1135890.834 ms, per step time: 1328.537 ms
epoch: 4 step: 878, loss is 0.06556784
epoch time: 1180467.795 ms, per step time: 1380.664 ms

```

#### 在CPU上训练

```shell
python train.py --device_target=CPU --data_path=/path/to/data/ > train.log 2>&1 &

```

上述python命令将在后台运行，您可以通过`train.log`文件查看结果。

训练结束后，您可在默认输出文件夹下找到CKPT文件。

### 分布式训练

#### 在GPU上进行分布式训练（8卡）

```shell
# 输入脚本目录
cd scripts
# fpFP32
bash ./run_distribute_train_gpu_fp32.sh /path_prefix/LUNA16/train
# FP16
bash ./run_distribute_train_gpu_fp16.sh /path_prefix/LUNA16/train

```

上述shell脚本将在后台运行分布式训练。您可以通过文件`/train_parallel_fp[32|16]/train.log`查看结果。

默认情况下，您可以在训练后从`train_parallel_fp[32|16]/output/ckpt_[X]/`文件夹下获得检查点文件。

#### 在Ascend上进行分布式训练

> 注：
> RANK_TABLE_FILE参考[链接](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)，device_ip参考[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。对于像InceptionV4这样的大模型，最好导出外部环境变量`export HCCL_CONNECT_TIMEOUT=600`，将HCCL连接检查时间从默认的120秒延长到600秒。否则，连接可能会超时，因为编译时间会随着模型大小的增长而增加。
>

```shell

bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [IMAGE_PATH] [SEG_PATH]

```

上述shell脚本将在后台运行分布式训练。您可以通过`/train_parallel[X]/log.txt`文件查看结果。得到如下损失值：

```shell

epoch: 1 step: 110, loss is 0.8294426
epoch time: 468891.643 ms, per step time: 4382.165 ms
epoch: 2 step: 110, loss is 0.58278626
epoch time: 165469.201 ms, per step time: 1546.441 ms
epoch: 3 step: 110, loss is 0.43625978
epoch time: 158915.771 ms, per step time: 1485.194 ms
...
epoch: 9 step: 110, loss is 0.016280059
epoch time: 172815.179 ms, per step time: 1615.095 ms
epoch: 10 step: 110, loss is 0.020185348
epoch time: 140476.520 ms, per step time: 1312.865 ms

```

## [评估过程](#目录)

### 评估

#### 在GPU上评估

```shell
# 输入脚本目录
cd ./script
# FP32，1GPU
bash ./run_standalone_eval_gpu_fp32.sh /path_prefix/LUNA16/val /path_prefix/train_fp32/output/ckpt_0/Unet3D-10_877.ckpt
# FP16，1GPU
bash ./run_standalone_eval_gpu_fp16.sh /path_prefix/LUNA16/val /path_prefix/train_fp16/output/ckpt_0/Unet3D-10_877.ckpt
# FP32，8GPU
bash ./run_standalone_eval_gpu_fp32.sh /path_prefix/LUNA16/val /path_prefix/train_parallel_fp32/output/ckpt_0/Unet3D-10_110.ckpt
# FP16，8GPU
bash ./run_standalone_eval_gpu_fp16.sh /path_prefix/LUNA16/val /path_prefix/train_parallel_fp16/output/ckpt_0/Unet3D-10_110.ckpt

```

#### 在Ascend上评估

- 在Ascend上运行时对数据集进行评估

    在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对完整路径，例如，username/unet3d/Unet3d-10_110.ckpt。

    ```shell
    python eval.py --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ > eval.log 2>&1 &

    ```

上述python命令将在后台运行。您可以通过文件eval.log查看结果。测试数据集的准确率如下：

```shell

# grep "eval average dice is:" eval.log
eval average dice is 0.9502010010453671

```

#### 在CPU上评估

- 在CPU上运行时对数据集进行评估

    在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对完整路径，例如，username/unet3d/Unet3d-10_110.ckpt。

    ```shell
    python eval.py --device_target=CPU --data_path=/path/to/data/ --checkpoint_file_path=/path/to/checkpoint/ > eval.log 2>&1 &

    ```

上述python命令将在后台运行。您可以通过文件**eval.log**查看结果。

### ONNX评估

- 导出ONNX模型

  ```shell
  python export.py --ckpt_file /path/to/checkpoint.ckpt --file_name /path/to/exported.onnx --file_format ONNX --device_target GPU
  ```

- 运行ONNX评估

  ```shell
  python eval_onnx.py --file_name /path/to/exported.onnx --data_path /path/to/data/ --device_target GPU > output.eval_onnx.log 2>&1 &
  ```

- 以上python命令将在后台运行，您可以通过文件output.eval_onnx.log查看结果。您将获得以下准确率：

  ```log
  average dice: 0.9646
  ```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### [导出MindIR](#目录)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

必须设置ckpt_file参数。
`file_format`的值为AIR或MINDIR

### 在Ascend 310上进行推理

在进行推理之前，必须通过`export.py`脚本导出MindIR文件。下方为使用MindIR模型进行推理的例子。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS`表示是否需要预处理，其值为y或n。
- `DEVICE_ID`是可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中找到类似如下结果。

```shell

# grep "eval average dice is:" acc.log
eval average dice is 0.9502010010453671

```

## [模型说明](#目录)

### [性能](#目录)

#### 训练性能

| 参数         | Ascend                                                   |     GPU                                             |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| 模型版本      | Unet3D                                                   | Unet3D                                              |
| 资源           |  Ascend 910；CPU 2.60GHz，192核；内存755G；EulerOS 2.8| Nvidia V100 SXM2；CPU 1.526GHz，72核；内存42G；UbuntuOS 16|
| 上传日期      | 03/18/2021                              | 05/21/2021                          |
| MindSpore版本  | 1.2.0                                                    | 1.2.0                                               |
| 数据集            | LUNA16                                                   | LUNA16                                              |
| 训练参数| epoch = 10,  batch_size = 1                              | epoch = 10,  batch_size = 1                         |
| 优化器          | Adam                                                     | Adam                                                |
| 损失函数      | SoftmaxCrossEntropyWithLogits                            | SoftmaxCrossEntropyWithLogits                       |
| 速度              | 8卡：1795毫秒/步                                        | 8卡：1883毫秒/步                                   |
| 总时长         | 8卡：0.62小时                                          | 8卡：0.66小时                                     |
| 参数量（M）     | 34                                                       | 34                                                  |
| 脚本            | [Unet3D](https://gitee.com/mindspore/models/tree/master/research/cv/Unet3d)|

#### 推理性能

| 参数         | Ascend                     | GPU                        | Ascend 310                  |
| ------------------- | --------------------------- | --------------------------- | --------------------------- |
| 模型版本      | Unet3D                     | Unet3D                     | Unet3D                     |
| 资源           | Ascend 910，EulerOS 2.8    | Nvidia V100 SXM2；UbuntuOS 16| Ascend 310；EulerOS 2.8   |
| 上传日期      | 03/18/2021| 05/21/2021| 12/15/2021|
| MindSpore版本  | 1.2.0                      | 1.2.0                      | 1.5.0                      |
| 数据集            | LUNA16                     | LUNA16                     | LUNA16                     |
| batch_size         | 1                          | 1                          | 1                          |
| Dice               | dice = 0.93                | dice = 0.93                | dice = 0.93                |
| 推理模型| 56M（.ckpt）            | 56M（.ckpt）            | 56M（.ckpt）            |

# [随机情况说明](#目录)

我们在train.py中将种子设置为1。

## [ModelZoo主页](#目录)

请查看官方[主页](https://gitee.com/mindspore/models)。
