# 目录

- [目录](#目录)
- [PointNet描述](#pointnet2描述)
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
    - [Ascend 310推理过程](#ascend310推理过程)
        - [导出MindIR](#导出mindir)
- [模型说明](#模型说明)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [PointNet描述](#目录)

PointNet模型诞生于2017年，是一种将PointNet递归地应用于输入点集的嵌套分区的分层神经网络。该篇论文的作者提出了一种将深度学习模型直接应用于点云数据的方法，即PointNet。

[论文]((https://arxiv.org/abs/1612.00593): Qi, Charles R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
 arXiv preprint arXiv:1612.00593 (2017).

# [模型架构](#目录)

对于输入的每个n×3N/3N×3点云数据，先通过T-Net模型将输入数据在空间上进行对齐，再通过MLP将它映射到64维空间中对齐，最后将数据映射到1024维空间。此时，每个点云都有一个1024维向量，而这种向量对三维点云来说是多余的。因此，最大池化的操作能够让1024维通道保持最大化，即获得1×1024/1024×1。1024维的向量是n nn点云的全局特征。

# [数据集](#目录)

使用的数据集：[ShapeNet](<https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip>)子集上的分段数据

- 数据格式：TXT
    - 注：数据将在**src/dataset.py**中处理。

# [环境要求](#目录)

- 硬件
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```shell
# 运行单机训练
bash scripts/run_standalone_train.sh [DATA_PATH] [CKPT_PATH]
# 示例：
bash scripts/run_standalone_train.sh '/home/pointnet/shapenetcore_partanno_segmentation_benchmark_v0' '../results'

# 运行分布式训练
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_standalone_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt

# 评估
bash scripts/run_standalone_eval.sh [DATA_PATH] [MODEL_PATH]
# 示例：
bash scripts/run_standalone_eval.sh '/home/pointnet/shapenetcore_partanno_segmentation_benchmark_v0' '../results/pointnet_network_epoch_10.ckpt'
```

# [脚本说明](#目录)

# [脚本及样例代码](#目录)

```bash
├── .
    ├── pointnet
        ├── ascend310_infer
        │   ├── inc
        │   │   ├── utils.h
        │   ├── src
        │   │   ├── main.cc
        │   │   ├── utils.cc
        │   ├── build.sh
        ├── scripts
        │   ├── run_distribute_ascend.sh        # 使用Ascend进行分布式训练（8卡）
        │   ├── run_distribute_gpu.sh           # 使用GPU进行分布式训练（8卡）
        │   ├── run_standalone_eval_ascend.sh          # 使用Ascend进行评估（单卡）
        │   ├── run_standalone_eval_gpu.sh      # 使用GPU进行评估（单卡）
        │   ├── run_standalone_eval_onnx_gpu.sh      # 使用GPU评估ONNX模型（单卡）
        │   ├── run_infer_310.sh                # 使用Ascend 310进行推理
        │   ├── run_standalone_train_ascend.sh         # 使用Ascend进行单机训练（单卡）
        │   └── run_standalone_train_gpu.sh     # 使用GPU进行单机训练（单卡）
        ├── src
        │   ├── misc                     # 数据集部分
        │   ├── dataset.py               # 数据预处理
        │   ├── export.py                # 导出模型
        │   ├── loss.py                  # PointNet损失
        │   ├── network.py               # 自定义网络
        │   └── preprocess.py            # 预处理用于训练的数据
        ├── eval.py                      # 评估网络
        ├── eval_onnx.py                 # 评估ONNX模型
        ├── postprocess.py               # Ascend 310推理数据后处理
        ├── preprocess.py                # Ascend 310推理数据预处理
        ├── README.md
        ├── requirements.txt
        └── train.py                     # 训练网络
```

# [脚本参数](#目录)

```bash
train.py中主要的参数如下：
--device_id        # 用于训练的设备ID
--batchSize        # 批处理大小
--nepoch           # 总epochs数
--learning_rate    # 训练学习率
--data_url         # 训练和评估数据集路径
--loss_per_epoch   # 每轮训练的损失值
--train_url        # 训练生成文件的保存路径
--model            # 加载CKPT文件的路径
--enable_modelarts # 是否使用ModelArts
```

# [训练过程](#目录)

## 训练

- 在Ascend上运行

    ```shell
    # 在Ascend上进行单机训练
    bash scripts/run_standalone_train_ascend.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
    # 示例：
    bash scripts/run_standalone_train_ascend.sh ../shapenetcore_partanno_segmentation_benchmark_v0 ./ckpts 1



    # 在Ascend上进行分布式训练
    bash scripts/run_distribution_ascend.sh [RANK_TABLE_FILE] [CKPTS_DIR] [DATA_PATH]
    # 示例：
    bash scripts/run_distribution_ascend.sh [RANK_TABLE_FILE] ./ckpts ../shapenetcore_partanno_segmentation_benchmark_v0


    ```

- 在GPU上运行

    ```shell
    # 在GPU上进行单机训练
    bash scripts/run_standalone_train_gpu.sh [DATA_PATH] [CKPT_PATH] [DEVICE_ID]
    # 示例：
    bash scripts/run_standalone_train_gpu.sh ../shapenetcore_partanno_segmentation_benchmark_v0 ./ckpts 1
    # 在GPU上进行分布式训练
    bash scripts/run_distribute_gpu.sh [DATA_PATH] [CKPT_PATH]
    # 示例：
    bash scripts/run_distribute_gpu.sh ./ckpts ../shapenetcore_partanno_segmentation_benchmark_v0
    ```

分布式训练需要提前创建JSON格式的HCCL配置文件。具体操作请参考：[hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

训练后得到如下损失值：

```bash
# 训练日志
Epoch : 1/50  episode : 1/40   Loss : 1.3433  Accuracy : 0.489538 step_time: 1.4269
Epoch : 1/50  episode : 2/40   Loss : 1.2932  Accuracy : 0.541544 step_time: 1.4238
Epoch : 1/50  episode : 3/40   Loss : 1.2558  Accuracy : 0.567900 step_time: 1.4397
Epoch : 1/50  episode : 4/40   Loss : 1.1843  Accuracy : 0.654681 step_time: 1.4235
Epoch : 1/50  episode : 5/40   Loss : 1.1262  Accuracy : 0.726756 step_time: 1.4206
Epoch : 1/50  episode : 6/40   Loss : 1.1000  Accuracy : 0.736225 step_time: 1.4363
Epoch : 1/50  episode : 7/40   Loss : 1.0487  Accuracy : 0.814338 step_time: 1.4457
Epoch : 1/50  episode : 8/40   Loss : 1.0271  Accuracy : 0.782350 step_time: 1.4183
Epoch : 1/50  episode : 9/40   Loss : 0.9777  Accuracy : 0.831025 step_time: 1.4289

...
```

模型检查点将保存在**SAVE_DIR**目录中。

# [评估过程](#目录)

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

## 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- 在Ascend上运行

    ```shell
    # 在Ascend上进行评估
    bash scripts/run_standalone_eval_ascend.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]
    # 示例：
    bash scripts/run_standalone_eval_ascend.sh shapenetcore_partanno_segmentation_benchmark_v0 pointnet.ckpt 1
    ```

    您可以通过文件**log_standalone_eval_ascend**查看评估结果。测试数据集的准确性如下：

    ```bash
    # grep "mIOU " log_standalone_eval_ascend
    'mIOU for class Chair: 0.869'
    ```

- 在GPU上运行

    ```shell
    # 在GPU上评估
    bash scripts/run_standalone_eval_gpu.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]
    # 示例：
    bash scripts/run_standalone_eval_gpu.sh shapenetcore_partanno_segmentation_benchmark_v0 pointnet.ckpt 1
    ```

    您可以通过文件**log_standalone_eval_gpu**查看评估结果。测试数据集的准确性如下：

    ```bash
    # grep "mIOU " log_standalone_eval_gpu
    'mIOU for class Chair: 0.869'
    ```

## ONNX模型评估

- 在GPU上运行

  ```shell
  # 在GPU上评估
  bash scripts/run_standalone_eval_onnx_gpu.sh [DATA_PATH] [MODEL_PATH] [DEVICE_ID]
  # 示例：
  bash scripts/run_standalone_eval_onnx_gpu.sh dataset/shapenetcore_partanno_segmentation_benchmark_v0 mindir/pointnet.onnx 1
  ```

  您可以通过文件**log_standalone_eval_onnx_gpu**查看评估结果。测试数据集的准确性如下：

  ```bash
  # grep "mIOU " log_standalone_eval_onnx_gpu
  'mIOU for class Chair: 0.869'
  ```

# [Ascend 310推理过程](#ascend310推理过程)

## [导出MindIR](#导出mindir)

```bash
python src/export.py --model [CKPT_PATH] --file_format [FILE_FORMAT]
```

FILE_FORMAT可选值为AIR或MINDIR。

MindIR模型将导出到**./mindir/pointnet.mindir**。

## [在Ascend 310上运行推理](#评估)

在310上运行推理之前，应首先导出MindIR模型。然后运行下面的代码执行推理：

```bash
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
# 示例：
bash run_infer_310.sh ./mindir/pointnet.mindir ../shapenetcore_partanno_segmentation_benchmark_v0 [LABEL_PATH] N 2
```

在这里，DVPP应该是'N'！

## [结果](#评估)

```bash
'mIOU : 0.869 '
```

# [模型说明](#目录)

## [性能](#目录)

## 训练性能

| 参数                | Ascend                                           | GPU(V100(PCIE))                            |
| -------------------------- | ------------------------------------------------- | ------------------------------------------- |
| 模型版本             | PointNet                                         | PointNet                                   |
| 资源                  | Ascend 910；CPU 24核；内存256G；EulerOS 2.8| NVIDIA RTX Titan-24G                       |
| 上传日期             | 11/30/2021                      | 4/19/2022                 |
| MindSpore版本         | 1.3.0                                            | 1.3.0 1.5.0 1.6.0                          |
| 数据集                   | ShapeNet的子集                             | ShapeNet的子集                       |
| 训练参数       | epoch=50, steps=83, batch_size=64, lr=0.005      | epoch=50, steps=83, batch_size=64, lr=0.005|
| 优化器                 | Adam                                             | Adam                                       |
| 损失函数             | NLLLoss                                          | NLLLoss                                    |
| 输出                   | 概率                                      | 概率                                |
| 损失                      | 0.01                                             | 0.01                                       |
| 速度                     | 1.5秒/步（单卡）                                  | 0.19秒/步（单卡）                           |
| 总时长                | 0.3小时（单卡）                                       | 10分钟（单卡）                                  |
| 微调检查点| 17MB（.ckpt)                               | 17MB（.ckpt)                         |

## 推理性能

| 参数       | Ascend                                           | GPU(V100(PCIE))           |
| ----------------- | ------------------------------------------------- | -------------------------- |
| 模型版本    | PointNet                                         | PointNet                  |
| 资源         | Ascend 910；CPU 24核；内存256G；EulerOS 2.8| NVIDIA RTX Titan-24G      |
| 上传日期    | 11/30/2021                      | 4/19/2022|
| MindSpore版本| 1.3.0                                            | 1.3.0 1.5.0 1.6.0         |
| 数据集          | ShapeNet的子集                             | ShapeNet的子集      |
| Batch_size       | 64                                               | 64                        |
| 输出          | 概率                                      | 概率               |
| mIOU             | 86.3%（单卡)                                       | 86.3%（单卡)                |
| 总时长       | 1分钟                                            | 1分钟                     |

# [随机情况说明](#目录)

我们在train.py中设置了随机种子

# [ModelZoo主页](#目录)

请查看官方[主页](https://gitee.com/mindspore/models)。
