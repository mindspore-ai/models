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
- [模型说明](#模型说明)
    - [性能](#性能)
    - [训练性能](#训练性能)
    - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# [PointNet描述](#目录)

PointNet++模型诞生于2017年，是一种将PointNet递归地应用于输入点集的嵌套分区的分层神经网络。通过度量空间距离，PointNet能够随着上下文尺度的增加而学习局部特征。实验表明，PointNet++能够高效、稳健地学习深度点集特征。

[论文](http://arxiv.org/abs/1706.02413)：Qi, Charles R., et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." arXiv preprint arXiv:1706.02413 (2017).

# [模型架构](#目录)

PointNet++由多级*set abstraction*组成。每级set abstraction中，通过对一组点云集的处理和抽象，生成一个带有局部特征的新点云集。set abstraction由三个关键层组成：*采样层*、*组合层*和*PointNet层*。*采样层*从输入的点云中选择一个点来作为局部区域的中心点。然后，*分组层*通过在中心点周围找到相邻点来构建局部区域集。*PointNet层*使用迷你PointNet模型对局部区域集进行局部特征提取。

# [数据集](#目录)

使用的数据集：已对齐的[ModelNet40](<https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip>)

- 数据集大小：6.48G，每个点云包含2048个从形状表面均匀采样的点。每个点云都是零均值，并以球作为点云的单位。
    - 训练集：5.18G，9843个点云
    - 测试集：1.3G，2468个点云
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
bash scripts/run_standalone_train.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_standalone_train.sh modelnet40_normal_resampled save pointnet2.ckpt

# 运行分布式训练
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_distributed_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt

# 评估
bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME]
# 示例：
bash scripts/run_eval.sh modelnet40_normal_resampled pointnet2.ckpt
```

在GPU上训练模型

```shell
# 运行单机训练
bash scripts/run_standalone_train_gpu.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_standalone_train_gpu.sh modelnet40_normal_resampled save pointnet2.ckpt

# 运行分布式训练
bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_distributed_train_gpu.sh 8 modelnet40_normal_resampled save pointnet2.ckpt

# 评估
bash scripts/run_eval_gpu.sh [DATA_PATH] [CKPT_NAME]
# 示例：
bash scripts/run_eval_gpu.sh modelnet40_normal_resampled pointnet2.ckpt
```

# [脚本说明](#目录)

# [脚本及样例代码](#目录)

```bash
├── PointNet2
    ├── eval.py                             # 评估网络
    ├── export.py
    ├── README.md
    ├── README.md
    ├── scripts
    │   ├── run_distributed_train_gpu.sh    # 使用GPU进行分布式训练（8卡）
    │   ├── run_distributed_train.sh        # 使用Ascend进行分布式训练（8卡）
    │   ├── run_eval_gpu.sh                 # 使用GPU进行评估
    │   ├── run_eval.sh                     # 使用Ascend进行评估
    │   ├── run_standalone_train_gpu.sh     # 使用GPU进行单机训练（单卡）
    │   └── run_standalone_train.sh         # 使用Ascend进行单机训练（单卡）
    ├── src
    │   ├── callbacks.py                    # 自定义回调函数
    │   ├── dataset.py                      # 数据预处理
    │   ├── layers.py                       # 网络层初始化
    │   ├── lr_scheduler.py                 # 学习率调度器
    │   ├── pointnet2.py                    # 自定义网络
    │   ├── pointnet2_utils.py              # 自定义网络工具脚本
    │   ├── provider.py                     # 预处理用于训练的数据
    │
    │   ├── provider.py                     # 训练网络
```

# [脚本参数](#目录)

```bash
train.py中主要的参数如下：
--batch_size        # 批处理大小
--epoch             # 总训练epochs数
--learning_rate     # 训练学习率
--optimizer         # 用于训练的优化器。可选值为Adam和SGD。
--data_path         # 训练和评估数据集路径
--loss_per_epoch    # 每轮训练的损失值
--save_dir          # 保存训练过程中生成的文件的路径
--use_normals       # 训练中是否使用法向量数据
--pretrained_ckpt   # 加载检查点文件的路径
--enable_modelarts         # 是否使用ModelArts
```

# [训练过程](#目录)

## 训练

- 在Ascend上运行

```shell
# 运行单机训练
bash scripts/run_standalone_train.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_standalone_train.sh modelnet40_normal_resampled save pointnet2.ckpt

# 运行分布式训练
bash scripts/run_distributed_train.sh [RANK_TABLE_FILE] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
# 示例：
bash scripts/run_distributed_train.sh hccl_8p_01234567_127.0.0.1.json modelnet40_normal_resampled save pointnet2.ckpt
```

- 在GPU上运行

    ```shell
    # 运行单机训练
    bash scripts/run_standalone_train_gpu.sh [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
    # 示例：
    bash scripts/run_standalone_train_gpu.sh modelnet40_normal_resampled save pointnet2.ckpt

    # 运行分布式训练
    bash scripts/run_distributed_train_gpu.sh [DEVICE_NUM] [DATA_PATH] [SAVE_DIR] [PRETRAINDE_CKPT(optional)]
    # 示例：
    bash scripts/run_distributed_train_gpu.sh 8 modelnet40_normal_resampled save pointnet2.ckpt
    ```

训练后得到如下损失值：

```bash
# 训练日志
epoch: 1 step: 410, loss is 1.4731973
epoch time: 704454.051 ms, per step time: 1718.181 ms
epoch: 2 step: 410, loss is 1.0621885
epoch time: 471478.224 ms, per step time: 1149.947 ms
epoch: 3 step: 410, loss is 1.176581
epoch time: 471530.000 ms, per step time: 1150.073 ms
epoch: 4 step: 410, loss is 1.0118457
epoch time: 471498.514 ms, per step time: 1149.996 ms
epoch: 5 step: 410, loss is 0.47454038
epoch time: 471535.602 ms, per step time: 1150.087 ms
...
```

模型检查点将保存在**SAVE_DIR**目录中。

# [评估过程](#目录)

## 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- 在Ascend上运行

    ```shell
    # 评估
    bash scripts/run_eval.sh [DATA_PATH] [CKPT_NAME]
    # 示例：
    bash scripts/run_eval.sh modelnet40_normal_resampled pointnet2.ckpt
    ```

- 在GPU上运行

    ```shell
    # 评估
    bash scripts/run_eval_gpu.sh [DATA_PATH] [CKPT_NAME]
    # 示例：
    bash scripts/run_eval_gpu.sh modelnet40_normal_resampled pointnet2.ckpt
    ```

您可以通过文件eval.log查看结果。测试数据集的准确性如下：

```bash
# grep "Accuracy: " eval.log
'Accuracy': 0.916
```

# [模型说明](#目录)

## [性能](#目录)

## 训练性能

| 参数                | Ascend                                           | GPU                                            |
| -------------------------- | ------------------------------------------------- | ----------------------------------------------- |
| 模型版本             | PointNet++                                       | PointNet++                                     |
| 资源                  | Ascend 910；CPU 24核；内存256G；EulerOS 2.8| RTX 3090；GPU内存24268MB；Ubuntu          |
| 上传日期             | 08/31/2021                      | 12/20/2021                    |
| MindSpore版本         | 1.3.0                                            | 1.5.0rc1                                          |
| 数据集                   | ModelNet40                                       | ModelNet40                                     |
| 训练参数       | epoch=200, steps=82000, batch_size=24, lr=0.001  | epoch=200, steps=82000, batch_size=24, lr=0.001|
| 优化器                 | Adam                                             | Adam                                           |
| 损失函数             | NLLLoss                                          | NLLLoss                                        |
| 输出                   | 概率                                      | 概率                                    |
| 损失                      | 0.01                                             | 0.01                                           |
| 速度                     | 1.5秒/步（单卡）                                  | 390毫秒/步（单卡）                               |
| 总时长                | 27.3小时（单卡）                                      | 8小时8分钟6秒（单卡）                                    |
| 微调检查点| 17MB（.ckpt)                               | 17MB（.ckpt)                             |

## 推理性能

| 参数       | Ascend                                           | GPU                                  |
| ----------------- | ------------------------------------------------- | ------------------------------------- |
| 模型版本    | PointNet++                                       | PointNet++                           |
| 资源         | Ascend 910；CPU 24核；内存256G；EulerOS 2.8| RTX 3090；GPU内存24268MB；Ubuntu|
| 上传日期    | 08/31/2021                      | 12/20/2021          |
| MindSpore版本| 1.3.0                                            | 1.3.0                                |
| 数据集          | ModelNet40                                       | ModelNet40                           |
| Batch_size       | 24                                               | 24                                   |
| 输出          | 概率                                      | 概率                          |
| 准确率         | 91.5%（单卡）                                       | 91.42%（单卡）                          |
| 总时长       | 2.5分钟                                          | 1.13分钟                             |

# [随机情况说明](#目录)

我们在dataset.py，provider.py和pointnet2_utils.py中设置了随机种子。

# [ModelZoo主页](#目录)

请查看官方[主页](https://gitee.com/mindspore/models)。
