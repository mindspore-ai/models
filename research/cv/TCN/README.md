# 目录

<!-- TOC -->

- [目录](#目录)
    - [TCN描述](#tcn描述)
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
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend310执行推理](#在ascend310执行推理)
    - [模型描述](#模型描述)
        - [性能](#性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## TCN描述

TCN是一种特殊的卷积神经网络——时序卷积网络（Temporal convolutional network， TCN），于2018年被提出。相较于经典的时序模型RNN结构，TCN模型拥有较高的并行性、更加灵活的感受野，稳定的梯度和更小的内存消耗等优点，在多个时序问题上表现优异。

[论文](https://arxiv.org/pdf/1803.01271.pdf)An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling

## 模型架构

## 数据集

数据集：[Permuted MNIST](<http://yann.lecun.com/exdb/mnist/>)

- 数据集大小：52.4M，共10个类，6万张 28*28图像
    - 训练集：6万张图像
    - 测试集：5万张图像
- 数据格式：二进制文件
    - 注：数据在dataset.py中处理。

- 目录结构如下：

```bash
└─data
    └─MNIST
        ├─test
        │   ├─t10k-images.idx3-ubyte
        │   └─t10k-labels.idx1-ubyte
        └─train
            ├─train-images.idx3-ubyte
            └─train-labels.idx1-ubyte
```

数据集：Adding Problem

- 数据集描述：

  在该任务中，每个输入由深度为2的长度T序列组成，所有值在维度1的[0,1]中随机选择。第二个维度由除两个元素外的所有零组成，这两个元素用1标记。目标是将第二维度标记为1的两个随机值相加。我们可以把它看作是计算二维的点积。
  简单预测总和为1时，MSE应为0.1767左右。

  注：因为TCN的感受野取决于网络的深度和滤波器的大小，我们需要确保我们使用的模型能够覆盖序列长度T。

- 数据处理：

  可以使用create_datasetAP.py文件用于生成训练集和测试集，并以bin文件格式保存在`../data/AddProb`目录下。

  文件datasetAP.py用于读取已经生成的测试集和训练集。

## 环境要求

- 硬件（Ascend/GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```python
# 进入脚本目录，训练TCN
bash
run_train_ascend.sh [permuted_mnist | adding_problem] [DATA_PATH] [TEST_PATH] [CKPT_PATH]
# example: bash run_train_ascend.sh permuted_mnist ../data/MNIST/train ../data/MNIST/test ../checkpoint_path

# 进入脚本目录，评估TCN
bash
run_eval_ascend.sh [permuted_mnist | adding_problem] [DATA_PATH] [CKPT_FILE]
# example: bash run_eval_ascend.sh permuted_mnist ../data/MNIST/test ../checkpoint_tcn-30_937.ckpt
```

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    ```bash
    # 在 ModelArts 上使用单卡训练permuted_mnist数据集
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "train_data_path='/cache/data/MNIST/train'"
    #          在 default_config.yaml 文件中设置 "test_data_path='/cache/data/MNIST/test'"
    #          在 default_config.yaml 文件中设置 "ckpt_path='/cache/train'"
    #          (可选)在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "train_data_path='/cache/data/MNIST/train'"
    #          在网页上设置 "test_data_path='/cache/data/MNIST/test'"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "ckpt_path='/cache/train'"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 上传原始 MNIST 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/tcn"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证permuted_mnist数据集
    # (1) 执行a或者b
    #       a. 在 default_config.yaml 文件中设置 "enable_modelarts=True"
    #          在 default_config.yaml 文件中设置 "data_path='/cache/data'"
    #          在 default_config.yaml 文件中设置 "test_data_path='/cache/data/MNIST/test'"
    #          在 default_config.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 default_config.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/checkpoint_tcn-30_937.ckpt'"
    #          在 default_config.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/checkpoint_tcn-30_937.ckpt'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 上传原始 MNIST 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/tcn"
    # (6) 在网页上设置启动文件为 "eval.py"
    # (7) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (8) 创建训练作业
    ```

    ```bash
    # 在 ModelArts 上使用单卡训练adding_problem数据集
    # (1) 执行a或者b
    #       a. 在 config_addingproblem.yaml 文件中设置 "enable_modelarts=True"
    #          在 config_addingproblem.yaml 文件中设置 "data_path='/cache/data'"
    #          在 config_addingproblem.yaml 文件中设置 "train_data_path='/cache/data/AddProb/train'"
    #          在 config_addingproblem.yaml 文件中设置 "test_data_path='/cache/data/AddProb/test'"
    #          在 config_addingproblem.yaml 文件中设置 "ckpt_path='/cache/train'"
    #          (可选)在 config_addingproblem.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 config_addingproblem.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "train_data_path='/cache/data/AddProb/train'"
    #          在网页上设置 "test_data_path='/cache/data/AddProb/test'"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "ckpt_path='/cache/train'"
    #          (可选)在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 如果选择微调您的模型，上传你的预训练模型到 S3 桶上
    # (4) 生成原始 AddingProblem 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/tcn"
    # (6) 在网页上设置启动文件为 "train.py"
    # (7) 在网页上设置config_path为"../../config_addingproblem.yaml"
    # (8) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (9) 创建训练作业
    #
    # 在 ModelArts 上使用单卡验证adding_problem数据集
    # (1) 执行a或者b
    #       a. 在 config_addingproblem.yaml 文件中设置 "enable_modelarts=True"
    #          在 config_addingproblem.yaml 文件中设置 "data_path='/cache/data'"
    #          在 config_addingproblem.yaml 文件中设置 "test_data_path='/cache/data/MNIST/test'"
    #          在 config_addingproblem.yaml 文件中设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在 config_addingproblem.yaml 文件中设置 "ckpt_file='/cache/checkpoint_path/checkpoint_tcn-25_1563.ckpt'"
    #          在 config_addingproblem.yaml 文件中设置 其他参数
    #       b. 在网页上设置 "enable_modelarts=True"
    #          在网页上设置 "data_path='/cache/data'"
    #          在网页上设置 "checkpoint_url='s3://dir_to_your_pretrained/'"
    #          在网页上设置 "ckpt_file='/cache/checkpoint_path/checkpoint_tcn-25_1563.ckpt'"
    #          在网页上设置 其他参数
    # (2) 准备模型代码
    # (3) 上传你训练好的模型到 S3 桶上
    # (4) 生成原始 AddingProblem 数据集到 S3 桶上
    # (5) 在网页上设置你的代码路径为 "/path/TCN"
    # (6) 在网页上设置启动文件为 "eval.py"
    # (7) 在网页上设置config_path为"../../config_addingproblem.yaml"
    # (8) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (9) 创建训练作业
    ```

## 脚本说明

### 脚本及样例代码

```bash
    ├── TCN
        ├── README.md                       // TCN相关说明
        ├── ascend310                       // 实现310推理源代码
        ├── scripts
        │   ├──run_train_ascend.sh          // 在Ascend中训练的脚本
        │   ├──run_eval_ascend.sh           //  在Ascend中评估的脚本
        │   ├──run_infer_310.sh             //  在310中进行离线推理的脚本
        ├── src
        │   ├──dataset.py                   // 创建MNIST数据集
        │   ├──create_datasetAP.py          // 生成AddingProblem数据集
        │   ├──datasetAP.py                 // 读取AddingProblem数据集
        │   ├──TCN.py                       // TCN主要架构
        │   ├──model.py                     // 为了适应MNIST数据的模型
        │   ├──metric.py                    // 自定义模型评价标准
        │   ├──weight_norm.py               // 权重归一化
        │   ├──loss.py                      // 损失
        │   ├──lr_generator.py              // 动态学习率
        │   └──model_utils
        │      ├──config.py                 // 训练配置
        │      ├──device_adapter.py         // 获取云上id
        │      ├──local_adapter.py          // 获取本地id
        │      └──moxing_adapter.py         // 参数处理
        ├── default_config.yaml             // MNIST数据集训练参数配置文件
        ├── config_addingproblem.yaml       // AddingProblem数据集训练参数配置文件
        ├── train.py                        // 训练脚本
        ├── eval.py                         // 评估脚本
        ├── export.py                       // 导出脚本
        ├── postprocess.py                  // 310推理后处理脚本
        ├── preprocess.py                   // 310推理前处理脚本
        ├── requirements.txt                // 所需要的python库
```

### 脚本参数

```bash
train.py和config.py中主要参数如下：
--enable_modelarts：允许云上适配。
--train_data_path：到训练的路径。
--test_data_path：到评估的路径。
--output_path：保存checkpoint文件的路径。
--load_path：加载checkpoint文件。
--checkpoint_path：训练后保存的检查点文件的绝对完整路径。
--data_path：数据集所在路径。
--device_target：实现代码的设备。
--epoch_size：总训练轮次。
--epoch_change：学习率发生变化的轮次。
--batch_size：训练批次大小。
--image_height：图像高度作为模型输入（仅在permuted mnist数据集）。
--image_width：图像宽度作为模型输入（仅在permuted mnist数据集）。
--dataset_name：数据集名称。可选值为"permuted_mnist"和"adding_problem"。
--channel_size：输入通道数。
--num_classes：类别个数。
--lr：学习率。
--batch_train：运行训练集的batch size。
--batch_test：运行测试集的batch size。
--dropout：dropout大小。
--kernel_size：卷积核大小。
--level：TCN的层数。
--nhid：每一层的节点数目。
--save_checkpoint_steps：间隔多少个step保存checkpoint文件。
--keep_checkpoint_max：最多保存checkpoint文件的数目。
--N_train：adding problem数据集训练集大小（仅在adding problem数据集）。
--N_test：adding problem数据集测试集大小（仅在adding problem数据集） 。
--seq_length：adding problem数据的序列长度（仅在adding problem数据集）。
--device_id：运行设备id。
--file_name：保存MINDIR文件的名称。
--file_format：默认为"MINDIR”。

```

### 训练过程

#### 训练

- Ascend处理器环境运行permuted_mnist数据集

```bash
python train.py  --config_path ../../default_config.yaml --train_data_path data/MNIST/train --test_data_path data/MNIST/test --ckpt_path checkpoint_path > log 2>&1 &
# 或进入脚本目录，执行脚本
bash run_train_ascend.sh permuted_mnist ../data/MNIST/train ../data/MNIST/test ../checkpoint_path
```

训练结果

```bash
============== Starting Training ==============
epoch: 1 step: 937, loss is 0.41428128
epoch time: 59074.480 ms, per step time: 63.046 ms
{'Accuracy': 0.9264823717948718}
epoch: 2 step: 937, loss is 0.22595052
epoch time: 27975.444 ms, per step time: 29.856 ms
{'Accuracy': 0.9477163461538461}
...
epoch: 29 step: 937, loss is 0.015848802
epoch time: 27978.896 ms, per step time: 29.860 ms
{'Accuracy': 0.9740584935897436}
epoch: 30 step: 937, loss is 0.3095672
epoch time: 27985.821 ms, per step time: 29.867 ms
{'Accuracy': 0.9745592948717948}
```

- Ascend处理器环境运行adding_problem数据集

```bash
python train.py  --config_path ../../config_addingproblem.yaml --train_data_path data/AddProb/train --test_data_path data/AddProb/test --ckpt_path checkpoint_path > log 2>&1 &
# 或进入脚本目录，执行脚本
bash run_train_ascend.sh adding_problem ../data/AddProb/train ../data/AddProb/test ../checkpoint_path
```

训练结果

```bash
============== Starting Training ==============
epoch: 1 step: 1563, loss is 0.0007961707
epoch time: 60970.954 ms, per step time: 39.009 ms
{'Accuracy': Tensor(shape=[], dtype=Float32, value= 0.00389967)}
epoch: 2 step: 1563, loss is 0.0012416712
epoch time: 27022.374 ms, per step time: 17.289 ms
{'Accuracy': Tensor(shape=[], dtype=Float32, value= 0.00148527)}
...
epoch: 24 step: 1563, loss is 1.0663439e-05
epoch time: 26905.874 ms, per step time: 17.214 ms
{'Accuracy': Tensor(shape=[], dtype=Float32, value= 3.18133e-05)}
epoch: 25 step: 1563, loss is 2.3555269e-05
epoch time: 26909.343 ms, per step time: 17.216 ms
{'Accuracy': Tensor(shape=[], dtype=Float32, value= 1.7229e-05)}
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- Ascend处理器环境运行permuted_mnist数据集

  ```bash
  python eval.py --config_path ../../default_config.yaml --test_data_path ../data/MNIST/test --ckpt_file checkpoint_path/checkpoint_tcn-30_937.ckpt  > eval.log 2>&1 &
  #或进入脚本目录，执行脚本
  bash run_eval_ascend.sh permuted_mnist ../data/MNIST/test ../checkpoint_path/checkpoint_tcn-30_937.ckpt
  ```

  可通过"eval.log”文件查看结果。

  ```text
    ============== Starting Testing ==============
    ============== {'Accuracy': 0.9746594551282052} ==============
  ```

- Ascend处理器环境运行adding_problem数据集

  ```bash
  python eval.py --config_path ../../config_addingproblem.yaml --test_data_path /home/data/AddProb/test --ckpt_file checkpoint_add/checkpoint_tcn-25_1563.ckpt  > eval.log 2>&1 &
  #或进入脚本目录，执行脚本
  bash run_eval_ascend.sh adding_problem ../data/AddProb/test ../checkpoint_path/checkpoint_tcn-25_1563.ckpt
  ```

  可通过"eval.log”文件查看结果。

  ```text
    ============== Starting Testing ==============
    ============== {'Accuracy': 1.7229e-05} ==============  
  ```

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

### 导出MindIR

```shell
python export.py  --config_path [CONFIG_PATH] --ckpt_file [CKPT_FILE]
```

### 在Ascend310执行推理

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

## 模型描述

### 性能

在Permuted MNIST数据集上训练TCN模型：
| 参数          | TCN(permuted_mnist)                                       | TCN(permuted_mnist)                                       |
| ------------- | :-------------------------------------------------------- | --------------------------------------------------------- |
| 模型版本      | TCN                                                       | TCN                                                       |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8 |GPU NV SMX2 V100-32G |
| 上传日期      | 2021-11-26                                                | 2021-11-26                                                |
| MindSpore版本 | 1.3.0                                                     | 1.8.0(Pytorch)                                           |
| 数据集        | permuted_mnist                                            | permuted_mnist                                            |
| 训练参数      | epoch=30, steps=973, batch_size = 64, lr=0.003            | epoch=30, steps=973, batch_size = 64, lr=0.003           |
| 优化器        | Adam(weight_decay=1e-4)                                   | Adam(weight_decay=1e-4)                                   |
| 损失函数      | NLLLoss                                                   | NLLLoss                                                    |
| 输出          | 类别概率                                                  | 类别概率                                                      |
| 精度          | 0.9745                                                    | 0.972                                             |
| 速度          | 1卡：20.3 毫秒/步                                           |1卡：21.5 毫秒/步                                            |
| 调优检查点    | 895KB（.ckpt 文件）                                       | 297KB（.pkl文件）                                       |

在Adding Problem数据集上训练TCN模型：
| 参数          | TCN(adding_problem)                                       | TCN(adding_problem)                                       |
| ------------- | :-------------------------------------------------------- | --------------------------------------------------------- |
| 模型版本      | TCN                                                       | TCN                                                       |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8 |GPU NV SMX2 V100-32G  |
| 上传日期      | 2021-11-26                                                | 2021-11-26                                                |
| MindSpore版本 | 1.3.0                                                     | 1.8.0(Pytorch)                                        |
| 数据集        | adding_problem                                            | adding_problem                                            |
| 训练参数      | epoch=25, steps=1563, batch_size = 32, lr=0.004            | epoch=25, steps=1563, batch_size = 32, lr=0.004           |
| 优化器        | Adam                                   | Adam                                                      |
| 损失函数      | MSELos                                                   | MSELos                                                    |
| 输出          | 概率                                                  | 概率                                                      |
| 精度          | 1.7229e-05(loss)                                   | 5.8e-05(loss)                                          |
| 速度          | 1卡：11.6 毫秒/步                                           | 1卡：17.1毫秒/步                                            |
| 调优检查点    | 978KB（.ckpt 文件）                                       |471KB（.pkl文件）                                 |

## 随机情况说明

在dataset.py和create_datasetAP.py中设置了随机种子

## ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。
