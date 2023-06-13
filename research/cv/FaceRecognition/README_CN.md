# 目录

- [人脸识别描述](#人脸识别描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [运行示例](#运行示例)
- [模型说明](#模型说明)
    - [性能](#性能)
- [ModelZoo主页](#modelzoo主页)

# [人脸识别描述](#目录)

这是一个基于Resnet的人脸识别网络，支持在Ascend 910、CPU或GPU上进行训练和评估。

残差神经网络（ResNet）是由微软研究院的Kaiming He等四位中国人提出的。通过ResNet单元，成功地训练了152层神经网络，并在ilsvrc2015中获得了冠军。Top 5的误差率为3.57%，参数量低于vggnet，效果非常突出。传统的卷积网络或全连接网络或多或少会有信息丢失。同时，会导致梯度消失或爆炸，从而导致深度网络训练的失败。ResNet在一定程度上解决了这个问题。通过将输入信息传递到输出，信息的完整性得到保护。整个网络只需要学习输入输出差异的部分，简化了学习目标和难度，ResNet结构可以加速神经网络的训练，模型的准确性也大大提高。同时，ResNet非常受欢迎，甚至可以直接用于ConceptNet网络。

[论文](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"

# [模型架构](#目录)

人脸识别使用Resnet网络执行特征提取，更多详细信息，请点击[链接](https://arxiv.org/pdf/1512.03385.pdf)。

# [数据集](#目录)

在本例中，我们使用大约470万张人脸图像作为训练数据集，110万张人脸图像作为评估数据集，您也可以使用自己的数据集或开源数据集（例如，Face_emore）。
目录结构如下：

```python
.
└─ dataset
  ├─ train dataset
    ├─ ID1
      ├─ ID1_0001.jpg
      ├─ ID1_0002.jpg
      ...
    ├─ ID2
      ...
    ├─ ID3
      ...
    ...
  ├─ test dataset
    ├─ ID1
      ├─ ID1_0001.jpg
      ├─ ID1_0002.jpg
      ...
    ├─ ID2
      ...
    ├─ ID3
      ...
    ...
```

# [环境要求](#目录)

- 硬件（Ascend, CPU, GPU）
    - 使用Ascend、CPU或GPU处理器来搭建硬件环境。
    硬件环境
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## [脚本及样例代码](#目录)

整体代码结构如下：

```python
└─ FaceRecognition
  ├── ascend310_infer
  ├── README.md                             // 关于人脸识别的说明
  ├── scripts
  │   ├── run_distribute_train_base.sh      // Ascend分布式训练shell脚本
  │   ├── run_distribute_train_beta.sh      // Ascend分布式训练shell脚本
  │   ├── run_distribute_train_for_gpu.sh   // GPU分布式训练shell脚本
  │   ├── run_eval.sh                       // Ascend评估shell脚本
  │   ├── run_eval_cpu.sh                   // CPU评估shell脚本
  │   ├── run_eval_gpu.sh                   // GPU评估shell脚本
  │   ├── run_export.sh                     // 导出air/mindir模型的shell脚本
  │   ├── run_standalone_train_base.sh      // Ascend单机训练shell脚本
  │   ├── run_standalone_train_beta.sh      // Ascend单机训练shell脚本
  │   ├── run_standalone_train_for_gpu.sh   // GPU单机训练shell脚本
  │   ├── run_train_base_cpu.sh             // CPU训练shell脚本
  │   ├── run_train_btae_cpu.sh             // CPU训练shell脚本
  ├── src
  │   ├── backbone
  │   │   ├── head.py                       // 头单元
  │   │   ├── resnet.py                     // resnet架构
  │   ├── callback_factory.py               // 回调日志
  │   ├── custom_dataset.py                 // 自定义数据集和采样器
  │   ├── custom_net.py                     // cell自定义
  │   ├── dataset_factory.py                // 创建数据集
  │   ├── init_network.py                   // 初始化网络参数
  │   ├── my_logging.py                     // 日志格式设置
  │   ├── loss_factory.py                   // 损失计算
  │   ├── lrsche_factory.py                 // 学习率调度
  │   ├── me_init.py                        // 网络参数init方法
  │   ├── metric_factory.py                 // FC层指标
  ── model_utils
  │   ├── __init__.py                       // 初始化文件
  │   ├── config.py                         // 参数分析
  │   ├── device_adapter.py                 // 设备适配器
  │   ├── local_adapter.py                  // 本地适配器
  │   ├── moxing_adapter.py                 // moxing适配器
  ├─ base_config.yaml                       // 参数配置
  ├─ base_config_cpu.yaml                   // 参数配置
  ├─ beta_config.yaml                       // 参数配置
  ├─ beta_config_cpu.yaml                   // 参数配置
  ├─ inference_config.yaml                  // 参数配置
  ├─ inference_config_cpu.yaml              // 参数配置
  ├─ train.py                               // 训练脚本
  ├─ eval.py                                // 评估脚本
  └─ export.py                              // 导出air/mindir模型
```

## [运行示例](#目录)

### 训练

- 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：如果在GPU上运行，请在python命令中添加`--device_target=GPU`或使用GPU shell脚本（"xxxgpu.sh"）。
- 在运行网络之前，准备hccl_8p.json文件。
    - 生成hccl_8p.json，运行utils/hccl_tools/hccl_tools.py脚本。
      以下参数"[0-8)"表示生成0~7卡的hccl_8p.json文件。
        - 该命令生成的json文件名为hccl_8p_01234567_{host_ip}.json。为了方便表达，使用hccl_8p.json表示该json文件。

      ```
      python hccl_tools.py --device_num "[0,8)"
      ```

- 在运行网络之前，准备数据集，并在xxx_config.yaml文件中设置"data_dir='/path_to_dataset/'"。
- 如果使用beta模式进行训练，则准备已训练的base模型（.ckpt文件），并在运行网络之前在beta_config.yaml文件上设置"pretrained='/path_to_checkpoint_path/model.ckpt'"。

- 单机模式（Ascend）

    - base模型

      ```bash
      cd ./scripts
      bash run_standalone_train_base.sh [USE_DEVICE_ID]
      ```

      例如：

      ```bash
      cd ./scripts
      bash run_standalone_train_base.sh 0
      ```

    - beta模型

      ```bash
      cd ./scripts
      bash run_standalone_train_beta.sh [USE_DEVICE_ID]
      ```

      例如：

      ```bash
      cd ./scripts
      bash run_standalone_train_beta.sh 0
      ```

- 单机模式（GPU）

    - base/beta模型

      ```bash
      cd ./scripts
      bash run_standalone_train_for_gpu.sh [base/beta] [DEVICE_ID](optional)
      ```

      例如：

      ```bash
      #base
      cd ./scripts
      bash run_standalone_train_for_gpu.sh base 3
      #beta
      cd ./scripts
      bash run_standalone_train_for_gpu.sh beta 3
      ```

- 分布式（Ascend，推荐）

    - base模型

      ```bash
      cd ./scripts
      bash run_distribute_train_base.sh [RANK_TABLE]
      ```

      例如：

      ```bash
      cd ./scripts
      bash run_distribute_train_base.sh ./rank_table_8p.json
      ```

    - beta模型

      ```bash
      cd ./scripts
      bash run_distribute_train_beta.sh [RANK_TABLE]
      ```

      例如：

      ```bash
      cd ./scripts
      bash run_distribute_train_beta.sh ./rank_table_8p.json
      ```

- 分布式（GPU）

    - base模型

      ```bash
      cd ./scripts
      bash run_distribute_train_for_gpu.sh [RANK_SIZE] [base/beta] [CONFIG_PATH](optional)
      ```

      例如：

      ```bash
      #base
      cd ./scripts
      bash run_distribute_train_for_gpu.sh 8 base
      #beta
      cd ./scripts
      bash run_distribute_train_for_gpu.sh 8 beta
      ```

- 单机模式（CPU）

    - base模型

      ```bash
      cd ./scripts
      bash run_train_base_cpu.sh
      ```

      例如：

      ```bash
      cd ./scripts
      bash run_train_base_cpu.sh
      ```

    - beta模型

      ```bash
      cd ./scripts
      bash run_train_beta_cpu.sh
      ```

      例如：

      ```bash
      cd ./scripts
      bash run_train_beta_cpu.sh
      ```

- ModelArts（如果想在ModelArts中运行，请查看[ModelArts官方文档](https://support.huaweicloud.com/modelarts/)，并按照以下方式开始训练）

    - base模型

      ```python
      # （1）在网页上添加"config_path='/path_to_code/base_config.yaml'"。
      # （2）执行a或b。
      #       a. 在base_config.yaml文件中设置"enable_modelarts=True"。
      #          在base_config.yaml文件中设置"is_distributed=1"。
      #          在base_config.yaml文件中设置其他参数。
      #       b. 在网页上添加"enable_modelarts=True"。
      #          在网页上添加"is_distributed=1"。
      #          在网页上添加其他参数。
      # （3）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
      # （4）在网页上设置代码目录为"/path/FaceRecognition"。
      # （5）在网页上设置启动文件为"train.py"。
      # （6）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
      # （7）创建作业。
      ```

    - beta模型

      ```python
      # （1）复制或上传训练好的模型到S3桶。
      # （2）在网页上添加"config_path='/path_to_code/beta_config.yaml'"。
      # （3）执行a或b。
      #       a. 在beta_config.yaml文件中设置"enable_modelarts=True"。
      #          在base_config.yaml文件中设置"is_distributed=1"。
      #          在beta_config.yaml文件中设置"pretrained='/cache/checkpoint_path/model.ckpt'"。
      #          在beta_config.yaml文件中设置"checkpoint_url=/The path of checkpoint in S3/"。
      #       b. 在网页上添加"enable_modelarts=True"。
      #          在网页上添加"is_distributed=1"。
      #          在default_config.yaml文件中添加"pretrained='/cache/checkpoint_path/model.ckpt'"。
      #          在default_config.yaml文件中添加"checkpoint_url=/The path of checkpoint in S3/"。
      # （4）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
      # （5）在网页上设置代码目录为"/path/FaceRecognition"。
      # （6）在网页上设置启动文件为"train.py"。
      # （7）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
      # （8）创建作业。
      ```

可在"./scripts/data_parallel_log_[DEVICE_ID]/outputs/logs/[TIME].log"或"./scripts/log_parallel_graph/face_recognition_[DEVICE_ID].log"中获得每个epoch的损失值：

```python
epoch[0], iter[100], loss:(Tensor(shape=[], dtype=Float32, value= 50.2733), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 32768)), cur_lr:0.000660, mean_fps:743.09 imgs/sec
epoch[0], iter[200], loss:(Tensor(shape=[], dtype=Float32, value= 49.3693), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 32768)), cur_lr:0.001314, mean_fps:4426.42 imgs/sec
epoch[0], iter[300], loss:(Tensor(shape=[], dtype=Float32, value= 48.7081), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 16384)), cur_lr:0.001968, mean_fps:4428.09 imgs/sec
epoch[0], iter[400], loss:(Tensor(shape=[], dtype=Float32, value= 45.7791), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 16384)), cur_lr:0.002622, mean_fps:4428.17 imgs/sec

...
epoch[8], iter[27300], loss:(Tensor(shape=[], dtype=Float32, value= 2.13556), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4429.38 imgs/sec
epoch[8], iter[27400], loss:(Tensor(shape=[], dtype=Float32, value= 2.36922), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4429.88 imgs/sec
epoch[8], iter[27500], loss:(Tensor(shape=[], dtype=Float32, value= 2.08594), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4430.59 imgs/sec
epoch[8], iter[27600], loss:(Tensor(shape=[], dtype=Float32, value= 2.38706), Tensor(shape=[], dtype=Bool, value= False), Tensor(shape=[], dtype=Float32, value= 65536)), cur_lr:0.004000, mean_fps:4430.37 imgs/sec
```

### 评估

```bash
cd ./scripts
sh run_eval.sh [USE_DEVICE_ID]
```

在"./scripts/log_inference/outputs/models/logs/[TIME].log"中查看以下结果：
[test_dataset]: zj2jk=0.9495, jk2zj=0.9480, avg=0.9487

如果想在ModelArts中运行，请查看[ModelArts官方文档](https://support.huaweicloud.com/modelarts/)，并按照以下方式开始评估。

```python
# 在ModelArts上运行评估
# （1）复制或上传训练好的模型到S3桶。
# （2）在网页上添加"config_path='/path_to_code/inference_config.yaml'"。
# （3）执行a或b。
#       a. 在default_config.yaml文件中设置"weight='/cache/checkpoint_path/model.ckpt'"。
#          在default_config.yaml文件中设置"checkpoint_url=/The path of checkpoint in S3/"。
#       b. 在网页上添加"weight='/cache/checkpoint_path/model.ckpt'"。
#          在网页上添加"checkpoint_url=/The path of checkpoint in S3/"。
# （4）上传zip数据集到S3桶（也可以上传源数据集，但速度很慢）。
# （5）在网页上设置代码目录为"/path/FaceRecognition"。
# （6）在网页上设置启动文件为"eval.py"。
# （7）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
# （8）创建作业。
```

### 转换模型

如果您想推断Ascend 310上的网络，则应将模型转换为AIR/MINDIR：

```bash
cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```

- `BATCH_SIZE`：应为0。
- `PRETRAINED_BACKBONE`：必填项，必须指定包括文件名在内的MINDIR路径。
- `USE_DEVICE_ID`：必填项，默认值为0。

例如：

```bash
cd ./scripts
sh run_export.sh 1 0 ./0-1_1.ckpt
```

```python
# 在ModelArts上运行导出
# （1）复制或上传训练好的模型到S3桶。
# （2）在网页上添加"config_path='/path_to_code/inference_config.yaml'"。
# （3）执行a或b。
#       a. 在inference_config.yaml文件中设置"pretrained='/cache/checkpoint_path/model.ckpt'"。
#          在inference_config.yaml文件中设置"checkpoint_url='/The path of checkpoint in S3/'"。
#          在inference_config.yaml文件中设置"batch_size=1"。
#       b. 在网页上添加"pretrained=/cache/checkpoint_path/model.ckpt"。
#          在网页上添加"checkpoint_url=/The path of checkpoint in S3/"。
#          在网页上添加"batch_size=1"。
# （4）在网页上设置代码目录为"/path/FaceRecognition"。
# （5）在网页上设置启动文件为"export.py"。
# （6）在网页上设置自己的"Dataset path"、"Output file path"、"Job log path"。
# （7）创建作业。
```

### 推理

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

```bash
cd ./scripts
sh run_infer_310.sh [MINDIR_PATH] [USE_DEVICE_ID]
```

例如：

```bash
cd ./scripts
sh run_infer_310.sh ../facerecognition.mindir 0
```

如果'dis_dataset'的范围是文件夹'68680'到'68725'，你将在"./scripts/acc.log"中获得以下结果：
[test_dataset]: zj2jk=0.9863, jk2zj=0.9851, avg=0.9857

# [模型说明](#目录)

## [性能](#目录)

### 训练性能

| 参数                | 人脸识别                                           | 人脸识别  |
| -------------------------- | ----------------------------------------------------------- | ------------------ |
| 模型版本             | V1                                                         | V1                |
| 资源                  | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8| NV PCIE V100-32G   |
| 上传日期             | 14/10/2021                       | 14/10/2021|
| MindSpore版本         | 1.5.0                                                       | 1.5.0              |
| 数据集                   | 470万张图像                                         | 470万张图像|
| 训练参数       | epoch=18(base:9, beta:9), batch_size=192, momentum=0.9 | epoch=18(base:9, beta:9), batch_size=192, momentum=0.9 |
| 优化器                 | 动量                                                   | 动量          |
| 损失函数             | Cross Entropy                                               | Cross Entropy      |
| 输出                   | 概率                                                | 概率       |
| 速度                     | base: 单卡：350-600 fps; 8卡: 2500-4500 fps;    | base: 单卡：290-310 fps, 8卡: 2050-2150 fps;  |
|                            | beta: 单卡：350-600 fps; 8卡: 2500-4500 fps;    | beta: 单卡：400-430 fps, 8卡: 2810-2860 fps;  |
| 总时长                | 单卡：NA hours; 8卡: 10 hours   | 单卡：NA hours; 8卡: 5.6(base) + 4.2(beta) hours |
| 微调检查点| 768M（.ckpt，base）,582M（.ckpt，beta） | 768M（.ckpt，base）,582M（.ckpt，beta） |

### 评估性能

| 参数         | 人脸识别           | 人脸识别           |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本      | V1                         | V1                         |
| 资源           | Ascend 910; EulerOS 2.8     | NV SMX2 V100-32G           |
| 上传日期      | 14/10/2021| 29/07/2021|
| MindSpore版本  | 1.5.0                       | 1.3.0                       |
| 数据集            | 110万张图像         | 110万张图像         |
| batch_size          | 512                         | 512                         |
| 输出            | ACC                        | ACC                        |
| ACC                | 0.9                         | 0.9                         |
| 推理模型| 582M（.ckpt）          | 582M（.ckpt）          |

# [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
