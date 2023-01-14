# Contents

[View English](./README.md)

- [Contents](#contents)
- [MelGAN描述](#melgan描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# MelGAN描述

MelGAN是一种GAN网络，可将音频Mel谱特征转化为高质量的音频。该网络不需要任何硬件优化技巧，就可以在GPU或CPU上实现快速的音频合成。对比于相同功能的Wavenet网络，速度提高1000倍以上。

[论文](https://arxiv.org/abs/1910.06711):  Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville. "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis.".

# 模型架构

MelGAN模型是非自回归全卷积模型。它的参数比同类模型少得多，并且对于看不见的说话人也有很好的效果。它的生成器由 4 个上采样层和 4 个残差堆栈组成，而判别器是多尺度架构。跟文章设计的结构不一样的是，我们修改了判别器中部分卷积核的大小，同时我们使用一维卷积代替了判别器中的avpool。

# 数据集

所使用的数据集: [LJ Speech](<https://keithito.com/LJ-Speech-Dataset/>)

- Dataset size：2.6GB，包含13,100条只有一个说话人的短语音。语音的内容来自7本纪实书籍。

- 数据格式：每条语音文件都是单声道、16-bit以及采样率为22050。
    - 语音需要被处理为Mel谱, 可以参考脚本[Mel谱处理脚本](https://github.com/seungwonpark/melgan/blob/master/preprocess.py)。非CUDA环境需删除`utils/stfy.py`中的`.cuda()`，因为要保存`npy`格式的数据，所以`preproccess.py`也需要修改以下，参考代码如下：

    ```
    # 37 - 38 行
    melpath = wavpath.replace('.wav', '.npy').replace('wavs', 'mel')
    if not os.path.exists(os.path.dirname(melpath)):
        os.makedirs(os.path.dirname(melpath), exist_ok=True)
    np.save(melpath, mel.squeeze(0).detach().numpy())
    ```

    - 数据目录结构如下:

      ```
        ├── dataset
            ├── val
            │   ├─ wavform1.npy
            │   ├─ ...
            │   └─ wavformn.npy
            ├── train
                ├─ wav
                │    ├─wavform1.wav
                │    ├─ ...
                │    └─wavformn.wav
                └─ mel
                    ├─wavform1.npy
                    ├─ ...
                    └─wavformn.npy
      ```

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

  ```yaml
  # 在yaml文件中修改数据路径, 以数据集LJSpeech为例
  data_path:/home/LJspeech/dataset/

  # 在继续训练之前在yaml文件中加入chcekpoint_path路径
  checkpoint_path:/home/model/saved_model/melgan_20-215_176000.ckpt
  ```

  ```python
  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 使用脚本的单卡训练
  bash run_standalone_train_ascend.sh DEVICE_ID

  # 使用脚本的分布式训练
  bash run_distribute_train_ascend.sh RANK_TABLE_FILE

  # 运行推理示例
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)，具体操作如下：

  ```bash
    # 在modelarts上使用分布式训练示例：
    # (1) 选择a或者b其中一种方式。
    #       a. 在yaml文件中设置 "enable_modelarts=True"。
    #          在yaml文件上设置其它所需的参数。
    #       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
    #          在modelarts的界面上增加其它所需的参数。
    # (2) 在modelarts的界面上设置代码的路径 "/path/MelGAN"。
    # (3) 在modelarts的界面上设置模型的启动文件 "train.py" 。
    # (4) 在modelarts的界面上设置模型的数据路径 "Dataset path",模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
    # (5) 开始模型的训练。

    # 在modelarts上使用模型推理的示例
    # (1) 把训练好的模型复制到桶的对应位置。
    # (2) 选择a或者b其中一种方式。
    #       a. 在yaml文件中设置 "enable_modelarts=True"。
    #          在yaml文件中设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt"
    #          在yaml文件中设置 "checkpoint_url=/The path of checkpoint in S3/"
    #       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
    #          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
    #          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
    # (3) 在modelarts的界面上设置代码的路径"/path/MelGAN"。
    # (4) 在modelarts的界面上设置模型的启动文件 "eval.py"。
    # (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
    # (6) 开始模型的推理。
  ```

# 脚本说明

## 脚本及样例代码

```text
├── melgan
    ├── README.md                     // MelGAN说明
    ├── README_CN.md                  // MelGAN中文说明
    ├── ascend310_infer               // 实现310推理源代码
    ├── scripts
    │   ├──run_standalone_train_ascend           // 启动Ascend单机训练
    │   ├──run_distribute_train_ascend.sh        // 启动Ascend分布式训练（8卡）
    │   ├──run_eval_ascend.sh                    // 启动评估
    │   ├──run_infer_310.sh                      // 启动310评估
    ├── src
    │   ├──dataset.py           // 创建数据集
    │   ├──model.py             // 生成器和判别器网络结构
    │   ├──loss.py              // 计算损失函数
    │   ├── model_utils
    │       ├──config.py                      // 参数配置
    │       ├──device_adapter.py              // 设备配置
    │       ├──local_adapter.py               // 本地设备配置
    │       ├──moxing_adapter.py              // modelarts设备配置
    ├── train.py                 // 训练网络脚本
    ├── eval.py                  //  评估网络脚本
    ├── config.yaml              // 参数配置项
     ├── export.py               // 将checkpoint文件导出到air/mindir
```

## 脚本参数

在config.yaml中可以同时配置训练参数和评估参数。

- config for MelGAN, LJ Speech dataset

  ```python
  'pre_trained': 'Flase'    # 是否基于预训练模型训练
  'checkpoint_path':  './melgan_20-215_176000.ckpt'
                            # 预训练模型路径
  'lr_g': 0.0001            # 生成器初始学习率
  'lr_d': 0.0001            # 判别器初始学习率
  'batch_size': 4           # 训练批次大小（使用单卡训练时可适当增大为16）
  'epoch_size': 5000        # 总训练epoch数
  'momentum': 0.9           # 权重衰减值
  'leaky_alpha': 0.2        # leaky relu参数
  'train_length': 64        # 训练时输入序列的帧数(最大值:240)

  'beta1':0.9               # 第一矩估计的指数衰减率
  'beta2':0.999             # 第二矩估计的指数衰减率
  'weight_decay':0.0        # 权重衰减值（L2惩罚）

  'hop_size': 256           # Mel谱中一帧的长度
  'mel_num': 80             # Mel谱中通道数
  'filter_length': 1024     # n点短时傅里叶变换
  'win_length': 1024        # 窗函数长度
  'segment_length': 16000   # 计算Mel谱时的最大长度
  'sample': 22050           # 训练音频采样率
  'data_path':'/home/datadisk0/voice/melgan/data/'
                            # 训练数据绝对路径
  'save_steps': 4000        # 保存点间隔.
  'save_checkpoint_name': 'melgan'
                            # 保存模型的名字.
  'save_checkpoint_path': './saved_model'
                            # 保存模型的绝对路径
  'eval_data_path': '/home/datadisk0/voice/melgan/val_data/'
                            # 验证集绝对路径
  'eval_model_path': './melgan_20-215_176000.ckpt'
                            # 验证模型路径
  'output_path': 'output/'  # 验证结果保存路径
  'eval_length': 240        # 验证时输入序列的帧数 (最大值:240)
  ```

## 训练过程

### 训练

  ```python
  python train.py > train.log 2>&1 &
  ```

  或通过shell脚本开始训练：

  ```bash
  bash scripts/run_standalone_train_ascend.sh DEVICE_ID
  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```python
  # grep "loss_G= " train.log
  1epoch 1iter loss_G=27.5 loss_D=27.5 0.30s/it
  1epoch 2iter loss_G=27.4 loss_D=27.4 0.30s/it
  ...
  ```

   模型保存在指定目录下。

### 分布式训练

  ```python
  bash scripts/run_distribute_train_ascend.sh
  ```

  上述shell脚本将在后台运行分布训练。您可以通过train_parallel[X]/log文件查看结果。损失值如下所示：

  ```python
  # grep "result: " train_parallel*/log
  train_parallel0/log:1epoch 1iter loss_G=27.5 loss_D=27.5 0.30s/it
  train_parallel0/log:1epoch 2iter loss_G=27.4 loss_D=27.4 0.30s/it
  ...
  train_parallel1/log:1epoch 1iter loss_G=27.5 loss_D=27.5 0.30s/it
  train_parallel1/log:1epoch 2iter loss_G=27.4 loss_D=27.4 0.30s/it
  ...
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估LJ Speech数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“/username/melgan/saved_model/melgan_20-215_176000.ckpt”。

  ```python
  bash run_eval_asecnd.sh DEVICE_ID PATH_CHECKPOINT
  ```

  上述命令将在后台运行，您可以在"output"文件夹中查看生成的音频文件

## 导出过程

### 导出

```shell
python export.py  --format [EXPORT_FORMAT] --checkpoint_path [CKPT_PATH]
```

## 推理过程

### 推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在进行推理之前我们需要先导出模型。Air模型只能在昇腾910环境上导出，mindir可以在任意环境上导出。batch_size只支持1。

```bash
bash run_infer_cpp.sh [MODEL_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

`DEVICE_ID` 可选，默认值为 0。
`DEVICE_TYPE` 可以为Ascend, GPU, 或CPU。

# 模型描述

## 性能

### 评估性能

| 参数                  | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| 模型版本          | MelGAN                                                       |
| 资源                   | Ascend 910；CPU 2.60GHz，56cores；Memory 755G; OS Euler2.8   |
| 更新时间           | 10/11/2021                                                   |
| MindSpore版本         | 1.3.0                                                        |
| 数据集              | LJ Speech                                                    |
| 训练参数           | epoch=3000, steps=2400000, batch_size=16, lr=0.0001          |
| 优化器              | Adam                                                         |
| 损失函数           | L1 Loss                                                      |
| 输出                   | waveforms                                                    |
| 速度                   | 1pc: 320 ms/step; 8pc: 310 ms/step                           |
| 训练时长            | 1pc: 220 hours; 8pc: 25 hours                                |
| 损失值               | loss_G=340.123449 loss_D=4.457899                            |
| 参数量 (M)         | generator : 4.26; discriminator : 56.4                       |
| 微调检查点        | 361.490M (.ckpt file)                                        |

### 推理性能

| 参数           |                             |                             |
| ------------------- | --------------------------- | --------------------------- |
| 模型版本   | MelGAN                      |                             |
| 资源            | Ascend 910                  | Ascend 310                  |
| 上传日期   | 10/11/2021                  | 10/11/2021                  |
| MindSpore版本  | 1.5.0                       | 1.5.0                       |
| 数据集       | LJ Speech                   | LJ Speech                   |
| 批大小         | 1                           | 1                           |
| 输出            | waveforms                   | waveforms                   |
| 准确率       | 3.2(mos分)               | 3.2(mos分)               |
| 推理的模型 | 361.490M (.ckpt file)    | 18.550M                     |

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models/tree/master).
