# 目录

- [目录](#目录)
    - [LPCNet 描述](#lpcnet描述)
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
        - [生成网络输入数据](#生成网络输入数据)
        - [导出MindIR](#导出mindir)
        - [Ascend310上运行推理](#Ascend310上运行推理)
        - [结果](#结果)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [评估性能](#评估性能)
        - [推理性能](#推理性能)
    - [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## [LPCNet描述](#目录)

LPCNet是一种将线性预测与递归神经网络相结合的低比特率神经网络声码器。

[论文](https://jmvalin.ca/papers/lpcnet_codec.pdf): J.-M. Valin, J. Skoglund, A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet, Proc. INTERSPEECH, arxiv:1903.12087, 2019.

## [模型架构](#目录)

LPCNet分为两部分：帧率网络和采样率网络。帧率网络由两个卷积层和两个全连接层组成，从5帧上下文中提取特征。采样率网络由两个GRU层和一个双全连接层（在全连接层的基础上进行修改）组成，且第一个GRU层的权重进行了稀疏化。采样率网络获取帧率网络提取的特征，同时对当前时间步进行线性预测，对前一个时间步进行采样结束激励，并通过sigmoid函数和二叉树概率表示预测当前激励。

## [数据集](#目录)

使用的数据集：[LibriSpeech](<https://www.openslr.org/12/>)

- 数据集大小：6.3G
    - 训练集为train-clean-100.tar.gz，测试数据集为test-clean.tar.gz。
- 数据格式：二进制文件
    - 注：LPCNet是一种混合压缩方法，通过量化18个巴克尺度倒谱系数和2个音调参数来实现压缩。网络从这些量化特征重建原始音频。

- 下载数据集（只需要.flac文件）。目录结构如下：

    ```train-clean-100
    ├─README.TXT
    ├─READERS.TXT
    ├─CHAPTERS.TXT
    ├─BOOKS.TXT
    └─train-clean-100
    ├─19
        ├─198
        ├─19-198.trans.txt
        ├─19-198-0001.flac
        ├─19-198-0002.flac
        ...
        └─19-198-0014.flac
    ```

## [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速入门](#目录)

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

```bash
# 输入脚本目录，开始编译特征量化。
bash run_compile.sh

# 生成训练数据（运行命令前需按照顺序安装FLAC和SoX）。
bash run_process_train_data.sh [TRAIN_DATASET_PATH] [OUTPUT_PATH]
# 示例：bash run_process_train_data.sh ./train-100-clean ~/dataset_path/training_dataset/

# 单卡训练LPCNet
bash run_standalone_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [CHECKPOINT_SAVE_PATH]
# 示例: bash run_standalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/

# 或者分布式训练（2卡、4卡、8卡）LPCNet
bash run_distribute_train_ascend.sh [PREPROCESSED_TRAINING_DATASET_PATH] [RANK_SIZE] [RANK_TABLE]
# 示例：bash run_distribute_train_ascend.sh ~/dataset_path/training_dataset/ 8 ./hccl_8p.json

# 生成测试数据（从test-clean中选择10个文件进行评估）。
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# 示例：bash run_process_eval_data.sh ./dataset_path/test_dataset ~/dataset_path/test_features/

# 评估LPCNet
bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
# 示例：bash run_eval_ascend.sh ~/dataset_path/test_features/ ./eval_results/ ./ckpt/lpcnet-4_37721.ckpt
```

## [脚本说明](#目录)

### [脚本及示例代码](#目录)

```text
├── audio
    ├── LPCNet
        ├── README.md                          // LPCNet说明
        ├── requirements.txt                   // 需求文件
        ├── scripts
        │   ├──run_compile.sh                  // 进行代码特征提取和量化编译
        │   ├──run_infer_310.sh                // Ascend 310上运行推理
        │   ├──run_proccess_train_data.sh      // 从.flac文件生成训练数据集
        |   ├──run_process_eval_data.sh        // 从.flac文件生成评估数据集
        │   ├──run_stanalone_train_ascend.sh   // Ascend上运行单卡训练
        │   ├──run_distribute_train_ascend.sh  // Ascend上运行分布式（2卡、4卡、8卡）训练
        ├── src
        │   ├──rnns                            // 动态GRU实现
        │   ├──dataloader.py                   // 加载数据用于模型训练
        │   ├──lossfuncs.py                    // 损失函数
        │   ├──lpcnet.py                       // LPCNet实现
        │   ├──mdense.py                       // 双全连接层实现
        │   ├──train_lpcnet_parallel.py        // 分布式训练
        |   └──ulaw.py                         // U-Law量化
        ├── third_party                        // （基于C++）特征提取和量化
        ├── ascend310_infer                    // （基于C++）在Ascend 310上运行推理
        ├── train.py                           // 用于在Ascend上运行训练的主程序
        ├── process_data.py                    // 从KITTI .bin文件主程序生成训练数据集
        ├── eval.py                            // 评估主程序
        ├──export.py                           // 导出模型用于推理
```

### [脚本参数](#目录)

```text
train.py中的主要参数包括：
features：二进制特征的路径
data：与特征相对应的16位PCM路径
output：存储.ckpt文件的路径
--batch-size：训练的batch size
--epochs：训练的epochs总数
--device：代码实现的设备，可选值包括"Ascend"，"GPU"，"CPU"。
--checkpoint：训练后保存检查点文件的路径。（建议设置）

eval.py中的主要参数包括：
test_data_path：测试数据集的路径，测试数据是由run_process_eval_data.sh提取和量化的特征
output_path：存储解压/重构文件的路径
model_file：待加载检查点文件的路径
```

### [训练过程](#目录)

#### 训练

- 在Ascend上运行

  ```bash
  python train.py [FEATURES_FILE] [AUDIO_DATA_FILE] [CHECKPOINT_SAVE_PATH] --device=[DEVICE_TARGET] --batch-size=[batch-size]
  # 或输入脚本目录，运行单卡训练脚本
  bash run_stanalone_train_ascend.sh ~/dataset_path/training_dataset/ ./ckpt/
  # 或输入脚本目录，运行2卡、4卡或8卡训练脚本
  bash run_distribute_train_ascend.sh ~/dataset_path/training_dataset/ 8 ./hccl_8p.json
  ```

  训练后，得到的损失值如下：

  ```bash
  epoch: 1 step: 37721, loss is 4.2791853
  ...
  epoch: 4 step: 37721, loss is 3.7296906
  ...
  ```

  模型检查点将保存在指定的目录中。

### [评估过程](#目录)

#### 评估

运行以下命令前，请检查用于评估的检查点路径。

- 在Ascend上运行

  ```bash
  python eval.py [TEST_DATA_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
  # 或输入脚本目录，运行评估脚本
  bash run_eval_ascend.sh [TEST_DATASET_PATH] [OUTPUT_PATH] [CHECKPOINT_SAVE_PATH]
  ```

## [推理过程](#目录)

### 生成网络输入数据

```shell
# 输入脚本目录，运行run_process_data.sh脚本
bash run_process_eval_data.sh [EVAL_DATASET_PATH] [OUTPUT_PATH]
# 示例：bash run_process_eval_data.sh ./dataset_path/test_dataset ~/dataset_path/test_features/
```

### 导出MindIR

```shell
python export.py --ckpt_file=[CKPT_PATH] --max_len=[MAX_LEN] --out_file=[OUT_FILE]
# 示例：
python export.py --ckpt_file='./checkpoint/ms-4_37721.ckpt'  --out_file=lpcnet --max_len 500
# 注：max_len表示可以处理的最大10毫秒帧数，超出的音频将被截断。
```

必须设置ckpt_file参数。

### [在Ascend 310上运行推理](#目录)

**推理前，请参考[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)设置环境变量。**

进行推理前，我们需要通过`export.py`脚本导出mindir文件。如下为导出MindIR模型进行推理的例子。

```bash
# 输入脚本目录，运行run_infer_310.sh脚本。
bash run_infer_310.sh [ENCODER_PATH] [DECODER_PATH] [DATA_PATH] [DEVICE_ID]（可选）
```

- `ENCODER_PATH`：\*_enc mindir的绝对路径
- `DECODER_PATH`：\*_dec mindir的绝对路径
- `DATA_PATH`：输入数据的绝对路径（在此路径下，应该有一个用run_process_eval_data.sh提取的\*.f32文件）
- `DEVICE_ID`：代码运行所在设备的ID

### 结果

推理结果保存在./infer.log目录下。

## [模型说明](#目录)

### [性能](#目录)

#### 评估性能

| 参数         | Ascend                                                      |
| ------------- | ------------------------------------------------------------ |
| 网络名称| LPCNet                                                  |
| 资源 | Ascend 910;CPU 191核；内存755G                           |
| 上传日期| 待定                                                         |
| MindSpore版本| 1.5.0                                                       |
| 数据集| 6.16G的干净语音                              |
| 训练参数| epoch=4, batch_size=64 , lr=0.001 |
| 优化器| Adam                                                        |
| 损失函数| SparseCategoricalCrossentropy                                          |
| 输出  | 分布                                        |
| 损失     | 3.24957                                                         |
| 速度    | 单卡：182个样本/秒；8卡：154个样本/秒            |
| 总时长| 单卡：17.5小时；8卡：2.5小时                              |
| 参数量（M）| 120M                                                       |
| 微调检查点| 15.7M（.ckpt文件）                                              |
| 脚本  | [LPCNet](https://gitee.com/mindspore/models/tree/master/official/audio/LPCNet)|

### 推理性能

| 参数       | Ascend                                                      |
| ----------------- | ------------------------------------------------------------ |
| 网络名称     | LPCNet                                                  |
| 资源         | Ascend 910                                                  |
| 上传日期    | 待定                                                         |
| MindSpore版本| 1.5.0                                                       |
| 数据集          | LibriSpeach test-clean中的10个文件构建的二进制特征文件                                      |
| batch_size       | 1                                                            |
| 输出           | 重构16位PCM音频|
| 准确率        | 0.004 (MSE)|

## [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
