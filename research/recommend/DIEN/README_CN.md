# 目录

<!-- TOC -->

- [目录](#目录)
    - [DIEN概述](#概述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
    - [模型描述](#模型描述)
        - [性能](#性能)
            - [训练性能](#训练性能)
            - [评估性能](#评估性能)
            - [推理和解释性能](#推理和解释性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

# 概述

DIEN（Deep Interest Evolution Network）是阿里提出的应用于CTR预测的深度兴趣进化网络。

论文：Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, C., Zhu, X., & Gai, K. (2019). Deep Interest Evolution Network
for Click-Through Rate Prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 5941-5948.

论文下载地址：https://doi.org/10.1609/aaai.v33i01.33015941

# 模型架构

1. 行为序列层（Behavior Layer)：主要将用户浏览过的商品转换为对应的embedding，并且按照浏览时间做排序，即把原始的id类行为序列特征转换成Embedding行为序列
2. 兴趣抽取层（Interest Extractor Layer）：主要是通过模拟用户的兴趣迁移过程，基于行为序列提取用户兴趣序列
3. 兴趣进化层（Interest Evolving Layer）：主要是通过在兴趣抽取层基础上加入Attention机制，模拟与当前目标广告相关的兴趣进化过程，对与目标物品相关的兴趣演化过程进行建模
4. 最后将兴趣表示、Target Ad、user profile、context feature的embedding向量进行拼接，最后使用MLP完成预测

# 数据集

利用amazon的数据集来训练DIEN网络，论文中使用的amazon数据集有两种，一个是books数据集，一个是electronics数据集。

数据集下载和读取：`bash prepare_books.sh`  和  `bash prepare_electronics.sh`

之后books和electronics数据集都会分成七个文件：

1. cat_voc.pkl：用户字典，用户名和其对应的id
2. mid_voc.pkl：movie字典，item和其对应的id
3. uid_voc.pkl：movie种类字典，category和其对应的id
4. local_train_splitByUser：训练数据，一行格式为：label、用户名、目标item、目标item类别、历史item、历史item的类别
5. local_test_splitByUser：测试数据，格式同训练数据
6. reviews-info：review元数据，格式为：userID、itemID、评分、时间戳、用于进行负采样的数据
7. item-info：item对应的category信息

# 环境要求

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练、评估、推理和解释：

- 克隆代码
- 下载原始Amazon数据集即Books与Electronics数据集，预处理原始数据,请勿同时运行预处理
  (备注，Books数据集在官方tf的代码中已处理完成，建议直接下载tf中的数据集，[链接](https://github.com/mouna99/dien),
  分为三个文件，分别为data，data1，data2，解压至Books文件夹中即可)

```bash
bash scripts/prepare_books.sh
bash scripts/prepare_electronics.sh
```

- 生成mindrecord格式训练数据集与测试数据集

```bash
# 生成Books训练数据集与测试数据集
python create_mindrecord.py --device_target=GPU --device_id=1 --mindrecord_path=./dataset_mindrecord --dataset_file_path=./Books --dataset_type=Books

# 生成Electronics训练数据集与测试数据集
python create_mindrecord.py --device_target=GPU --device_id=1 --mindrecord_path=./dataset_mindrecord --dataset_file_path=./Electronics --dataset_type=Electronics
```

- 单卡训练

Books数据集训练

```bash
python train.py \
    --mindrecord_path=./dataset_mindrecord \
    --dataset_type=Books \
    --dataset_file_path=./Books \
    --device_id=0 \
    --device_target=Ascend
```

OR

```bash
bash scripts/run_standalone_train.sh 0 GPU ./dataset_mindrecord ./Books Books
```

Electronics数据集训练

```bash
python train.py \
    --mindrecord_path=./dataset_mindrecord \
    --dataset_type=Electronics \
    --dataset_file_path=./Electronics \
    --device_id=0 \
    --device_target=Ascend
```

OR

```bash
bash scripts/run_standalone_train.sh 0 GPU ./dataset_mindrecord ./Electronics Electronics
```

- 分布式训练 GPU

Books数据集线下多卡训练

```bash
bash scripts/run_distribute_train_gpu.sh 8 ./dataset_mindrecord ./Books Books
```

Electronics数据集线下多卡训练

```bash
bash scripts/run_distribute_train_gpu.sh 8 ./dataset_mindrecord ./Electronics Electronics
```

- 分布式训练 Ascend

books数据集线下多卡训练

```bash
bash run_distribute_train_book.sh
```

electronics数据集线下多卡训练

```bash
bash run_distribute_train_elec.sh
```

- modelarts上训练

Books数据集训练

```bash
python train.py \
    --mindrecord_path=./dataset_mindrecord \
    --dataset_type=Books \
    --dataset_file_path=./Books \
    --is_modelarts=True \
    --run_distribute=True
```

Electronics数据集训练

```bash
python train.py \
    --mindrecord_path=./dataset_mindrecord \
    --dataset_type=Electronics \
    --dataset_file_path=./Electronics \
    --is_modelarts=True \
    --run_distribute=True
```

- 推理和解释

```bash
python eval.py \
    --mindrecord_path=./dataset_mindrecord \
    --dataset_type=Electronics \
    --dataset_file_path=./Electronics \
    --device_id=0 \
    --device_target=Ascend \
    --save_checkpoint_path=./ckpt
```

OR

```bash
bash scripts/run_eval.sh 0 GPU ./dataset_mindrecord ./Electronics Electronics ./ckpt
```

- 导出MindIR模型

```bash
python export.py \
--dataset_type=Electronics \
--save_checkpoint_path=./result/distribute_electronics/Electronics_DIEN2.ckpt \
--device_id=0
```

- 在Ascend910上生成二进制文件，用来在Ascend310上做推理

```bash
python preprocess.py \
--mindrecord_path=./dataset_mindrecord \
--dataset_type=Electronics \
--binary_files_path=./ascend310_data \
--device_id=0
```

- Ascend310上执行推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

```bash
bash run_infer_elec_310.sh [MINDIR_PATH] [INPUT_PATH] [DEVICE_ID]

如：
bash run_infer_elec_310.sh ../model/DIEN_Electronics.mindir ../ascend310_data/Electronics_data
```

- Ascend310精度计算的结果

```bash
acc:0.7252604166666666  test_auc:0.7918927228009263
spend_time: 0.002615213394165039
```

# 脚本说明

## 脚本和样例代码

```dien
.
└─DIEN
  ├─README_CN.md
  ├─ascend310_infer
    ├─inc
      ├─utils.h
    ├─src
        ├─main.cc
      ├─utils.cc
    ├─build.sh
    ├─CMakeLists.txt
  ├─scripts
    ├─prepare_books.sh                    # books数据集下载和切分脚本
    ├─prepare_electronics.sh              # electronics数据集下载和切分脚本
    ├─run_distribute_train_elec.sh        # 在Ascend上electronics数据集8卡训练脚本
    ├─run_distribute_train_books.sh       # 在Ascend上books数据集8卡训练脚本
    ├─run_distribute_train_gpu.sh         # 在GPU上进行分布式训练
    ├─run_eval.sh                         # 在GPU或Ascend上进行单卡验证
    ├─run_infer_book_310.sh               # books数据集310推理脚本
    ├─run_infer_elec_310.sh               # electronics数据集310推理脚本
    ├─run_standalone_train.sh             # 在GPU或Ascend上进行单卡训练
  ├─src
    ├─__init__.py                         # init文件
    ├─dataset.py                          # 数据集文件
    ├─generate_voc.py                     # 生成voc文件
    ├─gru.py                              # DIEN模型基础结构：GRU
    ├─local_aggretor.py                   # train，test文件处理
    ├─dien.py                             # DIEN模型整体
    ├─process_data.py                     # 原始数据集预处理
    ├─shuffle.py                          # 乱序
    ├─split_by_user.py                    # 分割训练，测试数据集
    ├─utils.py                            # utils ，用于计算auc
  ├─create_mindrecord.py                  # 创建mindrecord数据集
  ├─eval.py                               # 验证网络
  ├─export.py                             # export网络
  ├─preprocess.py                         # 310推理的数据预处理文件
  ├─postprocess.py                        # 310推理的精度计算
  ├─train.py                              # 训练网络
```

## 脚本参数

- train.py参数

```text
--data_url                   dataset path in the obs bucket
--train_url                  training output path in the obs bucket
--mindrecord_path            mindrecord format dataset path
--dataset_type               dataset type(Books/Electronics)
--dataset_file_path          dataset files path
--pretrained_ckpt_path       pretrained ckpt path in the obs bucket
--device_target              device target
--device_id                  device id
--is_modelarts               whether to run in the modelarts
--run_distribute             whether to run distribute training
```

- eval.py参数

```text
--mindrecord_path            mindrecord format dataset path
--dataset_type               dataset type(Books/Electronics)
--dataset_file_path          dataset files path
--device_target              device target
--device_id                  device id
--save_checkpoint_path       checkpoint directory or checkpoint file path
```

- preprocess.py参数

```text
--mindrecord_path     mindrecord format dataset path
--dataset_type        dataset type(Books/Electronics)
--binary_files_path   the generated binary files path
--device_target       device target
--device_id           device id
```

- postprocess.py参数

```text
--binary_files_path                 the generated binary files path
--target_binary_files_path          the target binary files path
--dataset_type                      dataset type(Books/Electronics)
```

# 模型描述

## 性能

### 训练性能

| 参数           | Ascend单机/八卡                                              | GPU单机/八卡                                |
| -------------- | ------------------------------------------------------------ | ------------------------------------------ |
| 资源           | Ascend 910；系统 Euler2.8                                    | 英伟达 Tesla V100;系统 Ubuntu 18.04              |
| 上传日期       | 2021-11-15                                                   | 2021-11-22                                 |
| MindSpore版本  | 1.6.0                                                        | 1.6.0                                      |
| 数据集         | 2                                                            | 2                                          |
| 训练参数       | Epoch=3, batch_size=128                                      | Epoch=3, batch_size=128                    |
| 优化器         | Adam                                                         | Adam                                       |
| 损失函数       | Sigmoid交叉熵                                                | Sigmoid交叉熵                              |
| 推理检查点     | books是63.03MB，elec是16.64MB（.ckpt文件）                   | books是63MB，elec是16.6MB（.ckpt文件） |
| 单step训练耗时 | books单机是27s，books八卡是23s；elec单机是6.7s，elec八卡6.7s | books单机是250ms，books八卡是280ms；elec单机是190ms，elec八卡240s |
| 训练总耗时     | books单机是189h45m45s，books八卡是20h17m54s；elec单机是15h16m46s，elec八卡是1h49m32s | books单机是1h45m，books八卡是15ms；elec单机是27m，elec八卡是3m |
| 最终收敛的loss | books单机是0.062，books八卡是0.088；elec单机是0.098，elec八卡是0.191 | books单机是0.141，books八卡是0.160；elec单机是0.142，elec八卡是0.195 |

### 推理和解释性能

| 参数          | DIEN                                       | DIEN                                       |
| ------------- | ------------------------------------------| ------------------------------------------|
| 资源          | Ascend 910；系统 Euler2.8                  | 英伟达 Tesla V100;系统 Ubuntu 18.04      |
| 上传日期      | 2021-11-15                                 | 2021-11-22                                 |
| MindSpore版本 | 1.6.0                                      | 1.6.0                                      |
| 数据集        | 2                                          | 2                                          |
| 批次大小      | 128                                        | 128                                        |
| 输出          | AUC                                        | AUC                                        |
| 准确率        | books是AUC=0.8464，electronics是AUC=0.7787 |books是AUC=0.8510，electronics是AUC=7762 |

# 随机情况说明

- 数据集的打乱
- 模型权重的随机初始化

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)。