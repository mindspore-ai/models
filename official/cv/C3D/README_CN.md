# 目录

- [目录](#目录)
    - [C3D描述](#c3d描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
            - [在Ascend上训练](#在ascend上训练)
        - [分布式训练](#分布式训练)
            - [Ascend分布式训练](#ascend分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
            - [在Ascend上评估](#在ascend上评估)
    - [推理过程](#推理过程)
        - [导出MindIR](#导出mindir)
        - [在Ascend 310上推理](#在ascend310上推理)
        - [结果](#结果)
    - [模型说明](#模型说明)
        - [性能](#性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)
    - [ModelZoo主页](#modelzoo主页)

## [C3D描述](#目录)

C3D模型广泛应用于3D视觉任务。C3D网络结构与普通的2D ConvNet相似，主要区别在于C3D使用了Conv3D等3D操作，而2D ConvNet则完全基于2D架构。有关C3D网络的详细信息，请参阅论文《Learning Spatiotemporal Features with 3D Convolutional Networks》。

## [模型架构](#目录)

C3D网络有8个卷积层、5个max池化层和2个全连接层，以及1个softmax输出层。所有3D卷积核在空间和时间维度上都是3×3×3，步长为1。3D池化层用pool1到pool5表示。池化内核为2x2x2，除了pool1是1x2x2。每个全连接层有4096个输出单元。

## [数据集](#目录)

使用的数据集：[HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

- 描述：HMDB51数据来源广泛，大部分来自电影，小部分来自Prelinger档案、YouTube和谷歌视频等公共数据库。数据集包含6849个短片，分为51类动作，每类至少包含101个视频。

- 数据集大小：6849个视频
    - 注：使用官方的训练/测试分组（[test_train_splits](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar))。
- 数据格式：RAR
    - 注：数据将在dataset_preprocess.py中处理。
- 数据目录结构

    ```text
    .
    └─hmdb-51                                // 包含51个文件夹
    ├── brush_hair                        // 包含107个视频
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi      // 视频文件
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1.avi      // 视频文件
    │    ...
    ├── cartwheel                         // 包含107个图像文件
    │   ├── (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi       // 视频文件
    │   ├── Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8.avi   // 视频文件
    │    ...
    ...
    ```

    使用的数据集：[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

- 描述：UCF101动作识别数据集包含从YouTube收集的现实动作视频，有101个动作类。此数据集基于UCF50数据集的50个动作类进行扩展。

- 数据集大小：13320个视频
    - 注：使用官方的训练/测试分组（[UCF101TrainTestSplits](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip))。
- 数据格式：RAR
    - 注：数据将在dataset_preprocess.py中处理
- 数据目录结构

    ```text
    .
    └─ucf101                                     // 包含101文件夹
    ├── ApplyEyeMakeup                        // 包含145个视频
    │   ├── v_ApplyEyeMakeup_g01_c01.avi     // 视频文件
    │   ├── v_ApplyEyeMakeup_g01_c02.avi     // 视频文件
    │    ...
    ├── ApplyLipstick                         // 包含114个图像文件
    │   ├── v_ApplyLipstick_g01_c01.avi      // 视频文件
    │   ├── v_ApplyLipstick_g01_c02.avi      // 视频文件
    │    ...
    ...
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

- 处理原始数据文件

    ```text
    # 将视频转换为图像。
    bash run_dataset_preprocess.sh HMDB51 [RAR_FILE_PATH] 1
    # 或
    bash run_dataset_preprocess.sh UCF101 [RAR_FILE_PATH] 1
    # 例如：bash run_dataset_preprocess.sh HMDB51 /Data/hmdb51/hmdb51_org.rar 1
    ```

- 在YAML文件中配置数据集名称和数据集的分类数

    ```text
    dataset:         [DATASET_NAME]  # 数据集的名称，'HMDB51'或'UCF101'
    num_classes:     [CLASS_NUMBER]  # 数据集的分类数，HMDB51为51，UCF101为101
    # 例如：
    # dataset:       HMDB51
    # num_classes:   51
    # 或
    # dataset:       UCF101
    # num_classes:   101
    ```

- 在YAML文件中配置处理后的数据集的路径

    ```text
    json_path:   [JSON_FILE_PATH]        # 给定数据集的json文件的路径
    img_path:    [IMAGE_FILE_PATH]       # 给定数据集的图像文件的路径
    # 例如：
    # json_path: /Data/hmdb-51_json/
    # img_path: /Data/hmdb-51_img/
    ```

- 下载预训练模型

    ```text
    mkdir ./pretrained_model
    # 下载C3D预训练文件
    wget -O ./pretrained_model/c3d-pretrained.pth https://download.mindspore.cn/thirdparty/c3d_pretrained.pth
    # 下载C3D均值文件
    wget -O ./pretrained_model/sport1m_train16_128_mean.npy https://download.mindspore.cn/thirdparty/sport1m_train16_128_mean.npy
    ```

- 将PyTorch预训练模型文件转换为MindSpore文件（必须安装Pytorch和MindSpore。）

    ```text
    # 将PyTorch预训练模型文件转换为MindSpore文件。
    bash run_ckpt_convert.sh [PYTORCH_FILE_PATH] [MINDSPORE_FILE_PATH]
    # 例如：bash run_ckpt_convert.sh /home/usr/c3d/pretrain_model/c3d_pretrained.pth /home/usr/c3d/pretrain_model/c3d_pretrained_ms.ckpt
    ```

    参阅`default_config.yaml`，支持快速入门的参数配置。

- 在Ascend上运行

    ```bash
    cd scripts
    # 运行训练示例
    bash run_standalone_train_ascend.sh
    # 运行分布式训练示例
    bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]
    # 运行评估示例
    bash run_standalone_eval_ascend.sh [CKPT_FILE_PATH]
    ```

- 在GPU上运行

    ```bash
    cd scripts
    # 运行训练示例
    bash run_standalone_train_gpu.sh [CONFIG_PATH] [DEVICE_ID]
    # 运行分布式训练示例
    bash run_distribute_train_gpu.sh [CONFIG_PATH]
    # 运行评估示例
    bash run_standalone_eval_gpu.sh [CKPT_FILE_PATH] [CONFIG_PATH]
    ```

## [脚本说明](#目录)

### [脚本及样例代码](#目录)

```text
.
└─c3d
  ├── README.md                           // 关于C3D的说明
  ├── scripts
  │   ├──run_dataset_preprocess.sh       // 预处理数据集的shell脚本
  │   ├──run_ckpt_convert.sh             // 在GPU上将PyTorch CKPT文件转换为pickle文件的shell脚本
  │   ├──run_distribute_train_ascend.sh  // Ascend分布式训练shell脚本
  │   ├──run_distribute_train_gpu.sh  // GPU分布式训练shell脚本
  │   ├──run_infer_310.sh                // Ascend 310上用于推理的shell脚本
  │   ├──run_standalone_train_ascend.sh  // Ascend训练shell脚本
  │   ├──run_standalone_train_gpu.sh  // GPU训练shell脚本
  │   ├──run_standalone_eval_ascend.sh   // Ascend测试shell脚本
  │   ├──run_standalone_eval_gpu.sh   // GPU测试shell脚本
  ├── src
  │   ├──dataset.py                    // 创建数据集
  │   ├──evalcallback.py               // eval回调
  │   ├──lr_schedule.py                // 学习率调度器
  │   ├──transform.py                  // 处理数据集
  │   ├──loss.py                       // 损失
  │   ├──utils.py                      // 通用组件（回调函数）
  │   ├──c3d_model.py                  // Unet3D模型
          ├── model_utils
          │   ├──config.py                    // 参数配置
          │   ├──device_adapter.py            // 设备适配器
          │   ├──local_adapter.py             // 本地适配器
          │   ├──moxing_adapter.py            // moxing适配器
          ├── tools
          │   ├──ckpt_convert.py          // 将PyTorch CKPT文件转换为pickle文件
          │   ├──dataset_preprocess.py    // 预处理数据集
  ├── default_config.yaml               // 参数配置
  ├── default_config_gpu.yaml           // GPU参数配置
  ├── requirements.txt                  // 需求配置
  ├── export.py                         // 将MindSpore CKPT文件转换为MINDIR文件
  ├── train.py                          // 评估脚本
  ├── eval.py                           // 训练脚本
```

### [脚本参数](#目录)

可以在default_config.yaml中设置训练和评估的参数。

- 配置C3D、HMDB51数据集

    ```text
    # ==============================================================================
    # 设备
    device_target:     "Ascend"

    # 数据集设置
    clip_length:       16                    # 短片帧数
    clip_offset:       0                     # 视频和短片开头的帧偏移（仅限第一个短片）
    clip_stride:       1                     # 连续短片之间的帧偏移，必须为>=1
    crop_shape:        [112,112]             # 帧的(高度,宽度)
    crop_type:         Random                # 裁剪类型(Random、Central、None)
    final_shape:       [112,112]             # 提供给CNN的输入(高度,宽度)
    num_clips:         -1                    # 从视频中截取的短片数量（<0：均匀采样，0：整个视频截成短片，>0：定义短片数量）
    random_offset:     0                     # 是否生成用于从视频中截取短片的长度
    resize_shape:      [128,171]             # 调整原始数据的大小为(高度,宽度)
    subtract_mean:     ''                    # 预处理期间从所有帧中减去均值(R,G,B)
    proprocess:        default               # 用于选择预处理类型的字符串参数

    # 实验设置
    model:             C3D                   # 待加载的模型名称
    acc_metric:        Accuracy              # 准确率指标
    batch_size:        16                    # mini-batch中的视频数量
    dataset:           HMDB51                # 数据集名称，'HMDB51'或'UCF101'
    epoch:             30                    # epoch总数
    gamma:             0.1                   # 更改学习率的乘数
    json_path:         ./JSON/hmdb51/        # 给定数据集的json文件路径
    img_path:          ./Data/hmdb-51_img/   # 给定数据集的img文件的路径
    sport1m_mean_file_path: ./pretrained_model/sport1m_train16_128_mean.npy # Sport1m数据分布信息
    num_classes:       51                    # 数据集的分类数，HMDB51为51，UCF101为101
    loss_type:         M_XENTROPY            # 损失函数
    lr:                1e-4                  # 学习率
    milestones:        [10, 20]              # 更改学习率的epoch值
    momentum:          0.9                   # 优化器中的动量值
    loss_scale:        1.0
    num_workers:       8                     # 用于加载数据的CPU worker数
    opt:               sgd                   # 优化器名称
    pre_trained:       1                     # 加载预训练网络
    save_dir:          ./results             # results目录的路径
    seed:              666                   # 复现种子
    weight_decay:      0.0005                # 权重衰减

    # 训练设置
    is_distributed:    0                     # 是否分布式训练
    is_save_on_master: 1                     # 仅在master设备上保存CKPT
    device_id:         0                     # 设备ID
    ckpt_path:         ./result/ckpt         # CKPT保存路径
    ckpt_interval:     1                     # 保存CKPT间隔
    keep_checkpoint_max: 40                  # 最大CKPT文件数量

    # ModelArts设置
    enable_modelarts:  0                     # 在ModelArts上训练
    result_url:        ''                    # 结果保存路径
    data_url:          ''                    # 数据路径
    train_url:         ''                    # 训练路径

    # 导出设置
    ckpt_file:         './C3D.ckpt'          # MindSporeckpt文件路径
    mindir_file_name:  './C3D'               # 保存文件路径
    file_format:       'mindir'              # 保存文件格式
    # ==============================================================================

    ```

    可以在default_config_gpu.yaml中设置训练和评估的参数。

- 配置C3D、UCF101数据集

    ```text
    # ==============================================================================
    # 设备
    device_target:     "GPU"

    # 数据集设置
    clip_length:       16                    # 短片帧数
    clip_offset:       0                     # 视频和短片开头的帧偏移（仅限第一个短片）
    clip_stride:       1                     # 连续短片之间的帧偏移，必须为>=1
    crop_shape:        [112,112]             # 帧的(高度,宽度)
    crop_type:         Random                # 裁剪类型(Random、Central、None)
    final_shape:       [112,112]             # 提供给CNN的输入(高度,宽度)
    num_clips:         -1                    # 从视频中截取的短片数量（<0：均匀采样，0：整个视频截成短片，>0：定义短片数量）
    random_offset:     0                     # 是否生成用于从视频中截取短片的长度
    resize_shape:      [128,171]             # 调整原始数据的大小为(高度,宽度)
    subtract_mean:     ''                    # 预处理期间从所有帧中减去均值(R,G,B)

    # 实验设置
    model:             C3D                   # 待加载的模型名称
    acc_metric:        Accuracy              # 准确率指标
    batch_size:        8                    # mini-batch中的视频数量
    dataset:           UCF101                # 数据集名称
    epoch:             150                   # epoch总数
    gamma:             0.1                   # 更改学习率的乘数
    json_path:         ./ucf101/JSON         # 给定数据集的json文件路径
    img_path:          ./ucf101/UCF-101_img  # 给定数据集的img文件的路径
    num_classes:       101                   # 数据集的分类数
    loss_type:         M_XENTROPY            # 损失函数
    lr:                0.00025                  # 学习率
    milestones:        [15, 30, 75]              # 更改学习率的epoch值
    momentum:          0.9                   # 优化器中的动量值
    loss_scale:        1.0
    num_workers:       4                     # 用于加载数据的CPU worker数
    opt:               sgd                   # 优化器名称
    pre_trained:       1                     # 加载预训练网络
    save_dir:          ./results             # results目录的路径
    seed:              666                   # 可重复性的种子
    weight_decay:      0.0005                # 权重衰减
    sport1m_mean_file_path: ./pretrained_model/sport1m_train16_128_mean.npy   # sports1m预训练模型的路径
    pre_trained_ckpt_path: ./pretrained_model/c3d-pretrained.ckpt   # c3d预训练模型的路径

    # 训练设置
    is_distributed:    0                     # 是否分布式训练
    is_evalcallback:   1                     # 是否使用eval callback
    is_save_on_master: 1                     # 仅在master设备上保存CKPT
    device_id:         0                     # 设备ID
    ckpt_path:         ./result/ckpt         # CKPT保存路径
    ckpt_interval:     1                     # 保存CKPT间隔
    keep_checkpoint_max: 3                  # 最大CKPT文件数量

    # ModelArts设置
    enable_modelarts:  0                     # 在ModelArts上训练
    result_url:        ''                    # 结果保存路径
    data_url:          ''                    # 数据路径
    train_url:         ''                    # 训练路径

    # 导出设置
    ckpt_file:         './C3D.ckpt'          # MindSporeckpt文件路径
    mindir_file_name:  './C3D'               # 保存文件路径
    file_format:       'mindir'              # 保存文件格式
    # ==============================================================================

    ```

## [训练过程](#目录)

### 训练

#### 在Ascend上训练

```text
# 输入脚本目录
cd scripts
# 训练
bash run_standalone_train_ascend.sh
```

上述python命令将在后台运行，您可以通过`eval.log`文件查看结果。

训练结束后，可在默认脚本文件夹下找到检查点文件。得到如下损失值：

- HMDB51的train.log

    ```shell
    epoch: 1 step: 223, loss is 2.8705792
    epoch time: 74139.530 ms, per step time: 332.464 ms
    epoch: 2 step: 223, loss is 1.8403366
    epoch time: 60084.907 ms, per step time: 269.439 ms
    epoch: 3 step: 223, loss is 1.4866445
    epoch time: 61095.684 ms, per step time: 273.972 ms
    ...
    epoch: 29 step: 223, loss is 0.3037338
    epoch time: 60436.915 ms, per step time: 271.018 ms
    epoch: 30 step: 223, loss is 0.2176594
    epoch time: 60130.695 ms, per step time: 269.644 ms
    ```

- UCF101的train.log

    ```shell
    epoch: 1 step: 596, loss is 0.53118783
    epoch time: 170693.634 ms, per step time: 286.399 ms
    epoch: 2 step: 596, loss is 0.51934457
    epoch time: 150388.783 ms, per step time: 252.330 ms
    epoch: 3 step: 596, loss is 0.07241724
    epoch time: 151548.857 ms, per step time: 254.277 ms
    ...
    epoch: 29 step: 596, loss is 0.034661677
    epoch time: 150932.542 ms, per step time: 253.243 ms
    epoch: 30 step: 596, loss is 0.0048465515
    epoch time: 150760.797 ms, per step time: 252.954 ms
    ```

#### 在GPU上训练

> 注：若出现以下错误信息：
> “Bad performance attention, it takes more than 25 seconds to fetch and send a batch of data into device, which might result `GetNext` timeout problem.“
> 请将参数“dataset_sink_mode”改为False。

```text
# 输入脚本目录
cd scripts
# 训练
bash run_standalone_train_gpu.sh [CONFIG_PATH] [DEVICE_ID]
```

上述shell脚本将在后台运行分布式训练。可通过`./Train[X].log`文件查看结果。得到如下损失值：

- HMDB51的train.log

    ```shell
    epoch: 1 step: 446, loss is 2.5606058
    epoch time: 233965.570 ms, per step time: 298.655 ms
    epoch: 2 step: 446, loss is 2.526373
    epoch time: 217318.708 ms, per step time: 251.523 ms
    epoch: 3 step: 446, loss is 1.2750342
    epoch time: 218027.155 ms, per step time: 232.700 ms
    ...
    epoch: 29 step: 446, loss is 0.5408877
    epoch time: 217762.113 ms, per step time: 220.512 ms
    epoch: 30 step: 446, loss is 0.52792186
    epoch time: 218174.035 ms, per step time: 228.359 ms
    ```

- UCF101的train.log

    ```shell
    epoch: 1 step: 1192, loss is 0.8381556
    epoch time: 593197.024 ms, per step time: 301.297 ms
    epoch: 2 step: 1192, loss is 0.5701107
    epoch time: 576058.976 ms, per step time: 260.542 ms
    epoch: 3 step: 1192, loss is 0.1724325
    epoch time: 578041.281 ms, per step time: 235.868 ms
    ...
    epoch: 99 step: 1192, loss is 6.3519354e-05
    epoch time: 573493.252 ms, per step time: 225.237 ms
    epoch: 100 step: 1192, loss is 4.852382e-05
    epoch time: 575237.743 ms, per step time: 229.164 ms
    ```

### 分布式训练

#### Ascend分布式训练

> 注：
> RANK_TABLE_FILE文件，请参考[链接](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html)。如需获取设备IP，请点击[链接](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。对于InceptionV4等大模型，最好导出外部环境变量`export HCCL_CONNECT_TIMEOUT=600`，将hccl连接检查时间从默认的120秒延长到600秒。否则，连接可能会超时，因为随着模型增大，编译时间也会增加。
>

```text
# 输入脚本目录
cd scripts
# 分布式训练
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE]
```

上述shell脚本将在后台运行分布式训练。可通过`./train[X].log`文件查看结果。得到如下损失值：

- HMDB51的train0.log

    ```shell
    epoch: 1 step: 223, loss is 3.1376421
    epoch time: 103888.471 ms, per step time: 465.868 ms
    epoch: 2 step: 223, loss is 1.4189289
    epoch time: 13111.264 ms, per step time: 58.795 ms
    epoch: 3 step: 223, loss is 3.3972843
    epoch time: 13140.217 ms, per step time: 58.925 ms
    ...
    epoch: 29 step: 223, loss is 0.099318035
    epoch time: 13094.187 ms, per step time: 58.718 ms
    epoch: 30 step: 223, loss is 0.00515177
    epoch time: 13101.518 ms, per step time: 58.751 ms
    ```

- UCF101的train0.log

    ```shell
    epoch: 1 step: 596, loss is 0.51830626
    epoch time: 82401.300 ms, per step time: 138.257 ms
    epoch: 2 step: 596, loss is 0.5527372
    epoch time: 30820.129 ms, per step time: 51.712 ms
    epoch: 3 step: 596, loss is 0.007791209
    epoch time: 30809.803 ms, per step time: 51.694 ms
    ...
    epoch: 29 step: 596, loss is 7.510604e-05
    epoch time: 30809.334 ms, per step time: 51.694 ms
    epoch: 30 step: 596, loss is 0.13138217
    epoch time: 30819.966 ms, per step time: 51.711 ms
    ```

#### GPU分布式训练

```text
# 输入脚本目录
cd scripts
# 分布式训练
vim run_distribute_train_gpu.sh to set start_device_id
bash run_distribute_train_gpu.sh [CONFIG_PATH]
```

- HMDB51的train_distributed.log

    ```shell
    epoch: 1 step: 55, loss is 3.084101915359497
    epoch: 1 step: 55, loss is 3.001408100128174
    epoch: 1 step: 55, loss is 2.5211687088012695
    epoch: 1 step: 55, loss is 2.9400177001953125
    epoch: 1 step: 55, loss is 3.598146677017212
    epoch: 1 step: 55, loss is 2.37894344329834
    epoch: 1 step: 55, loss is 1.7159693241119385
    epoch: 1 step: 55, loss is 2.2993032932281494
    epoch time: 24400.126 ms, per step time: 443.639 ms
    epoch time: 25218.947 ms, per step time: 458.526 ms
    epoch time: 24354.300 ms, per step time: 442.805 ms
    epoch time: 25144.561 ms, per step time: 457.174 ms
    epoch time: 21726.022 ms, per step time: 395.019 ms
    epoch time: 24929.812 ms, per step time: 453.269 ms
    epoch time: 25210.959 ms, per step time: 458.381 ms
    epoch time: 24913.004 ms, per step time: 452.964 ms
    epoch: 2 step: 55, loss is 2.177777051925659
    epoch: 2 step: 55, loss is 2.5466907024383545
    epoch: 2 step: 55, loss is 1.9837493896484375
    epoch: 2 step: 55, loss is 1.4902374744415283
    epoch: 2 step: 55, loss is 2.3385510444641113
    epoch: 2 step: 55, loss is 1.3293176889419556
    epoch: 2 step: 55, loss is 1.8483796119689941
    epoch: 2 step: 55, loss is 0.8703243136405945
    epoch time: 23577.773 ms, per step time: 428.687 ms
    epoch time: 23325.131 ms, per step time: 424.093 ms
    epoch time: 23254.383 ms, per step time: 422.807 ms
    epoch time: 22038.587 ms, per step time: 400.702 ms
    epoch time: 21795.474 ms, per step time: 396.281 ms
    epoch time: 22424.821 ms, per step time: 407.724 ms
    epoch time: 23331.409 ms, per step time: 424.207 ms
    epoch time: 22178.912 ms, per step time: 403.253 ms
    ...
    ```

- UCF101的train_distributed.log

    ```shell
    epoch: 1 step: 149, loss is 0.97137051820755
    epoch: 1 step: 149, loss is 1.1462825536727905
    epoch: 1 step: 149, loss is 1.484191656112671
    epoch: 1 step: 149, loss is 0.639738142490387
    epoch: 1 step: 149, loss is 1.1133722066879272
    epoch: 1 step: 149, loss is 1.5043989419937134
    epoch: 1 step: 149, loss is 1.2063453197479248
    epoch: 1 step: 149, loss is 1.3174564838409424
    epoch time: 183002.444 ms, per step time: 1228.204 ms
    epoch time: 183388.214 ms, per step time: 1230.793 ms
    epoch time: 183560.571 ms, per step time: 1231.950 ms
    epoch time: 183881.357 ms, per step time: 1234.103 ms
    epoch time: 184225.004 ms, per step time: 1236.409 ms
    epoch time: 184383.710 ms, per step time: 1237.475 ms
    epoch time: 184501.011 ms, per step time: 1238.262 ms
    epoch time: 184885.520 ms, per step time: 1240.842 ms
    epoch: 2 step: 149, loss is 0.10039880871772766
    epoch: 2 step: 149, loss is 0.5981963276863098
    epoch: 2 step: 149, loss is 0.4604840576648712
    epoch: 2 step: 149, loss is 0.215419739484787
    epoch: 2 step: 149, loss is 0.2556331753730774
    epoch: 2 step: 149, loss is 0.03653889149427414
    epoch: 2 step: 149, loss is 1.4467300176620483
    epoch: 2 step: 149, loss is 1.0422033071517944
    epoch time: 53143.686 ms, per step time: 356.669 ms
    epoch time: 52175.739 ms, per step time: 350.173 ms
    epoch time: 54300.036 ms, per step time: 364.430 ms
    epoch time: 53026.808 ms, per step time: 355.885 ms
    epoch time: 52941.203 ms, per step time: 355.310 ms
    epoch time: 53144.090 ms, per step time: 356.672 ms
    epoch time: 53896.009 ms, per step time: 361.718 ms
    epoch time: 53584.895 ms, per step time: 359.630 ms
    ...
    ```

## [评估过程](#目录)

### 评估

#### 在Ascend上评估

- Ascend上运行数据集评估

    在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如，"username/ckpt_0/c3d-hmdb51-0-30_223.ckpt"。

    ```text
    # 输入脚本目录
    cd scripts
    # eval
    bash run_standalone_eval_ascend.sh [CKPT_FILE_PATH]
    ```

    上述python命令将在后台运行。可通过"eval.log"文件查看结果。测试数据集的准确率如下：

- HMDB51的eval.log

    ```text
    start create network
    pre_trained model: username/ckpt_0/c3d-hmdb51-0-30_223.ckpt
    setep: 1/96, acc: 0.3125
    setep: 21/96, acc: 0.375
    setep: 41/96, acc: 0.9375
    setep: 61/96, acc: 0.6875
    setep: 81/96, acc: 0.4375
    eval result: top_1 50.196%
    ```

- UCF101的eval.log

    ```text
    start create network
    pre_trained model: username/ckpt_0/c3d-ucf101-0-30_596.ckpt
    setep: 1/237, acc: 0.625
    setep: 21/237, acc: 1.0
    setep: 41/237, acc: 0.5625
    setep: 61/237, acc: 1.0
    setep: 81/237, acc: 0.6875
    setep: 101/237, acc: 1.0
    setep: 121/237, acc: 0.5625
    setep: 141/237, acc: 0.5
    setep: 161/237, acc: 1.0
    setep: 181/237, acc: 1.0
    setep: 201/237, acc: 0.75
    setep: 221/237, acc: 1.0
    eval result: top_1 79.381%
    ```

#### 在GPU上评估

- GPU上运行数据集评估

    在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如，"./results/xxxx-xx-xx_time_xx_xx_xx/ckpt_0/0-30_223.ckpt"。

    ```text
    # 输入脚本目录
    cd scripts
    # eval
    bash run_standalone_eval_gpu.sh [CKPT_FILE_PATH] [CONFIG_PATH]
    ```

- HMDB51的eval.log

    ```text
    start create network
    pre_trained model:./results/2021-10-21_time_11_16_48/ckpt_0/0-30_223.ckpt
    setep: 1/96, acc: 0.6875
    setep: 21/96, acc: 0.6875
    setep: 41/96, acc: 0.4375
    setep: 61/96, acc: 0.3125
    setep: 81/96, acc: 0.75
    eval result: top_1 50.327%
    ```

- UCF101的eval.log

    ```text
    start create network
    pre_trained model: ./results/2021-11-02_time_07_30_42/ckpt_0/0-85_223.ckpt
    setep: 1/237, acc: 0.75
    setep: 21/237, acc: 1.0
    setep: 41/237, acc: 0.625
    setep: 61/237, acc: 1.0
    setep: 81/237, acc: 0.875
    setep: 101/237, acc: 1.0
    setep: 121/237, acc: 0.9375
    setep: 141/237, acc: 0.5625
    setep: 161/237, acc: 1.0
    setep: 181/237, acc: 1.0
    setep: 201/237, acc: 0.5625
    setep: 221/237, acc: 1.0
    eval result: top_1 80.412%
    ```

## 推理过程

### [导出MindIR](#目录)

```shell
python export.py --ckpt_file [CKPT_PATH] --mindir_file_name [FILE_NAME] --file_format [FILE_FORMAT] --num_classes [NUM_CLASSES] --batch_size [BATCH_SIZE]
```

- `ckpt_file`：必填参数。
- `file_format`：取值范围["AIR", "MINDIR"]。
- `NUM_CLASSES`：数据集的分类数，HMDB51为51，UCF101为101。
- `BATCH_SIZE`：目前mindir不支持动态shape，此网络仅支持batch_size为1的推理。

### 在Ascend 310上推理

**推理前需参照[MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md)进行环境变量设置。**

在推理前，必须通过`export.py`脚本导出mindir文件。下面以使用MINDIR模型推理为例。

```shell
# Ascend 310推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET] [NEED_PREPROCESS] [DEVICE_ID]
```

- `DATASET`：取值为'HMDB51'或'UCF101'。
- `NEED_PREPROCESS`：表示是否需要预处理，其值为'y'或'n'。
- `DEVICE_ID`：可选参数，默认值为0。

### 结果

推理结果保存在当前路径中，您可以在acc.log文件中查看如下结果。

## [模型说明](#目录)

### [性能](#目录)

#### 评估性能

- 用于HMDB51的C3D

    | 参数         | Ascend                                                     | GPU                                                  |
    | ------------------- | ---------------------------------------------------------  |-------------------------------------------------------|
    | 模型版本      | C3D                                                       | C3D                                                  |
    | 资源           | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8| V100                                                 |
    | 上传日期      | 09/22/2021                               | 11/06/2021                          |
    | MindSpore版本  | 1.2.0                                                      | 1.5.0                                                 |
    | 数据集            | HMDB51                                                     | HMDB51                                                |
    | 训练参数| epoch = 30,  batch_size = 16                               | epoch = 150,  batch_size = 8                          |
    | 优化器          | SGD                                                        | SGD                                                   |
    | 损失函数      | Max_SoftmaxCrossEntropyWithLogits                          | Max_SoftmaxCrossEntropyWithLogits                     |
    | 速度              | 单卡：270.694 ms/step                                        | 单卡：226.299 ms/step                                    |
    | Top_1               | 单卡：49.78%                                                | 单卡：50.131%                                           |
    | 总时长         | 单卡：0.32 hours                                             | 单卡：1.3 hours                                          |
    | 参数量（M）     | 78                                                         | 78

- 用于UCF101的C3D

    | 参数         | Ascend                                                     | GPU                                                      |
    | ------------------- | ---------------------------------------------------------  |---------------------------------------------------------  |
    | 模型版本      | C3D                                                       | C3D                                                      |
    | 资源           | Ascend 910；CPU 2.60GHz, 192核；内存755G；EulerOS 2.8| V100                                                     |
    | 上传日期      | 09/22/2021                               | 11/06/2021                              |
    | MindSpore版本  | 1.2.0                                                      | 1.5.0                                                     |
    | 数据集            | UCF101                                                    | UCF101                                                   |
    | 训练参数| epoch = 30, batch_size = 16                                | epoch = 150,  batch_size = 8                              |
    | 优化器          | SGD                                                        | SGD                                                       |
    | 损失函数      | Max_SoftmaxCrossEntropyWithLogits                          | Max_SoftmaxCrossEntropyWithLogits                         |
    | 速度              | 单卡：253.372 ms/step                                        | 单卡：237.128 ms/step                                            |
    | Top_1               | 单卡：80.33%                                                | 单卡：80.138%                                               |
    | 总时长         | 单卡：1.31 hours                                            | 单卡：4 hours                                               |
    | 参数量（M）     | 78                                                         | 78

# [随机情况说明](#目录)

在default_config.yaml和default_config_gpu.yaml中设置随机种子为666。

## [ModelZoo主页](#目录)

请浏览官网[主页](https://gitee.com/mindspore/models)。
