# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [Sphereface描述](#Sphereface描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
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
            - [CASIA-WebFace上的Sphereface](#CASIA-WebFace上的Sphereface)
        - [推理性能](#推理性能)
            - [LFW上的Sphereface](#LFW上的Sphereface)
    - [使用流程](#使用流程)
        - [推理](#推理-1)
        - [继续训练预训练模型](#继续训练预训练模型)
        - [迁移学习](#迁移学习)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# Sphereface描述

Spherenet是2017年提出的使用改进的softamax在人脸识别领域进行的一项创新。他提出了angular softmax loss用来改进原来的softmax loss，在large margin softmax loss的基础上增加了||W||=1以及b=0的两个限制条件，使得预测结果仅仅取决于W和x之间的角度。他的应用场景是在开集环境下的，该论文也主要解决了这一任务，使得在特定的测度空间下，能够尽可能的满足最大类内距离小于不同类的最小类间距离，该论文使得特征更加具有可判别性。

[论文](https://arxiv.org/abs/1704.08063)：Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song."SphereFace: Deep Hypersphere Embedding for Face Recognition."*Proceedings of the IEEE conference on computer vision and pattern recognition*.2017.

# 模型架构

Sphereface论文中给出了多种网络架构，本代码仅实现了其20层的网络架构，他使用多个**步长为1与2的3x3卷积，并且使用残差结构以及PReLu**进行特征提取，并在最后一个全连层进行图像分类，通过最后一层的训练使得权重可以对图像进行特征判别。

# 数据集

使用的数据集：[CASIA-WebFace](<https://download.csdn.net/download/fire_light_/10291726>)

- 数据集大小：4G，共10575个类、494414张250x250彩色或黑白图像
    - 训练集：3.7G，共45万张图像
    - 测试集：0.3G，共4万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

使用的数据集: [LFW](<http://vis-www.cs.umass.edu/lfw/>)

- 数据集大小：180M，共5749个类、13233张彩色图像
    - 训练集：162M，共11910张图像
    - 测试集：18M，共1323张图像
- 数据格式：RGB
    - 注：数据将在src/eval.py中处理。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend、GPU处理器来搭建硬件环境。(目前CPU因PReLU算子不支持无法使用，可改用ReLU但是需要提高epoch)
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```yaml

  # 添加数据集路径,以训练CASIA-WebFace为例
  train_data_dir: "../casia_landmark.txt"
  train_img_dir : "../CASIA-WebFace/"

  # 推理前添加checkpoint路径参数(只需要给出ckpt存放的目录即可)
  ckpt_files: "../CKPTFILE"

  ```

  ```python

  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_distribute_train.sh

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval_standalone_Ascend.sh [DEVICE_ID] [CKPT_FILES]
  # example: bash run_train_Ascend.sh 0 "/data/sphereface/6000-9923.ckpt"

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU处理器环境运行

  ```python

  # 运行训练示例
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例
  bash scripts/run_distribute_train_GPU.sh 8 0,1,2,3,4,5,6,7

  # 运行评估示例
  python eval.py > eval.log 2>&1 &
  或
  bash run_eval_standalone_GPU.sh [DEVICE_ID] [CKPT_FILES]
  # example: bash run_eval_standalone_GPU.sh 0 "/data/sphereface/6000-9923.ckpt"

默认使用CASIA-WebFace数据集。您也可以将`$dataset`传入脚本，以便选择其他数据集。如需查看更多详情，请参考指定脚本。

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上使用单卡训练 CASIA-WebFace 数据集

      ```python

      # (1) 在网页上设置 "config_path='/path_to_code/sphereface_config.yaml'"
      # (2) 执行a或者b
      #       a. 在 sphereface_config.yaml 文件中设置 "enable_modelarts=True"
      #          在 sphereface_config.yaml 文件中设置 "train_url='/user_namr/Sphereface/output/'"
      #          在 sphereface_config.yaml 文件中设置 “data_url=‘/user_name/CAISA/’”
      #          在 sphereface_config.yaml 文件中设置 其他参数
      #       b. 在网页上设置 "enable_modelarts=True"
      #          在网页上设置 "train_url='/user_name/Sphereface/output/'"
      #          在网页上设置 “data_url=‘/user_name/CAISA/’”
      #          在网页上设置 其他参数
      # (3) 上传你的压缩数据集到 S3 桶上 (你也可以上传原始的数据集，但那可能会很慢。)
      # (4) 在网页上设置你的代码路径为 "/path/sphereface"
      # (5) 在网页上设置启动文件为 "train.py"
      # (6) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (7) 创建训练作业

      ```

# 脚本说明

## 脚本及样例代码

```bash

├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── sphereface
        ├── README.md                    // sphereface相关说明
        ├── ascend310_infer              // 实现310推理源代码
        ├── scripts
        │   ├──run_infer_310.sh            // 执行310推理的shell脚本
        │   ├──run_distribute_train_GPU.sh            // 分布式到GPU训练的shell脚本
        │   ├──run_eval_standalone_GPU.sh      // GPU评估的shell脚本
        │   ├──run_train_standalone_GPU.sh     // 单卡GPU训练的shell脚本
        │   ├──run_distribute_train.sh            // 分布式到Ascend训练的shell脚本
        │   ├──run_eval_standalone_Ascend.sh      // Ascend评估的shell脚本
        │   ├──run_train_standalone_Ascend.sh     // 单卡Ascend训练的shell脚本
        ├── src
        │   ├──datasets                           // 数据集处理
        │   │   ├──classfication.py               // 数据分类处理
        │   │   ├──sampler.py                     // 数据shuffle处理
        │   ├──losses                             // 损失函数
        │   │   ├──crossentropy.py                // ASoftMax损失函数
        │   ├──network                            // 网络结构
        │   │   ├──spherenet.py                   // Sphereface20层网络
        ├── train.py               // 训练脚本
        ├── eval.py               // 评估脚本
        ├── export.py            // 将checkpoint文件导出到air/mindir
        ├── sphereface_config.yaml  //配置文件

```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置Sphereface和CASIA-WebFace数据集与LFW测试集。

  ```python

    device_target: 'Ascend'         #目标设备
    net: "sphereface20a"            #网络名称
    dataset: "CASIA-WebFace"        #数据集名称
    is_distributed: 1               #是否分布式
    rank: 0                         #分类
    group_size: 1                   #数据组数
    label_smooth: 1                 #是否label smooth
    smooth_factor: 0.15             #label_smooth因子
    train_data_dir: "/data/sphereface/casia_landmark.txt"   #训练数据txt存储位置
    train_img_dir : "/data/sphereface/CASIA-WebFace/"       #数据集存放位置
    train_pretrained: ""            #pretrained的ckpt文件位置可为空
    image_size: "112, 96"           #crop到的目标图片大小
    num_classes: 10574              #训练时目标类数
    lr: 0.15                        #初始学习率
    lr_scheduler: "cosine_annealing"    #学习率模式
    eta_min: 0                      #cos学习率的最小eta
    T_max: 20                       #cos学习率的变换半周期
    max_epoch: 20                   #训练周期
    per_batch_size: 32              #batch大小
    warmup_epochs: 0                #热启动的epoch
    weight_decay: 0.0005            #weight_decay因子
    momentum: 0.9                   #momentum因子
    ckpt_interval: 10000            #ckpt存储一次的迭代数
    save_ckpt_path: "./"            #ckpt存储路径
    is_save_on_master: 1            #是否存储标志
    ckpt_files: "/data/sphereface/Sphereface/scripts/device0/"  #测试时ckpt存储位置
    datatxt_src: '/data/sphereface/lfw_landmark.txt'            #测试数据的txt文件
    pairtxt_src: '/data/sphereface/pairs.txt'                   #测试数据的配对txt文件
    datasrc: '/data/sphereface/lfw/'                            #测试数据存放位置
    device_id: 7                #设备号
    batch_size: 32              #测试时的batch

  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash

  bash scripts/run_train_standalone_Ascend.sh 3

  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash

  2021-10-25 09:21:16,137:INFO:epoch[0], iter[1774], loss:8.9384, mean_fps:0.00 imgs/sec
  2021-10-25 09:44:13,485:INFO:epoch[1], iter[3549], loss:8.737117, mean_fps:329.93 imgs/sec
  2021-10-25 10:07:25,687:INFO:epoch[2], iter[5324], loss:8.615589, mean_fps:326.40 imgs/sec
  ...

  ```

- GPU处理器环境运行

  ```bash

  bash scripts/run_train_standalone_GPU.sh 3

  ```

  上述python命令将在后台运行，您可以通过train.log文件查看结果。

  训练结束后，您可在默认脚本文件夹下找到检查点文件。采用以下方式达到损失值：

  ```bash

  2021-11-12 12:47:04,004:INFO:epoch[0], iter[1774], loss:8.9929695, mean_fps:0.00 imgs/sec
  2021-11-12 13:07:43,106:INFO:epoch[1], iter[3549], loss:8.491371, mean_fps:366.72 imgs/sec
  2021-11-12 13:28:21,060:INFO:epoch[2], iter[5324], loss:8.061005, mean_fps:367.06 imgs/sec
  2021-11-12 13:48:55,650:INFO:epoch[3], iter[7099], loss:7.3936157, mean_fps:368.06 imgs/sec
  ...

  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash

  bash scripts/run_distribute_train.sh

  ```

  上述shell脚本将在后台运行分布训练。您可以通过/scripts/device[X]/output.log文件查看结果。采用以下方式达到损失值：

  ```bash

  /scripts/device0/output.log:
  2021-10-26 17:43:54,981:INFO:epoch[0], iter[1774], loss:8.913538, mean_fps:0.00 imgs/sec
  2021-10-26 17:47:19,239:INFO:epoch[1], iter[3549], loss:8.307044, mean_fps:2224.67 imgs/sec
  2021-10-26 17:50:43,860:INFO:epoch[2], iter[5324], loss:8.096157, mean_fps:2220.70 imgs/sec
  ...
  /scripts/device1/output.log
  2021-10-26 17:43:53,738:INFO:epoch[0], iter[1774], loss:9.072665, mean_fps:0.00 imgs/sec
  2021-10-26 17:47:18,135:INFO:epoch[1], iter[3549], loss:8.373915, mean_fps:2223.15 imgs/sec
  2021-10-26 17:50:42,717:INFO:epoch[2], iter[5324], loss:8.244397, mean_fps:2221.12 imgs/sec
  ...

  ```

- GPU处理器环境运行

  ```bash

  bash scripts/run_distribute_train_GPU.sh 8 0,1,2,3,4,5,6,7

  ```

  上述shell脚本将在后台运行分布训练。您可以通过/scripts/GPU_distributed/output.log文件查看结果。采用以下方式达到损失值：

  ```bash

  2021-11-12 08:04:58,607:INFO:epoch[0], iter[1774], loss:9.040621, mean_fps:0.00 imgs/sec
  2021-11-12 08:04:58,607:INFO:epoch[0], iter[1774], loss:8.997707, mean_fps:0.00 imgs/sec
  2021-11-12 08:04:58,608:INFO:epoch[0], iter[1774], loss:8.859007, mean_fps:0.00 imgs/sec
  2021-11-12 08:04:59,476:INFO:epoch[0], iter[1774], loss:8.927963, mean_fps:0.00 imgs/sec
  2021-11-12 08:16:03,775:INFO:epoch[1], iter[3549], loss:8.358135, mean_fps:683.14 imgs/sec
  2021-11-12 08:16:03,776:INFO:epoch[1], iter[3549], loss:8.495537, mean_fps:683.14 imgs/sec
  2021-11-12 08:16:03,776:INFO:epoch[1], iter[3549], loss:8.600464, mean_fps:683.14 imgs/sec
  2021-11-12 08:16:03,776:INFO:epoch[1], iter[3549], loss:8.598774, mean_fps:683.14 imgs/sec
  2021-11-12 08:16:03,776:INFO:epoch[1], iter[3549], loss:8.281206, mean_fps:683.14 imgs/sec
  ...

  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估LFW数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“/data/sphereface/6000 9923.ckpt”。

  ```bash

  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval_standalone.sh 0 "/data/sphereface/6000 9923.ckpt"

  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9923 std=0.005 thd=0.3050

  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“/data/sphereface/scripts/device0/0-20_1775.ckpt”。测试数据集的准确性如下：

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9928 std=0.0045 thd=0.2940

  ```

- 在GPU环境运行时评估LFW数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“/data/sphereface/6000 9923.ckpt”。

  ```bash

  python eval.py > eval.log 2>&1 &
  OR
  bash run_eval_standalone_GPU.sh 0 "/data/sphereface/6000 9923.ckpt"

  ```

  上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9913 std=0.004 thd=0.2840

  ```

  注：对于分布式训练后评估，请将checkpoint_path设置为最后保存的检查点文件，如“/data/sphereface/scripts/GPU_distributed/0-20_1775.ckpt”。测试数据集的准确性如下：

  ```bash

  # grep "accuracy:" eval.log
  LFWACC=0.9918 std=0.0045 thd=0.2800

  ```

## 导出过程

### 导出

```shell
python export.py --file_format [EXPORT_FORMAT]
```

`EXPORT_FORMAT` 可选 ["AIR", "MINDIR"]

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 推理

在推理之前需要先导出模型，AIR模型只能在昇腾910环境上导出，MINDIR可以在任意环境上导出。

```shell

# 昇腾310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]

```

DATASET_NAME目前仅支持sphereface
推理的结果直接出现在shell中
sphereface网络使用LFW推理得到的结果如下:

  ```log

  now the idx is %d 9
  LFWACC=0.9922 std=0.0044 thd=0.2800

  ```

# 模型描述

## 性能

### 评估性能

#### CASIA-WebFace上的Sphereface

| 参数                 | Ascend                                                      |GPU|
| -------------------------- | ----------------------------------------------------------- |----|
| 模型版本              | Spherenet20a|Spherenet20a|
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8       |V100*8; CPU 2.6GHZ 56核；内存 377G；系统Euler2.8|
| 上传日期              | 2021-10-28                                |2021-11-23|
| MindSpore版本          | 1.3.0 |1.5.0
| 数据集                    | CASIA-WebFace                              |CASIA-WebFace                              |
| 训练参数        | epoch=20, steps=1775, batch_size = 256, lr=0.15    | epoch=20, steps=1775, batch_size = 256, lr=0.15    |
| 优化器                  | Momentum                                   |Momentum                                   |
| 损失函数              | ASoftmax交叉熵                                   |ASoftmax交叉熵                                   |
| 输出                    | 概率                                          |概率                                          |
| 损失                       | 3.1677                                     |3.2163                                     |
| 速度                      | 单卡：795毫秒/步;  8卡：931毫秒/步                 |单卡：697毫秒/步;  8卡：373毫秒/步                 |
| 总时长                 | 单卡：469.98分钟;  8卡：68.79分钟                     |单卡：412.39分钟;  8卡：220.69分钟                     |
| 参数(M)             | 13.0                                                 |13.0                                                 |
| 微调检查点 | 214.58M (.ckpt文件)                                        |214.58M (.ckpt文件)                                        |
| 脚本                    | [sphereface脚本](https://gitee.com/mindspore/models/tree/master/research/cv/sphereface) | [sphereface脚本](https://gitee.com/mindspore/models/tree/master/research/cv/sphereface) |

### 推理性能

#### LFW上的Sphereface

| 参数          | Ascend                      |GPU|
| ------------------- | --------------------------- |----|
| 模型版本       | Sphereface20a                |Sphereface20a|
| 资源            |  Ascend 910；系统 Euler2.8                  | V100*8；系统 Euler2.8|
| 上传日期       | 2021-10-28 |2021-11-13|
| MindSpore 版本   | 1.3.0                       |1.5.0|
| 数据集             | LFW, 1.3万张图像     |LFW, 1.3万张图像     |
| batch_size          | 60                         |60                         |
| 输出             | 概率                 |概率                 |
| 准确性            | 单卡: 99.23%;  8卡：99.28%   |单卡: 99.13%;  8卡：99.18%   |

## 使用流程

### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/quick_start/quick_video/inference.html?highlight=%E5%B9%B3%E5%8F%B0)。下面是操作步骤示例：

- Ascend、GPU处理器环境运行

  ```python

  # 设置上下文
  config.image_size = list(map(int,config.image_size.split(',')))
  context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        save_graphs=False)

  # 加载未知数据集进行推理
  ImgOut = getImg(datatxt_src,pairtxt_src,datasrc,network)

  # 定义模型
  network = spherenet(config.num_classes,True)

  # 加载预训练模型
   param_dict = load_checkpoint(model)
   param_dict_new = {}
       for key, values in param_dict.items():
           if key.startswith('moments.'):
               continue
           elif key.startswith('network.'):
               param_dict_new[key[8:]] = values
           else:
               param_dict_new[key] = values
       load_param_into_net(network, param_dict_new)

  # 对未知数据集进行预测
  for idx, (train, test) in enumerate(folds):
      best_thresh = find_best_threshold(thresholds, predicts[train])
      accuracy.append(eval_acc(best_thresh, predicts[test]))
      thd.append(best_thresh)
  print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

  ```

### 继续训练预训练模型

- Ascend、GPU处理器环境运行

  ```python

  # 加载数据集
  de_dataset = classification_dataset_imagenet(data_dir, image_size=[112,96],
                                                 per_batch_size=config.per_batch_size, max_epoch=config.max_epoch,
                                                 rank=0, group_size=config.group_size,
                                                 input_mode="txt", root=images_dir, shuffle=True)

  # 定义模型
  network = sphere20a(config.num_classes,feature=False)
  # 若pre_trained为True，继续训练
      if os.path.isfile(config.train_pretrained):
          param_dict = load_checkpoint(config.train_pretrained)
          param_dict_new = {}
          for key, values in param_dict.items():
              if key.startswith('moments.'):
                  continue
              elif key.startswith('network.'):
                  param_dict_new[key[8:]] = values
              else:
                  param_dict_new[key] = values
          load_param_into_net(network, param_dict_new)
          config.logger.info('load model %s success', str(config.train_predtrained))
  lr_schedule = lr_scheduler.get_lr()
  opt = Momentum(params=get_param_groups(network), learning_rate=Tensor(lr_schedule),
                   momentum=config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)
  criterion = AngleLoss(classnum=config.num_classes,smooth_factor=config.smooth_factor)
  model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=loss_scale_manager, amp_level="O0")

  # 设置回调
  ckpt_max_num = config.max_epoch * config.steps_per_epoch // config.ckpt_interval
  ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval,
                                       keep_checkpoint_max=ckpt_max_num)
  ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=config.save_ckpt_path,
                                  prefix='%s' % config.rank)
  callbacks.append(ckpt_cb)

  # 开始训练
  model.train(config.max_epoch, de_dataset, callbacks=callbacks,dataset_sink_mode=False)

  ```

### 迁移学习

待补充

# 随机情况说明

在dataset.py中，我们设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
