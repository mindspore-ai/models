
# 目录

- [RCNN 简介](#RCNN简介)
- [模型结构](#模型结构)
- [数据集](#数据集)
- [环境配置](#环境配置)
- [脚本简介](#脚本简介)
    - [脚本和示例代码](#脚本和示例代码)
    - [准备](#准备)
    - [数据预处理](#数据预处理)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型简介](#模型简介)
    - [性能](#性能)  
        - [训练](#训练性能)
        - [推理](#推理性能)
- [随机情况简介](#随机情况简介)
- [ModelZoo主页](#Modelzoo主页)

## [RCNN简介](#contents)

RCNN(Regions with CNN features)是将CNN方法应用到目标检测问题上的一个里程碑,借助CNN良好的特征提取和分类性能,通过RegionProposal方法实现目标检测问题的转化。

[论文](https://arxiv.org/abs/1311.2524) Ross Girshick,Jeff Donahue,Trevor Darrell,Jitendra Malik. "Rich feature hierarchies for accurate object detection and semantic segmentation." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

## [模型结构](#contents)

R-CNN的整体网络架构详情，请参考链接: [Link](https://arxiv.org/abs/1311.2524)

## [数据集](#contents)

使用的数据集: [VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

- 数据集大小: ~870M, 17125张彩色图像共20类
    - 训练集: 439M, 9963张图像
    - 测试集: 431M, 7162张图像
- 数据格式: RGB图像
    - 通过 src/dataset.py 进行处理

## [环境配置](#contents)

- 硬件（GPU/Ascend/CPU）
    - 准备GPU/Ascend/CPU服务器环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 更多详情，点击下面链接：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

## [脚本简介](#contents)

### [脚本和示例代码](#contents)

```shell
├── RCNN
  ├── README.md              # descriptions about RCNN
  ├── README_CN.md           # descriptions about RCNN in Chinese
  ├──scripts
  │   ├── run_distribute_train_ascend.sh        # shell script for distributed on Ascend
  │   ├── run_distribute_train_gpu.sh           # shell script for distributed on GPU
  │   ├── run_standalone_eval_ascend.sh         # shell script for evaluation on Ascend
  │   ├── run_standalone_train_ascend.sh        # shell script for single on Ascend
  │   ├── run_standalone_train_gpu.sh           # shell script for training on GPU
  │   ├── run_standalone_eval_gpu.sh            # shell script for evaluation on GPU
  ├── src
  │   ├── common
  │   │   ├──logger.py                  # logger utils
  │   │   ├──mindspore_utils.py         # mindspore utils, will be used in train.py and eval.py
  │   │   ├──multi_margin_loss.py       # multi_margin_loss, will be used in training svm.
  │   │   ├──trainer.py                 # the trainer of finetune,svm and bbox regression.
  │   ├── utils
  │   │   ├──util.py                    # Some functions used in training.
  │   ├──dataset.py          # create train or eval dataset.
  │   ├──generator_lr.py     # learning rate config
  │   ├──model.py            # RCNN architecture
  │   ├──paths.py            # network config setting, will be used in train.py and eval.py
  ├──process_data.py          # preprocess the dataset
  ├──train.py               # training script
  ├──eval.py                # evaluation script
  ├──requirements.txt       # python extension packages
```

### [准备](#contents)

准备AlexNet在ImageNet2012上的预训练模型，可直接通过链接下载[link](https://download.mindspore.cn/)；也可自己训练，脚本参考[ModelZoo](https://gitee.com/mindspore/models/tree/master/official/cv/alexnet)，并将checkpoints文件保存在“models”文件夹中。您还需要创建一个名为“data”的文件夹来存放数据集和处理后的数据。训练集和测试集分别在两个文件夹中，一个是“VOC2007”，另一个是“VOCtest_06-Nov-2007”。 路径也在“src/paths.py”中设置。

### [数据预处理](#contents)

运行此脚本，它将从每个图像中提取数千个类别独立的候选区域，并根据 IOU 阈值生成用于finetune、SVM 和regression的数据。
注意：请确保requirements.txt中相关依赖已经安装。

```shell
python process_data.py
 ```

### [训练过程](#contents)

#### [训练](#contents)

- 在Ascend上进行训练：

```shell
bash scripts/run_standalone_train_ascend.sh 0
 ```

上述命令将在后台运行，你可以在日志文件查看结果。

训练后，默认会在脚本文件夹下得到一些checkpoints文件，loss如下所示：:

```shell
# grep "loss is " logs/trainer_finetune_DEBUG.txt
[2021-07-18 00:18:43.394][DEBUG] trainer.py(69)->train(): [train] epoch 2, step: 149, time: 347ms/step, loss: 0.60761726
[2021-07-18 00:18:46.707][DEBUG] trainer.py(69)->train(): [train] epoch 2, step: 150, time: 375ms/step, loss: 0.5797861
[2021-07-18 00:18:50.057][DEBUG] trainer.py(69)->train(): [train] epoch 2, step: 151, time: 347ms/step, loss: 0.5278064
...
# grep "loss is " logs/trainer_svm_DEBUG.txt
[2021-07-18 12:29:03.883][DEBUG] trainer.py(69)->train(): [train] epoch 30, step: 44, time: 360ms/step, loss: 0.33891284
[2021-07-18 12:29:07.417][DEBUG] trainer.py(69)->train(): [train] epoch 30, step: 45, time: 414ms/step, loss: 0.3377576
[2021-07-18 12:29:10.989][DEBUG] trainer.py(69)->train(): [train] epoch 30, step: 46, time: 397ms/step, loss: 0.35292286
...
# grep "loss is " logs/trainer_regression_DEBUG.txt
[2021-07-18 18:15:08.116][DEBUG] trainer.py(69)->train(): [train] epoch 27, step: 5, time: 7568ms/step, loss: 0.006192993
[2021-07-18 18:15:49.120][DEBUG] trainer.py(69)->train(): [train] epoch 27, step: 6, time: 7540ms/step, loss: 0.006083932
[2021-07-18 18:16:29.862][DEBUG] trainer.py(69)->train(): [train] epoch 27, step: 7, time: 7579ms/step, loss: 0.006008984
...
```

所有的checkpoints文件都将保存在“models”文件夹。

- 在GPU上训练：

通过执行如下命令进行训练：

bash scripts/run_standalone_train_gpu.sh 0

训练后，默认会在脚本文件夹下得到一些checkpoints文件，可以在日志文件中查看结果，loss如下所示：

```shell
# grep loss is  logs/trainer_finetune_DEBUG.txt
[2021-10-25 14:40:46.285][DEBUG] trainer.py(121)->train(): [train] epoch 2, step: 793, time: 154 ms/step, loss: 0.5137034
[2021-10-25 14:40:48.912][DEBUG] trainer.py(121)->train(): [train] epoch 2, step: 794, time: 156 ms/step, loss: 0.5286188
[2021-10-25 14:40:51.518][DEBUG] trainer.py(121)->train(): [train] epoch 2, step: 795, time: 173 ms/step, loss: 0.4410156
[2021-10-25 14:40:54.039][DEBUG] trainer.py(121)->train(): [train] epoch 2, step: 796, time: 154 ms/step, loss: 0.5034308
...
# grep loss is  logs/trainer_svm_DEBUG.txt
[2021-10-25 16:54:39.869][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 1, time: 137 ms/step, loss: 0.32176015
[2021-10-25 16:54:40.345][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 2, time: 155 ms/step, loss: 0.31022772
[2021-10-25 16:54:40.854][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 3, time: 213 ms/step, loss: 0.32364023
[2021-10-25 16:54:41.296][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 4, time: 150 ms/step, loss: 0.35638717
[2021-10-25 16:54:41.745][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 5, time: 149 ms/step, loss: 0.34618762
[2021-10-25 16:54:42.222][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 6, time: 185 ms/step, loss: 0.32193667
[2021-10-25 16:54:42.656][DEBUG] trainer.py(121)->train(): [train] epoch 30, step: 7, time: 149 ms/step, loss: 0.34347594
...
# grep loss is  logs/trainer_regression_DEBUG.txt
[2021-10-25 17:45:07.499][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 4, time: 155 ms/step, loss: 0.009431448
[2021-10-25 17:45:07.944][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 5, time: 154 ms/step, loss: 0.00899493
[2021-10-25 17:45:08.422][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 6, time: 186 ms/step, loss: 0.009508412
[2021-10-25 17:45:08.886][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 7, time: 162 ms/step, loss: 0.008825495
[2021-10-25 17:45:09.377][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 8, time: 165 ms/step, loss: 0.008961924
[2021-10-25 17:45:09.866][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 9, time: 153 ms/step, loss: 0.008931074
[2021-10-25 17:45:10.352][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 10, time: 154 ms/step, loss: 0.009009103
[2021-10-25 17:45:10.821][DEBUG] trainer.py(121)->train(): [train] epoch 3, step: 11, time: 160 ms/step, loss: 0.009312078
...
```

所有的checkpoints文件都将保存在“models”文件夹。

#### [分布式训练](#contents)

- 在Ascend上进行分布式训练：

```shell
bash ./scripts/run_distribute_train_ascend.sh rank_table.json
```

上述脚本将在后台运行. 您可以通过文件 train_parallel[X]/logs 查看结果。 损失如下:

```shell
# grep "loss is " train_parallel*/logs/trainer_finetune_DEBUG.txt
[2021-07-16 16:23:09.119][DEBUG] trainer.py(69)->train(): [train] epoch 1, step: 395, loss: 0.84556377
[2021-07-16 16:23:12.482][DEBUG] trainer.py(69)->train(): [train] epoch 1, step: 396, loss: 0.84089524
[2021-07-16 16:23:15.846][DEBUG] trainer.py(69)->train(): [train] epoch 1, step: 397, loss: 0.7097651
...
# grep "loss is " train_parallel*/logs/trainer_svm_DEBUG.txt
[2021-07-17 02:41:51.617][DEBUG] trainer.py(69)->train(): [train] epoch 17, step: 119, loss: 0.32603985
[2021-07-17 02:41:52.461][DEBUG] trainer.py(69)->train(): [train] epoch 17, step: 120, loss: 0.31629387
[2021-07-17 02:41:53.319][DEBUG] trainer.py(69)->train(): [train] epoch 17, step: 121, loss: 0.28632772
...
# grep "loss is " train_parallel*/logs/trainer_regression_DEBUG.txt
[2021-07-15 22:07:59.251][DEBUG] trainer.py(67)->train(): [train] epoch 10, step: 2, loss: 0.21290894  
[2021-07-15 22:08:02.776][DEBUG] trainer.py(67)->train(): [train] epoch 10, step: 3, loss: 0.17405899
[2021-07-15 22:08:05.755][DEBUG] trainer.py(67)->train(): [train] epoch 10, step: 4, loss: 0.16900891
...
```

- 在GPU上进行分布式训练：

```shell
bash scripts/run_distribute_train_gpu.sh
```

上述脚本将在后台运行。

```shell
# grep loss is  log_train_finetune
scripts/log_train_finetune:[2021-11-09 13:54:46.741][DEBUG] trainer.py(172)->validate(): [valid] epoch 2, step: 79, loss: 0.72336304, acc: 76.5625%
scripts/log_train_finetune:[2021-11-09 13:54:46.818][DEBUG] trainer.py(172)->validate(): [valid] epoch 2, step: 79, loss: 1.1784661, acc: 62.1094%
scripts/log_train_finetune:[2021-11-09 13:54:47.999][DEBUG] trainer.py(172)->validate(): [valid] epoch 2, step: 80, loss: 0.90129405, acc: 71.8750%
scripts/log_train_finetune:[2021-11-09 13:54:48.271][DEBUG] trainer.py(172)->validate(): [valid] epoch 2, step: 80, loss: 0.6390413, acc: 82.8125%
scripts/log_train_finetune:[2021-11-09 13:54:48.419][DEBUG] trainer.py(172)->validate(): [valid] epoch 2, step: 80, loss: 0.945093, acc: 69.5312
...
# grep loss is log_train_svm
scripts/log_train_svm:[2021-11-09 14:10:59.110][INFO] trainer.py(178)->validate(): [valid] epoch 30, loss: 0.835343986749649, acc: 76.7253%
scripts/log_train_svm:[2021-11-09 14:10:59.166][DEBUG] trainer.py(172)->validate(): [valid] epoch 30, step: 6, loss: 0.7916494, acc: 75.5859%
scripts/log_train_svm:[2021-11-09 14:10:59.167][INFO] trainer.py(178)->validate(): [valid] epoch 30, loss: 0.8657607436180115, acc: 76.4648%
scripts/log_train_svm:[2021-11-09 14:10:59.277][DEBUG] trainer.py(172)->validate(): [valid] epoch 30, step: 6, loss: 0.87103647, acc: 73.6328%
scripts/log_train_svm:[2021-11-09 14:10:59.277][INFO] trainer.py(178)->validate(): [valid] epoch 30, loss: 0.8794184923171997, acc: 75.8789%
scripts/log_train_svm:[2021-11-09 14:10:59.441][DEBUG] trainer.py(172)->validate(): [valid] epoch 30, step: 6, loss: 0.80178535, acc: 76.9531%
scripts/log_train_svm:[2021-11-09 14:10:59.442][INFO] trainer.py(178)->validate(): [valid] epoch 30, loss: 0.8207429647445679, acc: 77.7995%
scripts/log_train_svm:[2021-11-09 14:10:59.449][DEBUG] trainer.py(172)->validate(): [valid] epoch 30, step: 6, loss: 0.5946242, acc: 79.2969%
scripts/log_train_svm:[2021-11-09 14:10:59.449][INFO] trainer.py(178)->validate(): [valid] epoch 30, loss: 0.793100893497467, acc: 77.1159%
...
# grep loss is log_train_regression
scripts/log_train_regression:[2021-11-09 14:40:48.421][DEBUG] trainer.py(121)->train(): [train] epoch 10, step: 14, time: 2555 ms/step, loss: 0.0092799105
scripts/log_train_regression:[2021-11-09 14:40:48.423][DEBUG] trainer.py(121)->train(): [train] epoch 10, step: 14, time: 892 ms/step, loss: 0.009516025
scripts/log_train_regression:[2021-11-09 14:40:48.424][DEBUG] trainer.py(121)->train(): [train] epoch 10, step: 14, time: 1317 ms/step, loss: 0.0088452175
scripts/log_train_regression:[2021-11-09 14:40:58.586][DEBUG] trainer.py(121)->train(): [train] epoch 10, step: 15, time: 2549 ms/step, loss: 0.00871471
scripts/log_train_regression:[2021-11-09 14:40:58.586][DEBUG] trainer.py(121)->train(): [train] epoch 10, step: 15, time: 1269 ms/step, loss: 0.009734297
scripts/log_train_regression:[2021-11-09 14:40:58.586][DEBUG] trainer.py(121)->train(): [train] epoch 10, step: 15, time: 1653 ms/step, loss: 0.
...
```

### [评估过程](#contents)

#### [评估](#contents)

- 在Ascend上用VOC2007数据集进行评估:

在运行以下命令之前，请检查用于评估的checkpoints路径，请将checkpoints路径设置为绝对完整路径。

```shell
bash scripts/run_standalone_eval_ascend.sh 0
 ```

上面的脚本将在后台运行。您可以通过“eval.log”查看结果,测试数据集的准确率如下：

```shell
# grep "accuracy: " scripts/result.txt
svm_thresh: 0.0, map: 0.31115641777358005
svm_thresh: 0.3, map: 0.3158392904125691
svm_thresh: 0.6, map: 0.31060216644862054
```

- 在GPU上用VOC2007数据集进行评估：

在运行以下命令之前，请检查用于评估的checkpoints路径，请将checkpoints路径设置为绝对完整路径。

```shell

bash scripts/run_standalone_eval_gpu.sh 0

 ```

上面的python命令将在后台运行。您可以通过“result.txt”查看结果,测试数据集的准确率如下：

```shell
# grep map scripts/result.txt
svm_thresh: 0.0, map: 0.31635177699636685
svm_thresh: 0.3, map: 0.3190850349142451
svm_thresh: 0.6, map: 0.3254243053285871
```

## [模型简介](#contents)

### [性能](#contents)

#### [训练性能](#contents)

| Parameters                 | Ascend                                                      |GPU                                                      |
| -------------------------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| Model Version              | V1                                                |V1               |
| Resource                   | Ascend 910, Memory 32G               |GeForce RTX 3090, Memory 24G               |
| Uploaded Date              | 2021/7/27                               | 2021/9/27                               |
| MindSpore Version          | 1.2.0                                                       |1.3.0                                                       |
| Dataset                    | VOC2007                                                |VOC2007                                                |
| Finetune Training Parameters        | epoch=2, batch_size=512, lr=0.013               |epoch=2, batch_size=512, lr=0.013               |
| SVM Training Parameters        | epoch=30, batch_size=512, lr=0.001               |epoch=30, batch_size=512, lr=0.001               |
| Regression Training Parameters        | epoch=30, batch_size=512, lr=0.0001               |epoch=30, batch_size=512, lr=0.0001                     |
| Finetune Optimizer                  | Momentum                                                    |Momentum                                                    |
| SVM Optimizer                  | Momentum                                               |Momentum|
| Regression Optimizer                  | Adam                                             | Adam|
| Finetune Loss Function              | Softmax Cross Entropy                                     |Softmax Cross Entropy|
| SVM Loss Function              | MultiMarginLoss                                     |MultiMarginLoss|
| Regression Loss Function              |  MSELoss+L2Loss                                    | MSELoss+L2Loss |
| Finetune Speed                      | 1 pc: 341 ms/step                        |1 pc: 163 ms/step  |
| SVM Speed                      | 1 pc: 344 ms/step                        |1 pc: 162 ms/step|
| Regression Speed                      | 1 pc: 7418 ms/step                        |1 pc: 156 ms/step|
| Finetune Total Time                 | 1 pc: 3.93hours                                           |1 pc: 2.55hours |
| SVM Total Time                 | 1 pc: 2.2hours                                             |1 pc: 1.55hours|
| Regression Total Time                 | 1 pc: 6.12hours                                             |1 pc: 11hours|
| Checkpoint for finetune | 214M (.ckpt file)                                         | 214M (.ckpt file)|
| Checkpoint for SVM | 214M (.ckpt file)                                         |214M (.ckpt file)|
| Checkpoint for regression | 214M (.ckpt file)                                         |214M (.ckpt file)|
| Scripts                    | [RCNN Scripts](https://gitee.com/mindspore/models/tree/master/research/cv/rcnn) |[RCNN Scripts](https://gitee.com/mindspore/models/tree/master/research/cv/rcnn) |

#### [推理性能](#contents)

| Parameters          | Ascend                      |GPU                      |
| ------------------- | --------------------------- |--------------------------- |
| Model Version       | V1                |V1                |
| Resource            | Ascend 910                 |GeForce RTX 3090                 |
| Uploaded Date       | 2021/7/27 |2021/9/27 |
| MindSpore Version   | 1.2.0                       | 1.3.0                       |
| Dataset             | VOC2007                | VOC2007                |
| batch_size          | 512                         | 512                         |
| Outputs             | mAP                 | mAP                 |
| Accuracy            | 31.58%                 | 32.54%                 |

## [随机情况简介](#contents)

在src/common/mindspore_utils.py中，我们在“MSUtils”类的“初始化”函数中设置了种子。我们在train.py中使用随机种子。

## [ModelZoo主页](#contents)

请查看[官方网站](https://gitee.com/mindspore/models)。
