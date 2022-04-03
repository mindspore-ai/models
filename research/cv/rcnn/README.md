# Contents

- [RCNN Description](#RCNN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Preparation](#data-preprocessing)
    - [Data Preprocessing](#data-preprocessing)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

## [RCNN Description](#contents)

RCNN (regions with CNN features) is a milestone in the application of CNN method to target detection.With the help of CNN's good feature extraction and classification performance, the transformation of target detection problem is realized by the region proposal method.

[Paper](https://arxiv.org/abs/1311.2524) Ross Girshick,Jeff Donahue,Trevor Darrell,Jitendra Malik. "Rich feature hierarchies for accurate object detection and semantic segmentation." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

## [Model architecture](#contents)

The overall network architecture of R-CNN is show below: [Link](https://arxiv.org/abs/1311.2524)

## [Dataset](#contents)

Dataset used: [VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

- Dataset size: ~870M, 17125 colorful images in 20 classes
    - Train: 439M, 9963 images
    - Test: 431M, 7162 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

## [Environment Requirements](#contents)

- Hardware（GPU/Ascend/CPU）
    - Prepare hardware environment with GPU/Ascend/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

## [Script description](#contents)

### [Script and sample code](#contents)

```shell
├── RCNN
  ├── README.md              # descriptions about RCNN
  ├── README_CN.md           # descriptions about RCNN in Chinese
  ├── scripts
  │   ├──run_distribute_train_ascend.sh        # shell script for distributed on Ascend
  │   ├──run_distribute_train_gpu.sh           # shell script for distributed on GPU
  │   ├──run_standalone_eval_ascend.sh         # shell script for evaluation on Ascend
  │   ├──run_standalone_train_ascend.sh        # shell script for single on Ascend
  │   ├──run_standalone_eval_gpu.sh            # shell script for evaluation on GPU
  │   ├──run_standalone_train_gpu.sh           # shell script for distributed on GPU
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
  ├──process_data.py        # preprocess the dataset
  ├──train.py               # training script
  ├──eval.py                # evaluation script
  ├──requirements.txt       # python extension packages
```

### [Preparation](#contents)

You need to have the model of AlexNet that trained over ImageNet2012.Therefore,you can train it with AlexNet scripts in [ModelZoo](https://gitee.com/mindspore/models/tree/master/official/cv/alexnet),and save the checkpoint file in a folder named "models".  The naming of the pretrained model is set in the "src/paths.py" .

You also need to create a floder named "data" to save data.The training set and test set are respectively in two folders, one is "VOC2007" and the other is "VOCtest_06-Nov-2007".The paths are also set in the "src/paths.py".

### [Data Preprocessing](#contents)

Run this script,it will extract thousands of candidate regions whose categories are independent from each image and generate data used in finetune, SVM and regression phases according to the IOU thresholds.
Note: Please make sure that the relevant dependencies in requirements.txt have been installed.

```shell
python process_data.py
 ```

### [Training Process](#contents)

#### [Training](#contents)

- running on Ascend：

```shell
bash scripts/run_standalone_train_ascend.sh 0
 ```

  The command above will run in the background, you can view the results through the file logs

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

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

  The model checkpoint file will be saved in the folder "models".
  <!-- The model checkpoint will be saved in the directory models.  -->

- running on GPU：

```shell
bash scripts/run_standalone_train_gpu.sh 0
 ```

  The command above will run in the background, you can view the results through the file logs

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

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

  The model checkpoint file will be saved in the folder "models".
  <!-- The model checkpoint will be saved in the directory models.  -->

#### [Distributed Training](#contents)

- distributed running on Ascend：

```shell
bash ./scripts/run_distribute_train_ascend.sh rank_table.json
```

  The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/logs. The loss value will be achieved as follows:

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

- distributed running on GPU：

```shell
bash scripts/run_distribute_train_gpu.sh
```

The above shell script will run distribute training in the background.

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

### [Evaluation Process](#contents)

#### [Evaluation](#contents)

- evaluation on VOC2007 dataset when running on Ascend:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path.

```shell
bash scripts/run_standalone_eval_ascend.sh 0
 ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep "accuracy: " scripts/result.txt
svm_thresh: 0.0, map: 0.31115641777358005
svm_thresh: 0.3, map: 0.3158392904125691
svm_thresh: 0.6, map: 0.31060216644862054
```

- evaluation on VOC2007 dataset when running on GPU:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path.

```shell
bash scripts/run_standalone_eval_gpu.sh 0
 ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep map scripts/result.txt
svm_thresh: 0.0, map: 0.31635177699636685
svm_thresh: 0.3, map: 0.3190850349142451
svm_thresh: 0.6, map: 0.3254243053285871
```

## [Model Description](#contents)

### [Performance](#contents)

#### [Training Performance](#contents)

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
| Scripts                    | [RCNN Scripts](https://gitee.com/mindspore/models/tree/master/research/cv/) |[RCNN Scripts](https://gitee.com/mindspore/models/tree/master/research/cv/rcnn) |

#### [Inference Performance](#contents)

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

## [Description of Random Situation](#contents)

In src/common/mindspore_utils.py, we set the seed inside “initialize" function of "MSUtils" class. We use random seed in train.py.

In train.py, we set the seed which is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).  
