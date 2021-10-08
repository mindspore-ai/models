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

# [RCNN Description](#contents)

RCNN (regions with CNN features) is a milestone in the application of CNN method to target detection.With the help of CNN's good feature extraction and classification performance, the transformation of target detection problem is realized by the region proposal method.

[Paper](https://arxiv.org/abs/1311.2524) Ross Girshick,Jeff Donahue,Trevor Darrell,Jitendra Malik. "Rich feature hierarchies for accurate object detection and semantic segmentation." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

# [Model architecture](#contents)

The overall network architecture of R-CNN is show below: [Link](https://arxiv.org/abs/1311.2524)

# [Dataset](#contents)

Dataset used: [VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

- Dataset size: ~870M, 17125 colorful images in 20 classes
    - Train: 439M, 9963 images
    - Test: 431M, 7162 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend/CPU）
    - Prepare hardware environment with Ascend/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```shell
├── RCNN
  ├── README.md              # descriptions about RCNN
  ├── scripts
  │   ├──run_distribute_train_ascend.sh        # shell script for single on Ascend
  │   ├──run_standalone_eval_ascend.sh         # shell script for evaluation on Ascend
  │   ├──run_standalone_train_ascend.sh        # shell script for distributed on Ascend
  ├── src
  │   ├── common
      │   ├──logger.py                  # logger utils
      │   ├──mindspore_utils.py         # mindspore utils, will be used in train.py and eval.py
      │   ├──multi_margin_loss.py       # multi_margin_loss, will be used in training svm.
      │   ├──trainer.py                 # the trainer of finetune,svm and bbox regression.
  │   ├── utils
      │   ├──util.py                    # Some functions used in training.
  │   ├──dataset.py          # create train or eval dataset.
  │   ├──generator_lr.py     # learning rate config
  │   ├──model.py            # RCNN architecture
  │   ├──paths.py            # network config setting, will be used in train.py and eval.py
  ├──process_data.py        # preprocess the dataset
  ├──train.py               # training script
  ├──eval.py                # evaluation script
  ├──requirements.txt       # python extension packages
```

## [Preparation](#contents)

You need to have the model of AlexNet that trained over ImageNet2012.Therefore,you can train it with [alexnet](https://gitee.com/mindspore/models/tree/master/official/cv/alexnet) scripts in modelzoo,and save the checkpoint file in a folder named "models".  The naming of the pretrained model is set in the "src/paths.py" .

You also need to create a floder named "data" to save data.The training set and test set are respectively in two folders, one is "VOC2007" and the other is "VOCtest_06-Nov-2007".The paths are also set in the "src/paths.py".

## [Data Preprocessing](#contents)

Run this script,it will extract thousands of candidate regions whose categories are independent from each image and generate data used in finetune, SVM and regression phases according to the IOU thresholds.

```shell
python process_data.py
 ```

## [Training Process](#contents)

### [Training](#contents)

- running on Ascend：

```shell
sh scripts/run_standalone_train_ascend.sh 0
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

### [Distributed Training](#contents)

- running on Ascend：

```shell
sh ./scripts/run_distribute_train_ascend.sh rank_table.json
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

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- evaluation on VOC2007 dataset when running on Ascend:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path.

```shell
sh scripts/run_standalone_eval_ascend.sh
 ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

```shell
# grep "accuracy: " scripts/result.txt
svm_thresh: 0.0, map: 0.31115641777358005
svm_thresh: 0.3, map: 0.3158392904125691
svm_thresh: 0.6, map: 0.31060216644862054
```

# [Model Description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                |
| Resource                   | Ascend 910, Memory 32G               |
| Uploaded Date              | 2021/7/27                               |
| MindSpore Version          | 1.2.0                                                       |
| Dataset                    | VOC2007                                                |
| Finetune Training Parameters        | epoch=2, batch_size=512, lr=0.013               |
| SVM Training Parameters        | epoch=30, batch_size=512, lr=0.001               |
| Regression Training Parameters        | epoch=30, batch_size=512, lr=0.0001               |
| Finetune Optimizer                  | Momentum                                                    |
| SVM Optimizer                  | Momentum
| Regression Optimizer                  | Adam
| Finetune Loss Function              | Softmax Cross Entropy                                     |
| SVM Loss Function              | MultiMarginLoss                                     |
| Regression Loss Function              |  MSELoss+L2Loss                                    |
| Finetune Speed                      | 1 pc: 341 ms/step                        |
| SVM Speed                      | 1 pc: 344 ms/step                        |
| Regression Speed                      | 1 pc: 7418 ms/step                        |
| Finetune Total Time                 | 1 pc: 3.93hours                                           |
| SVM Total Time                 | 1 pc: 2.2hours                                             |
| Regression Total Time                 | 1 pc: 6.12hours                                             |
| Checkpoint for finetune | 214M (.ckpt file)                                         |
| Checkpoint for SVM | 214M (.ckpt file)                                         |
| Checkpoint for regression | 214M (.ckpt file)                                         |
| Scripts                    | [RCNN Scripts](https://gitee.com/mindspore/models/tree/master/research/cv/rcnn) |

### [Inference Performance](#contents)

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | V1                |
| Resource            | Ascend 910                 |
| Uploaded Date       | 2021/7/27 |
| MindSpore Version   | 1.2.0                       |
| Dataset             | VOC2007                |
| batch_size          | 512                         |
| Outputs             | mAP                 |
| Accuracy            | 31.58%                 |

# [Description of Random Situation](#contents)

In src/common/mindspore_utils.py, we set the seed inside “initialize" function of "MSUtils" class. We use random seed in train.py.

In train.py, we set the seed which is used by numpy.random, mindspore.common.Initializer, mindspore.ops.composite.random_ops and mindspore.nn.probability.distribution.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).  
