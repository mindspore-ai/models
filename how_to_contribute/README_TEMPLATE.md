# Content

# Model name

> The Description of Model. The paper present this model.

## Model Architecture

> There could be various architecture about some model. Represent the architecture of your implementation.

## Dataset

> Provide the information of the dataset you used. Check the copyrights of the dataset you used, usually you need to provide the hyperlink to download the dataset, scope and data size.

## Features(optional)

> Represent the distinctive feature you used in the model implementation. Such as distributed auto-parallel or some special training trick.

## Requirements

> Provide details of the software required, including:
>
> * The additional python package required. Add a `requirements.txt` file to the root dir of model for installing dependencies.
> * The necessary third-party code.
> * Some other system dependencies.
> * Some additional operations before training or prediction.

## Quick Start

> How to take a try without understanding anything about the model.
> Maybe include：
> * run train，run eval，run export
> * Ascend version, GPU version，CPU version
> * offline version，ModelArts version

## Script Description

> The section provide the detail of implementation.

### Scripts and Sample Code

> Show the scope of project(include children directory), Explain every file in your project.

### Script Parameter

> Explain every parameter of the model. Especially the parameters in `config.py`. If there are multiple config files, please explain separately.

## Training

> Provide training information. Include usage and log.

### Training Process

> Provide the usage of training scripts.

e.g. Run the following command for distributed training on Ascend.

```shell
bash run_distribute_train.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

> Provide training logs.

```log
# grep "loss is " train.log
epoch:1 step:390, loss is 1.4842823
epcoh:2 step:390, loss is 1.0897788
```

> Provide training result.
e.g. Training checkpoint will be stored in `XXXX/ckpt_0`. You will get result from log file like the following:

```log
epoch: 11 step: 7393 ,rpn_loss: 0.02003, rcnn_loss: 0.52051, rpn_cls_loss: 0.01761, rpn_reg_loss: 0.00241, rcnn_cls_loss: 0.16028, rcnn_reg_loss: 0.08411, rcnn_mask_loss: 0.27588, total_loss: 0.54054
epoch: 12 step: 7393 ,rpn_loss: 0.00547, rcnn_loss: 0.39258, rpn_cls_loss: 0.00285, rpn_reg_loss: 0.00262, rcnn_cls_loss: 0.08002, rcnn_reg_loss: 0.04990, rcnn_mask_loss: 0.26245, total_loss: 0.39804
```

### Transfer Training(Optional)

> Provide the guidelines about how to run transfer training based on an pretrained model.

### Distribute Training

> Same as Training

## Evaluation

### Evaluation Process 910

> Provide the use of evaluation scripts.

### Evaluation Result 910

> Provide the result of evaluation.

## Export

### Export Process

> Provide the use of export scripts.

### Export Result

> Provide the result of export.

## Evaluation 310

### Evaluation Process 310

> Provide the use of evaluation scripts.

### Evaluation Result 310

> Provide the result of evaluation.

## Performance

### Training Performance

> Provide the detail of training performance including finishing loss, throughput, checkpoint size and so on.

e.g. you can reference the following template

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | ResNet18                                                     |  ResNet18                                     |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  PCIE V100-32G                                |
| uploaded Date              | 02/25/2021 (month/day/year)                                  | 07/23/2021 (month/day/year)                   |
| MindSpore Version          | 1.1.1                                                        | 1.3.0                                         |
| Dataset                    | CIFAR-10                                                     | CIFAR-10                                      |
| Training Parameters        | epoch=90, steps per epoch=195, batch_size = 32               | epoch=90, steps per epoch=195, batch_size = 32|
| Optimizer                  | Momentum                                                     | Momentum                                      |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                         |
| outputs                    | probability                                                  | probability                                   |
| Loss                       | 0.0002519517                                                 |  0.0015517382                                 |
| Speed                      | 13 ms/step（8pcs）                                           | 29 ms/step（8pcs）                            |
| Total time                 | 4 mins                                                       | 11 minds                                      |
| Parameters (M)             | 11.2                                                         | 11.2                                          |
| Checkpoint for Fine tuning | 86M (.ckpt file)                                             | 85.4 (.ckpt file)                             |
| Scripts                    | [link](https://gitee.com/mindspore/models/tree/master/official/cv/resnet)                                    |

### Inference Performance

> Provide the detail of evaluation performance including latency, accuracy and so on.

e.g. you can reference the following template

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | ResNet18                    |
| Resource            | Ascend 910; OS Euler2.8     |
| Uploaded Date       | 02/25/2021 (month/day/year) |
| MindSpore Version   | 1.1.1                       |
| Dataset             | CIFAR-10                    |
| batch_size          | 32                          |
| outputs             | probability                 |
| Accuracy            | 94.02%                      |
| Model for inference | 43M (.air file)             |

## Description of Random Situation

> Explain the random situation in the project.

## Reference Example

[maskrcnn_readme](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/README.md)

## Contributions

This part should not exist in your readme.
If you want to contribute, please review the [contribution guidelines](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING.md) and [how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

### Contributors

Update your school and email/gitee.

* [c34](https://gitee.com/c_34) (Huawei)

## ModeZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).
