# contents

<!-- TOC -->

- [Contents](#contents)
- [ResNet50_bam description](#resnet50_bam-description)
- [Dataset](#data-set)
- [Characteristic](#characteristic)
    - [Mixed precision](#mixed-precision)
- [Environmental requirements](#environmental-requirements)
- [Quick start](#quick-start)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Training process](#training-process)
        - [Standalone training](#standalone-training)
        - [Distributed training](#distributed-training)
    - [Evaluation process](#evaluation-process)
        - [Evaluate](#evaluate)
    - [Export process](#export-process)
        - [Export](#export)
    - [Inference process](#inference-process)
        - [Inference](#inference)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Ascend performance](#ascend-performance)
            - [Resnet50_bam on ImageNet2012](#resnet50_bam-on-imagenet2012)
        - [Inference performance](#inference-performance)
            - [Resnet50_bam on ImageNet2012](#resnet50_bam-on-imagenet2012-1)
- [ModelZoo homepage](#modelzoo-homepage)

<!-- /TOC -->

# resnet50_bam description

The author of resnet50_bam proposed a simple but effective Attention model-BAM, which can be combined with any forward propagation convolutional neural network. The author puts BAM between each stage in the ResNet network. Multi-layer BAMs form a hierarchical attention mechanism, which is a bit like a human perception mechanism. BAM eliminates background semantic features between each stage. The low-level features of, and then gradually focus on the high-level semantics-clear goals.

# Data Set

Data set used: [ImageNet2012](http://www.image-net.org/)

- Data set size: 125G, a total of 1000 categories, 1.25 million color images
    - Training set: 120G, a total of 1,281,167 images
    - Test set: 5G, a total of 50,000 images
- Data format: RGB
    - Note: The data will be processed in src/dataset.py.
- Download the data set, the directory structure is as follows:

  ```text
  └─dataset
      ├─ILSVRC2012_train   # Training dataset
      └─ILSVRC2012_val     # Validation dataset
  ```

# characteristic

## Mixed precision

The [mixed-precision](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html) training method uses single-precision and half-precision data to improve the training speed of deep learning neural networks, while maintaining the network accuracy that can be achieved by single-precision training. Mixed-precision training increases computing speed and reduces memory usage, while supporting training larger models or achieving larger batches of training on specific hardware.

# Environmental requirements

- Hardware (Ascend or GPU)
    - Use Ascend or GPU to build a hardware environment.

- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For details, please refer to the following resources：
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:：

- Ascend processor environment operation

  ```bash
  # Example of starting training
  python train.py --device_id=0 > train.log 2>&1 &

  # run distribute train for Ascend
  bash ./scripts/run_distribute_train.sh [RANK_TABLE_FILE]

  # run evaluation
  python eval.py --checkpoint_path ./ckpt > ./eval.log 2>&1 &

  # Inference example
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  For distributed training, you need to create an hccl configuration file in JSON format in advance.

  Please follow the instructions in the link below:

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- GPU

  ```bash
  # Example of starting training
  python train.py --device_target=GPU > train.log 2>&1 &

  # run distribute train for GPU
  bash ./scripts/run_distribute_train_gpu.sh

  # run evaluation
  python eval.py --device_target="GPU" --checkpoint_path = './ckpt' ./logs/eval.log 2>&1 &
  ```

# Script description

## Script and sample code

```bash
├── model_zoo
    ├── README.md                       // All instructions related to the model
    ├── resnet50_bam
        ├── ascend310_infer             // Ascend310 inference
        ├── scripts
        │   ├──run_distribute_train.sh  // distribute Ascend training script
        │   ├──run_distribute_train_gpu.sh  // 8P GPU training
        │   ├──run_eval.sh              // eval Ascend script
        │   ├──run_eval_gpu.sh              // GPU evaluation
        │   ├──run_infer_310.sh         // inference Ascend310 script
        │   ├──run_standalone_train_gpu.sh  // 1P GPU training
        ├── src
        │   ├──config.py                // parameter configuration
        │   ├──dataset.py               // create datase
        │   ├──my_lossmonitor.py          // loss monitor
        │   ├──ResNet50_BAM.py          // resnet50_bam architecture
        ├── README.md                   // resnet50_bam description in English
        ├── README_CN.md                // resnet50_bam description in Chinese
        ├── create_imagenet2012_label.py // create imagenet 2012 label
        ├── eval.py                     // evaluation script
        ├── export.py                   // export a ckpt to air/mindir
        ├── postprocess.py              // Ascend310 postrocess
        ├── train.py                    // trainig script
```

## Script parameters

Training parameters and evaluation parameters can be configured in config.py at the same time.

- Configure resnet50_bam and ImageNet2012 data sets.

  ```text
  'name':'imagenet'        # dataset
  'pre_trained':'False'    # should you train with a pretrained model
  'num_classes':1000       # number of dataset classes
  'lr_init':0.02           # initial learning rate is set to 0.02 for single card training and 0.18 for parallel training with eight cards
  'batch_size':128         # training package size
  'epoch_size':160         # total number of training epochs
  'momentum':0.9           # momentum
  'weight_decay':1e-4      # the amount of weight loss
  'image_height':224       # height of the image entered into the model
  'image_width':224        # width of the image entered into the model
  'data_path':'/data/ILSVRC2012_train/'  # the absolute full path of the training dataset
  'val_data_path':'/data/ILSVRC2012_val/'  # the absolute full path of the eval dataset
  'device_target':'Ascend' # operating equipment
  'device_id':0            # the device ID used for training or evaluating a dataset can be ignored when using run_train.sh for distributed training
  'keep_checkpoint_max':25 # Save up to 25 ckpt model files
  'checkpoint_path':'./ckpt/train_resnet50_bam_imagenet-156_10009.ckpt'  # checkpoint_path
  ```

For more configuration details, please refer to the script `config.py`。

## Training process

### Standalone training

- Ascend processor environment operation

  ```bash
  python train.py --device_id=0 > train.log 2>&1 &
  ```

- GPU processor environment operation

  ```bash
  python train.py --device_target=GPU > train.log 2>&1 &
  ```

The above python commands will run in the background, and the results can be viewed through the generated train.log file.

### Distributed training

- Ascend processor environment operation

  ```bash
  bash ./scripts/run_distribute_train.sh [RANK_TABLE_FILE]
  ```

- GPU processor environment operation

  ```bash
  bash ./scripts/run_distribute_train_gpu.sh
  ```

  The above shell script will run distributed training in the background.

## Evaluation process

### Evaluate

- Evaluate the ImageNet2012 dataset while running in the Ascend environment

  "./ckpt" is the directory where the trained .ckpt model files are saved.

  ```bash
  python eval.py --checkpoint_path ./ckpt > ./eval.log 2>&1 &
  OR
  bash ./scripts/run_eval.sh
  ```

- Evaluate the ImageNet2012 dataset while running in the GPU environment

  ```bash
  python eval.py --device_target="GPU" --checkpoint_path = './ckpt' ./logs/eval.log 2>&1 &
  OR
  bash ./scripts/run_eval_gpu.sh [CHECKPOINT_PATH]
  ```

## Export process

### Export

Export the checkpoint file into a mindir format model.

```shell
python export.py --ckpt_file [CKPT_FILE] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
```

> DEVICE_TARGET: Ascend, GPU or CPU (Default: Ascend)
>
> FILE_FORMAT: MINDIR, AIR or ONNX (Default: MINDIR)

## Inference process

### Inference

Before inference, we need to export the model first. Mindir can be exported in any environment, and the air/mindir model can only be exported in the Shengteng 910 environment. The following shows an example of using the mindir model to perform inference.

- Use ImageNet2012 data set for inference on Shengteng 31

  The results of the inference are stored in the scripts directory, and results similar to the following can be found in the acc.log log file.

  ```shell
  # Ascend310 inference
  bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  Total data: 50000, top1 accuracy: 0.77234, top5 accuracy: 0.93536.
  ```

# Model description

## Performance

### Training performance

#### Resnet50_bam on ImageNet2012

| parameter                  | Ascend                                                                                     | GPU                                                       |
| -------------------------- | -----------------------------------------------------------------------------------------  |-----------------------------------------------------------|
| Model version              | resnet50_bam                                                                               | resnet50_bam                                              |
| Resource                   | Ascend 910                                                                                 | V100-PCIE                                                 |
| Upload date                | 2021-06-02                                                                                 | 2021-12-02                                                         |
| MindSpore version          | 1.2.0                                                                                      | 1.5.0 (Docker build, CUDA 11.1)                           |
| Dataset                    | ImageNet2012                                                                               | ImageNet2012                                            |
| Training parameters        | epoch=160, batch_size=128, lr_init=0.02 (for a single card - 0.02, for eight cards - 0.19  | epoch=160, batch_size=128, lr_init=0.18 (for eight cards) |
| Optimizer                  | Momentum                                                                                   | Momentum                                                  |
| Loss function              | Softmax cross entropy                                                                      | Softmax cross entropy                                     |
| Output                     | Probability                                                                                | Probability                                               |
| Classification accuracy    | Single card: top1: 77.23%, top5: 93.56%; eight cards: top1: 77.35%, top5: 93.56%           | Eight cards: top1: 77.36%, top5: 93.26%                   |
| Speed                      | Single card: 96 milliseconds/step; Eight cards: 101 milliseconds/step                      | 1p: 185 ms/step, 8P: 228ms/step                           |
| Total time                 | Single card: 45.2 hours/160 rounds; Eight cards: 5.7 hours/160 rounds                      | Single card: 81 hours; Eight cards: 12.7 hours            |

### Inference performance

#### Resnet50_bam on ImageNet2012

| Parameter                  | Ascend                                         | GPU                                    |
| -------------------------- | ---------------------------------------------- | -------------------------------------- |
| Model                      | resnet50_bam                                   | resnet50_bam                           |
| Device                     | Ascend 310                                     | V100-PCIE                              |
| Upload date                | 2021-06-16                                     | 2021-12-02                                      |
| MindSpore version          | 1.2.0                                          | 1.5.0 (docker build, CUDA 11.1)        |
| Dataset                    | ImageNet2012                                   | ImageNet2012                           |
| Accuracy                   | top1: 77.23%, top5: 93.54%                     | top1: 77.36%, top5: 93.26%  (8 cards)  |
| Speed                      | Average time 4.8305 ms of infer_count 50000    | 751 images/second (~1.3 ms/image)      |

# ModelZoo homepage

 Please visit the official website [homepage](https://gitee.com/mindspore/models)
