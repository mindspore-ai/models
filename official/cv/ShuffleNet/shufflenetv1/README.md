# Contents

- [Contents](#contents)
- [ShuffleNetV1Description](#shufflenetv1-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Launch](#launch)
        - [Result](#result)
    - [Evaluation Process](#evaluation-process)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Training Performance](#training-performance)
- [Random Seed Description](#random-seed-description)
- [ModelZoo](#modelzoo)

# ShuffleNetV1 Description

ShuffleNetV1 is a computing-efficient CNN model proposed by Face++. It is mainly used on mobile devices. Therefore, the model is designed to use limited computing resources to achieve the best model accuracy. The core design of ShuffleNetV1 is to introduce two operations: pointwise group convolution and channel shuffle, which greatly reduces the calculation workload of the model while maintaining the accuracy. Therefore, similar to MobileNet, ShuffleNetV1 compresses and accelerates models by designing a more efficient network structure.

[Paper](https://arxiv.org/abs/1707.01083): Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun."ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

# Model Architecture

The core of ShuffleNetV1 is divided into three stages. In each stage, several basic units of ShuffleNetV1 are stacked. The first basic unit uses pointwise group convolution with 2 strides to reduce the width and height of the feature map by half and double the channels. The stride of subsequent basic units in each stage is 1, and the feature map and the number of channels remain unchanged. In addition, the channel shuffle operation is added to each ShuffleNetV1 basic unit to reorganize the feature map by channel after group convolution, so that information can be transmitted between different groups.

# Dataset

Dataset used: ImageNet

- Dataset size: 146 GB, 1330,000 color images of 1000 classes
    - Training set: 140 GB, 1280,000 images
    - Test set: 6 GB, 50,000 images
- Data format: RGB
    - Note: Data will be processed in **src/dataset.py**.

# Environment Requirements

- Hardware
    - Set up the hardware environment with Ascend AI Processors.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/api_python/mindspore.html)

# Script Description

## Script and Sample Code

```shell
├─shufflenetv1
  ├─README.md                              # ShuffleNetV1 description
  ├─cpp_infer                              # C++ inference
  ├─scripts
    ├─run_standalone_train.sh                 # Script for single-device training in the Ascend environment
    ├─run_standalone_train_gpu.sh             # Script for single-device training in the GPU environment
    ├─run_distribute_train.sh                 # Script for 8-device training in the Ascend environment
    ├─run_distribute_train_gpu.sh             # Script for 8-device training in the GPU environment
    ├─run_eval.sh                             # Script for evaluation in the Ascend environment
    ├─run_eval_gpu.sh                             # Script for evaluation in the GPU environment
    ├─run_infer_cpp.sh                        # Shell script for inference on Ascend 310 AI Processors
    ├─run_onnx.sh                             # Shell script for ONNX inference
  ├─src
    ├─dataset.py                              # Data preprocessing
    ├─shufflenetv1.py                         # Network model definition
    ├─crossentropysmooth.py                   # Loss function definition
    ├─lr_generator.py                         # Learning rate generation function
    ├──model_utils
      ├──config.py                            # Parameter configuration
      ├──device_adapter.py                    # Device-related information
      ├──local_adapter.py                     # Device-related information
      ├──moxing_adapter.py                    # Decorator (mainly used to copy ModelArts data)
  ├─default_config.yaml                       # Parameter file
  ├─gpu_default_config.yaml                   # GPU parameter file
  ├─train.py                                  # Network training script
  ├─export.py                                 # Script for converting the model format
  ├─eval.py                                   # Network evaluation script
  ├─mindspore_hub_conf.py                     # Hub configuration script
  ├─postprogress.py                           # Script for post-processing after inference on Ascend 310 AI Processors
  ├─transfer_config.yaml                      # Transfer learning parameter file
  ├─transfer_dataset_process.py               # Transfer learning dataset processing script
  ├─mindspore_quick_start.ipynb               # Transfer learning and inference visualization script
  ├─infer_shufflenetv1_onnx.py                # Script of ONNX model inference
  ├─requirements.txt                          # Requirements of third party package
```

## Script Parameters

The parameters used during model training and evaluation can be set in the **default_config.yaml** file.

```default_config.yaml
'epoch_size': 250,                  # Number of epochs
'keep_checkpoint_max': 5,           # Maximum number of CKPT files that can be saved
'save_ckpt_path': "./",       # Path for storing CKPT files
'save_checkpoint_epochs': 1,        # Number of epochs for saving a CKPT file
'save_checkpoint': True,            # Specifies whether CKPT files are saved.
'amp_level': 'O3',                  # Training accuracy
'batch_size': 128,                  # Batch size
'num_classes': 1000,                # Number of dataset classes
'label_smooth_factor': 0.1,         # Label smoothing factor
'lr_decay_mode': "cosine",          # Learning rate decay mode
'lr_init': 0.00,                    # Initial learning rate
'lr_max': 0.50,                     # Maximum learning rate
'lr_end': 0.00,                     # Minimum learning rate
'lr_end': 0.00,                     # Number of warmup epochs
'loss_scale': 1024,                 # Loss scale
'weight_decay': 0.00004,            # Weight decay rate
'momentum': 0.9                     # Momentum parameter
```

For more information, see `default_config.yaml` for Ascend and `gpu_default_config.yaml` for GPU.

## Training Process

### Startup

You can use Python or shell scripts for training.

```shell
# Training example
- running on Ascend with default parameters

  python:
      Ascend AI Processor-based single-device training: python train.py --train_dataset_path [DATA_DIR]
      # example: python train.py --train_dataset_path /home/DataSet/ImageNet_Original/train > log.txt 2>&1 &

  shell:
      Ascend AI Processor-based 8-device parallel training: bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_DIR]
      # example: bash scripts/run_distribute_train.sh ~/hccl_8p.json /home/DataSet/ImageNet_Original/train

      Ascend AI Processor-based single-device training: bash scripts/run_standalone_train.sh [DEVICE_ID] [DATA_DIR]
      # example: bash scripts/run_standalone_train.sh 0 /home/DataSet/ImageNet_Original/train

- running on GPU with gpu default parameters

  python:
      GPU-based single-device training: python train.py --config_path [CONFIG_PATH] --device_target [DEVICE_TARGET] --train_dataset_path [TRAIN_DATA_DIR]
      Ascend AI Processor-based 8-device training:
          export RANK_SIZE=8
          mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
          python train.py --config_path [CONFIG_PATH] \
                          --train_dataset_path [TRAIN_DATA_DIR] \
                          --is_distributed=True \
                          --device_target=GPU > log.txt 2>&1 &

  shell:
      GPU-based single-device training: bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [DATA_DIR]
      GPU-based 8-device training: bash scripts/run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR]

- running transfer learning on CPU with default parameters

  python:
      CPU-based training: python train.py --config_path=./transfer_config.yaml > log.txt 2>&1 &
```

  For distributed training, you need to create an HCCL configuration file in JSON format in advance.

  Follow the instructions in the following link:

  [Link](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)

### Results

The CKPT file is stored in the `save_ckpt_path` directory, and training logs are recorded in the `log.txt` directory. An example of a training log is as follows:

```log
epoch time: 99854.980, per step time: 79.820, avg loss: 4.093
epoch time: 99863.734, per step time: 79.827, avg loss: 4.010
epoch time: 99859.792, per step time: 79.824, avg loss: 3.869
epoch time: 99840.800, per step time: 79.809, avg loss: 3.934
epoch time: 99864.092, per step time: 79.827, avg loss: 3.442
```

## Evaluation Process

### Startup

You can use Python or shell scripts for evaluation.

```shell
# Ascend-based evaluation example
  python:
      python eval.py --eval_dataset_path [DATA_DIR] --ckpt_path [PATH_CHECKPOINT]
      # example: python eval.py --eval_dataset_path /home/DataSet/ImageNet_Original/validation_preprocess --ckpt_path /home/model/shufflenetv1/ckpt/shufflenetv1-250_1251 > eval_log.txt 2>&1 &

  shell:
      bash scripts/run_eval.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]
      # example: bash scripts/run_eval.sh 0 /home/DataSet/ImageNet_Original/validation_preprocess /home/model/shufflenetv1/ckpt/shufflenetv1-250_1251

# GPU-based evaluation example
  python:
      python eval.py --config_path [CONFIG_PATH] --eval_dataset_path [DATA_DIR] --ckpt_path [PATH_CHECKPOINT]

  shell:
      sh scripts/run_eval_gpu.sh [DEVICE_ID] [DATA_DIR] [PATH_CHECKPOINT]

# CPU-based transfer training evaluation example
  python:
      python eval.py --config_path=./transfer_config.yaml > eval_log.txt 2>&1 &
```

### Results

You can view the evaluation results in `eval_log.txt`.

```log
result:{'Loss': 2.0479587888106323, 'Top_1_Acc': 0.7385817307692307, 'Top_5_Acc': 0.9135817307692308}, ckpt:'/home/shufflenetv1/train_parallel0/checkpoint/shufflenetv1-250_1251.ckpt', time: 98560.63866615295
```

- If you want to train a model on ModelArts, perform model training and inference by referring to the [ModelArts official guide](https://support.huaweicloud.com/modelarts/). The procedure is as follows:

    ```ModelArts
    #  Example of using distributed training on ModelArts:
    #  Dataset structure

    #  ├── ImageNet_Original         # dir
    #    ├── train                   # train dir
    #       ├── train_dataset        # train_dataset dir
    #       ├── train_predtrained    # Pre-trained model directory
    #    ├── eval                    # Evaluation directory
    #       ├── eval_dataset         # Validation dataset directory
    #       ├── checkpoint           # CKPT directory

    # (1) Perform step a (modifying parameters in the YAML file) or b (create a training job and modify parameters on ModelArts).
    #       a. Set enable_modelarts to True.
    #          Set is_distributed to True.
    #          Set save_ckpt_path to /cache/train/outputs_imagenet/.
    #          Set train_dataset_path to /cache/data/train/train_dataset/.
    #          Set resume to /cache/data/train/train_predtrained/pred file name if pre-training weight is not set (resume="").

    #       b. Set enable_modelarts to True on the ModeArts page.
    #          Set the parameters required by method a on the ModelArts page.
    #          Note: Paths do not need to be enclosed in quotation marks.

    # (2) Set the path of the network configuration file _config_path to /The path of config in default_config.yaml/.
    # (3) Set the code path /path/shufflenetv1 on the ModelArts page.
    # (4) Set the boot file train.py of the model on the ModelArts page.
    # (5) On the ModelArts page, set the model data path to .../ImageNet_Original (path of the ImageNet_Original directory).
    # Output file path and job log path of the model
    # (6) Start model training.

    # Example of model inference on ModelArts
    # (1) Place the trained model to the corresponding position in the bucket.
    # (2) Perform step a or b.
    #        a. Set enable_modelarts to True.
    #          Set eval_dataset_path to /cache/data/eval/eval_dataset/.
    #          Set ckpt_files to /cache/data/eval/checkpoint/ckpt file.

    #       b. Set enable_modelarts to True on the ModeArts page.
    #          Set the parameters required by method a on the ModelArts page.
    #          Note: Paths do not need to be enclosed in quotation marks.

    # (3) Set the path of the network configuration file _config_path to /The path of config in default_config.yaml/.
    # (4) Set the code path /path/shufflenetv1 on the ModelArts page.
    # (5) Set the boot file eval.py of the model on the ModelArts page.
    # (6) On the ModelArts page, set the model data path to .../ImageNet_Original (path of the ImageNet_Original directory).
    # Output file path and job log path of the model
    # (7) Start model inference.
    ```

## Export Process

### Export

```shell
python export.py --ckpt_path [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format [EXPORT_FORMAT] --batch_size [BATCH_SIZE]
```

`EXPORT_FORMAT`: ["AIR", "MINDIR"]

- Export MindIR on Modelarts

    ```ModelArts
    Export MindIR example on ModelArts
    Data storage method is the same as training
    # (1) Perform step a (modifying parameters in the YAML file) or b (creating a training job and modifying parameters on ModelArts).
    #       a. Set enable_modelarts to True.
    #          Set file_name to shufflenetv1.
    #          Set file_format to MINDIR.
    #          Set ckpt_file to /cache/data/checkpoint file name.

    #       b. Set enable_modelarts to True on the ModelArts page.
    #          Set the parameters required by method a on the ModelArts page.
    #          Note: Paths do not need to be enclosed in quotation marks.
    # (2) Set the path of the network configuration file _config_path to /The path of config in default_config.yaml/.
    # (3) Set the code path /path/shufflenetv1 on the ModelArts page.
    # (4) Set the boot file export.py of the model on the ModelArts page.
    # (5) On the ModelArts page, set the model data path to .../ImageNet_Original/eval/checkpoint (path of the ImageNet_Original/eval/checkpoint directory).
    # Output file path and job log path of the model
    ```

## Inference Process

**Set environment variables before inference by referring to [MindSpore C++ Inference Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md).**

### Inference

Before inference, you need to export the model. AIR models can be exported only from the Ascend 910 AI Processor environment, and MINDIR models can be exported from any environment.

```shell
# Inference based on Ascend 310 AI Processors
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_FILE] [DEVICE_ID]
```

-Note: The ShuffleNetV1 network uses the ImageNet dataset. The label number starts from 0 after folders are sorted.

The inference results are saved in the current directory. You can find results similar to the following in the acc.log file:
The following shows the inference results of the DenseNet-121 network using ImageNet.

  ```log
  Top_1_Acc=73.85%, Top_5_Acc=91.526%
  ```

# Model Description

## Training Performance

| Parameter                       | Ascend                                | GPU                                |
| -------------------------- | ------------------------------------- | -------------------------- |
| Model name                   | ShuffleNetV1                           | ShuffleNetV1                           |
| Runtime environment                   | Ascend 910 AI Processor; EulerOS 2.8              | Tesla V100; EulerOS 2.8                           |
| Upload date                   | 2020-12-3                             | 2021-07-15                            |
| MindSpore version            | 1.0.0                                 | 1.3.0                                 |
| Dataset                     | imagenet                              | imagenet                              |
| Training parameters                   | default_config.yaml                    | gpu_default_config.yaml                |
| Optimizer                     | Momentum                              | Momentum                              |
| Loss function                   | SoftmaxCrossEntropyWithLogits         | SoftmaxCrossEntropyWithLogits         |
| Final loss                   | 2.05                                  | 2.04                                  |
| Accuracy (8-device)                | Top1[73.9%], Top5[91.4%]               | Top1[73.8%], Top5[91.4%]               |
| Total trainingduration (8-device)            | 7.0h                                    | 20.0h                                    |
| Total evaluation duration                 | 99s                                    | 58s                                    |
| Parameters (M)                | 44M                                   | 51.3M                                   |
| Script                      | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1)|

# Random Seed Description

We set random seeds in the `dataset.py` and `train.py` scripts.

# ModelZoo

For details, please visit the [official website](https://gitee.com/mindspore/models).

# Transfer Learning

## Transfer Learning Dataset

The transfer learning dataset will be divided into two parts: 80% for training and 20% for validation.

> You can download the dataset from the [dataset download page](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz).

The directory structure of the downloaded dataset files is as follows:

```text
./flower_photos/
├── daisy
├── dandelion
├── roses
├── sunflowers
└── tulips
```

The `create_flower_dataset()` function in the `transfer_dataset_process.py` script is used to obtain the split and preprocessed training set and validation set. Random seeds are set in the `transfer_dataset_process.py` script.

## Transfer Training

Transfer training parameters are set in `transfer_config.yaml`. Before transfer training, set the CKPT file path of the pre-trained model to `resume` and the dataset path to `dataset_path`.

> You can download the pre-trained CKPT file from the [CKPT download page](https://www.mindspore.cn/resources/hub/details?MindSpore/1.7/shufflenetv1_imagenet2012).

Run 100 epochs in the CPU-based environment and use ValAccMonitor to save the CKPT file with the highest inference accuracy on the validation set.

## Training Results

'Top_1_Acc': 0.94375
