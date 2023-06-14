# Contents

- [Contents](#contents)
- [Tiny-DarkNet Description](#tiny-darknet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Procsee](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Tiny-DarkNet Description](#contents)

Tiny-DarkNet is a 16-layer image classification network model for the classic image classification data set ImageNet proposed by Joseph Chet Redmon and others. Tiny-DarkNet, as a simplified version of Darknet designed by the author to minimize the size of the model to meet the needs of users for smaller model sizes, has better image classification capabilities than AlexNet and SqueezeNet, and at the same time it uses only fewer model parameters than them. In order to reduce the scale of the model, the Tiny-DarkNet network does not use a fully connected layer, but only consists of a convolutional layer, a maximum pooling layer, and an average pooling layer.

For more detailed information on Tiny-DarkNet, please refer to the [official introduction.](https://pjreddie.com/darknet/tiny-darknet/)

# [Model Architecture](#contents)

Specifically, the Tiny-DarkNet network consists of 1×1 conv , 3×3 conv , 2×2 max and a global average pooling layer. These modules form each other to convert the input picture into a 1×1000 vector.

# [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.：
<!-- Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below. -->

<!-- Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)  -->

<!-- Dataset used ImageNet can refer to [paper](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- Dataset size: 125G, 1250k colorful images in 1000 classes
  - Train: 120G, 1200k images
  - Test: 5G, 50k images
- Data format: RGB images.
  - Note: Data will be processed in src/dataset.py  -->

Dataset used can refer to [paper](<https://ieeexplore.ieee.org/abstract/document/5206848>)

- Dataset size：125G，1250k colorful images in 1000 classes
    - Train: 120G,1200k images
    - Test: 5G, 50k images
- Data format: RGB images
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend/CPU/GPU）
    - Prepare hardware environment with Ascend/CPU processor/GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information,please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on Ascend：

  ```python
  # run in standalone environment
  cd scripts/
  bash run_standalone_train.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]

  # run in distribute environment
  cd scripts/
  bash run_distribute_train.sh [RANK_TABLE_FILE] [cifar10|imagenet] [TRAIN_DATA_DIR]

  # evaluation
  cd scripts/
  bash run_eval.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

  For distributed training, a hccl configuration file [RANK_TABLE_FILE] with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/r2.0/utils/hccl_tools.>

- running on GPU with gpu default parameters

  ```python
  # GPU standalone training example
  python train.py  \
  --config_path=./config/imagenet_config_gpu.yaml \
  --dataset_name=imagenet --train_data_dir=../dataset/imagenet_original/train --device_target=GPU
  # OR
  cd scripts/
  bash run_standalone_train_gpu.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]

  # GPU distribute training example
  export RANK_SIZE=8
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout  \
  python train.py  \
  --config_path=./config/imagenet_config_gpu.yaml \
  --dataset_name=imagenet \
  --train_data_dir=../dataset/imagenet_original/train \
  --device_target=GPU
  # OR
  cd scripts/
  bash run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR] [cifar10|imagenet]

  # GPU evaluation example
  python eval.py -device_target=GPU --val_data_dir=../dataset/imagenet_original/val --dataset_name=imagenet --config_path=./config/imagenet_config_gpu.yaml \
  --checkpoint_path=$PATH2
  # OR
  bash run_eval_gpu.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

- Running on ModelArts

  If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows.

    - Training with 8 cards on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/tinydarknet" on the website UI interface.
    # (4) Set the startup file to /{path}/tinydarknet/train.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/tinydarknet/imagenet_config.yaml.
    #         1. Set ”batch_size: 64“ (not necessary)
    #         2. Set ”enable_modelarts: True“
    #         3. Set ”modelarts_dataset_unzip_name: {filenmae}",if the data is uploaded in the form of zip package.
    #     b. adding on the website UI interface.
    #         1. Add ”batch_size=64“ (not necessary)
    #         2. Add ”enable_modelarts=True“
    #         3. Add ”modelarts_dataset_unzip_name={filenmae}",if the data is uploaded in the form of zip package.
    # (6) Upload the dataset or the zip package of dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of 8 cards.
    # (10) Create your job.
    ```

    - evaluating with single card on ModelArts

    ```python
    # (1) Upload the code folder to S3 bucket.
    # (2) Click to "create training task" on the website UI interface.
    # (3) Set the code directory to "/{path}/not necessary" on the website UI interface.
    # (4) Set the startup file to /{path}/not necessary/eval.py" on the website UI interface.
    # (5) Perform a or b.
    #     a. setting parameters in /{path}/not necessary/imagenet_config.yaml.
    #         1. Set ”enable_modelarts: True“
    #         2. Set “checkpoint_path: {checkpoint_path}”({checkpoint_path} Indicates the path of the weight file to be evaluated relative to the file 'eval.py', and the weight file must be included in the code directory.)
    #         3. Add ”modelarts_dataset_unzip_name: {filenmae}",if the data is uploaded in the form of zip package.
    #     b. adding on the website UI interface.
    #         1. Set ”enable_modelarts=True“
    #         2. Set “checkpoint_path={checkpoint_path}”({checkpoint_path} Indicates the path of the weight file to be evaluated relative to the file 'eval.py', and the weight file must be included in the code directory.)
    #         3. Add ”modelarts_dataset_unzip_name={filenmae}",if the data is uploaded in the form of zip package.
    # (6)  Upload the dataset or the zip package of dataset to S3 bucket.
    # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (there is only data or zip package under this path).
    # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
    # (9) Under the item "resource pool selection", select the specification of a single card.
    # (10) Create your job.
    ```

For more details, please refer the specify script.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash

├── tinydarknet
    ├── README.md                       // descriptions about Tiny-Darknet in English
    ├── README_CN.md                    // descriptions about Tiny-Darknet in Chinese
    ├── ascend310_infer                 // application for 310 inference
    ├── src
        ├── imagenet_config.yaml        // imagenet parameter configuration
        ├── imagenet_config_gpu.yaml    // imagenet parameter configuration for GPU
        ├── cifar10_config.yaml         // cifar10 parameter configuration
        ├── cifar10_config_gpu.yaml     // cifar10 parameter configuration for GPU
    ├── scripts
        ├── run_standalone_train.sh     // shell script for single on Ascend
        ├── run_standalone_train_gpu.sh // shell script for single on GPU
        ├── run_distribute_train.sh     // shell script for distributed on Ascend
        ├── run_distribute_train_gpu.sh // shell script for distributed on GPU
        ├── run_train_cpu.sh            // shell script for distributed on CPU
        ├── run_eval.sh                 // shell script for evaluation on Ascend
        ├── run_eval_cpu.sh             // shell script for evaluation on CPU
        ├── run_eval_gpu.sh             // shell script for evaluation on GPU
        ├── run_infer_310.sh            // shell script for inference on Ascend310
    ├── src
        ├── lr_scheduler                //learning rate scheduler
            ├── __init__.py             // init
            ├── linear_warmup.py        // linear_warmup
            ├── warmup_cosine_annealing_lr.py    // warmup_cosine_annealing_lr
            ├── warmup_step_lr.py       // warmup_step_lr
        ├── model_utils
            ├── config.py               // parsing parameter configuration file of "*.yaml"
            ├── device_adapter.py       // local or ModelArts training
            ├── local_adapter.py        // get related environment variables in local training
            └── moxing_adapter.py       // get related environment variables in ModelArts training
        ├── dataset.py                  // creating dataset
        ├── CrossEntropySmooth.py       // loss function
        ├── tinydarknet.py              // Tiny-Darknet architecture
    ├── train.py                        // training script
    ├── eval.py                         //  evaluation script
    ├── export.py                       // export checkpoint file into air/onnx
    ├── mindspore_hub_conf.py           // hub config
    ├── postprocess.py                  // postprocess script

```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `imagenet_config.yaml`

- config for Tiny-Darknet(only some parameters are listed)

  ```python
  pre_trained: False      # whether training based on the pre-trained model
  num_classes: 1000       # the number of classes in the dataset
  lr_init: 0.1            # initial learning rate
  batch_size: 128         # training batch_size
  epoch_size: 500         # total training epoch
  momentum: 0.9           # momentum
  weight_decay: 1e-4      # weight decay value
  image_height: 224       # image height used as input to the model
  image_width: 224        # image width used as input to the model
  train_data_dir: './ImageNet_Original/train/'  # absolute full path to the train datasets
  val_data_dir: './ImageNet_Original/val/'  # absolute full path to the evaluation datasets
  device_target: 'Ascend' # device running the program
  keep_checkpoint_max: 10 # only keep the last keep_checkpoint_max checkpoint
  checkpoint_path: '/train_tinydarknet.ckpt'  # the absolute full path to save the checkpoint file
  onnx_filename: 'tinydarknet.onnx' # file name of the onnx model used in export.py
  air_filename: 'tinydarknet.air'   # file name of the air/mindir model used in export.py
  lr_scheduler: 'exponential'     # learning rate scheduler
  lr_epochs: [70, 140, 210, 280]  # epoch of lr changing
  lr_gamma: 0.3            # decrease lr by a factor of exponential lr_scheduler
  eta_min: 0.0             # eta_min in cosine_annealing scheduler
  T_max: 150               # T-max in cosine_annealing scheduler
  warmup_epochs: 0         # warmup epoch
  is_dynamic_loss_scale: 0 # dynamic loss scale
  loss_scale: 1024         # loss scale
  label_smooth_factor: 0.1 # label_smooth_factor
  use_label_smooth: True   # label smooth
  ```

For more configuration details, please refer the script `imagenet_config.yaml`.

## [Training Process](#contents)

### [Training](#contents)

- running on Ascend：

  ```python
  cd scripts/
  bash run_standalone_train.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

  The command above will run in the background, you can view the results through the file train.log.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

  ```python
  # grep "loss is " train.log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

  The model checkpoint file will be saved in the current folder.
  <!-- The model checkpoint will be saved in the current directory.  -->

- running on GPU：

  ```python
  cd scripts/
  bash run_standalone_train_gpu.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

  The command above will run in the background, you can view the results through the file train.log.

  After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:
  <!-- After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows: -->

  ```python
  # grep "loss is " train.log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

- running on CPU

  ```python
  cd scripts/
  bash run_train_cpu.sh [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

### [Distributed Training](#contents)

- running on Ascend：

  ```python
  cd scripts/
  bash run_distribute_train.sh [RANK_TABLE_FILE] [cifar10|imagenet] [TRAIN_DATA_DIR]
  ```

  The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log. The loss value will be achieved as follows:

  ```python
  # grep "result: " train_parallel*/log
  epoch: 498 step: 1251, loss is 2.7798953
  Epoch time: 130690.544, per step time: 104.469
  epoch: 499 step: 1251, loss is 2.9261637
  Epoch time: 130511.081, per step time: 104.325
  epoch: 500 step: 1251, loss is 2.69412
  Epoch time: 127067.548, per step time: 101.573
  ...
  ```

- running on GPU：

  ```python
  cd scripts/
  bash run_standalone_train_gpu.sh [DEVICE_ID] [TRAIN_DATA_DIR] [cifar10|imagenet]
  ```

  The above shell script will run distribute training in the background. You can view the results through the file train_parallel[X]/log. The loss value will be achieved as follows:

  ```python
  # grep "result: " distribute_train_gpu/nohup.out
  epoch: 498 step: 1251, loss is 2.7825122
  epoch time: 200066.210 ms, per step time: 159.925 ms
  epoch: 499 step: 1251, loss is 2.799798
  epoch time: 199098.258 ms, per step time: 159.151 ms
  epoch: 500 step: 1251, loss is 2.8718748
  epoch time: 197784.661 ms, per step time: 158.101 ms
  ...
  ```

## [Evaluation Process](#contents)

### [Evaluation](#contents)

- evaluation on Imagenet dataset when running on Ascend:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "/username/tinydaeknet/train_tinydarknet.ckpt".

  ```python
  python eval.py  --val_data_dir=VAL_DATA_PATH --dataset_name=cifar10|imagenet --config_path=CONFIG_FILE --checkpoint_path=CHECKPOINT_PATH
  # OR
  bash run_eval.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5871979166666667, 'top_5_accuracy': 0.8175280448717949}
  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file. The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5871979166666667, 'top_5_accuracy': 0.8175280448717949}
  ```

- evaluation on Imagenet dataset when running on GPU:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "/username/tinydaeknet/train_tinydarknet.ckpt".

  ```python
  bash run_eval_gpu.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5896033653846153, 'top_5_accuracy': 0.8176482371794872}
  ```

  Note that for evaluation after distributed training, please set the checkpoint_path to be the last saved checkpoint file. The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_1_accuracy': 0.5896033653846153, 'top_5_accuracy': 0.8176482371794872}
  ```

- evaluation on cifar-10 dataset when running on CPU:

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "/username/tinydaeknet/train_tinydarknet.ckpt".

  ```python
  bash run_eval_cpu.sh [VAL_DATA_DIR] [cifar10|imagenet] [checkpoint_path]
  ```

  You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```python
  # grep "accuracy: " eval.log
  accuracy:  {'top_5_accuracy': 1.0, 'top_1_accuracy': 0.9829727564102564}
  ```

## [Inference process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Export MindIR

- Export on local

```shell
# Ascend310 inference
python export.py --dataset_name [DATASET] --file_name [FILE_NAME] --file_format [EXPORT_FORMAT]
```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/tinydarknet" on the website UI interface.
# (4) Set the startup file to /{path}/tinydarknet/export.py" on the website UI interface.
# (5) Perform a or b.
#     a. setting parameters in /{path}/tinydarknet/default_config.yaml.
#         1. Set ”enable_modelarts: True“
#         2. Set “checkpoint_path: ./{path}/*.ckpt”('checkpoint_path' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Set ”file_name: tinydarknet“
#         4. Set ”file_format：MINDIR“
#     b. adding on the website UI interface.
#         1. Add ”enable_modelarts=True“
#         2. Add “checkpoint_path=./{path}/*.ckpt”('checkpoint_path' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Add ”file_name=tinydarknet“
#         4. Add ”file_format=MINDIR“
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path (This step is useless, but necessary.).
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# You will see tinydarknet.mindir under {Output file path}.
```

The parameter does not have the ckpt_file option. Please store the ckpt file according to the path of the parameter `checkpoint_path` in `imagenet_config.yaml`.
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [LABEL_PATH] [DVPP] [DEVICE_ID]
```

- `LABEL_PATH` label.txt path. Write a py script to sort the category under the dataset, map the file names under the categories and category sort values,Such as[file name : sort value], and write the mapping results to the labe.txt file.
- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive.The size of the picture that MobilenetV2 performs inference is [224, 224], the DVPP hardware limits the width of divisible by 16, and the height is divisible by 2. The network conforms to the standard, and the network can pre-process the image through DVPP.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'top_1_accuracy': 59.07%, 'top_5_accuracy': 81.73%
```

# [Model Description](#contents)

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                        | Ascend                                                      | GPU                                                 |                       CPU                        |
| -------------------------- | ------------------------------------------------------------| ----------------------------------------------------|------------------------------------------------|
| Model Version                   | V1                                                          | V1                                                  |                       V1                      |
| Resource                        | Ascend 910；CPU 2.60GHz，56cores；内存 314G；系统 Euler2.8  | PCIE V100-32G                                    |                     CPU 72cores, Memory 503G               |
| Uploaded Date                   | 2020/12/22                                                  | 2021/07/15                         |                        2021/12/22                   |
| MindSpore Version              | 1.1.0                                                       | 1.3.0                                               |                        1.5.0                 |
| Dataset                     | 1200k images                                               | 1200k images                           |                       1200k图片                        |
| Training Parameters                   | epoch=500, steps=1251, batch_size=128, lr=0.1               | epoch=500, steps=1251, batch_size = 128, lr=0.005   |            epoch=500, steps=10009, batch_size=128, lr=0.1                |
| Optimizer                     | Momentum                                                    | Momentum                                            |                      Momentum                   |
| Loss Function                   | Softmax Cross Entropy                                       | Softmax Cross Entropy                               |                   Softmax Cross Entropy                  |
| Speed                       | 8pc: 104 ms/step                                            | 8pc: 255 ms/step                                          |                          1p：11081 ms/step                     |
| Parameters(M)                    | 4.0;                                                        | 4.0;                               |                             4.0;                       |
| Scripts                       | [Tiny-Darknet scripts](https://gitee.com/mindspore/models/tree/r2.0/research/cv/tinydarknet)

### [Evaluation Performance](#contents)

| Parameters          | Ascend                            | GPU                               |                              |
| ------------------- | ----------------------------------| ----------------------------------|------------------------------|
| Model Version       | V1                                | V1                                | V1                           |
| Resource            |  Ascend 910; Euler2.8             | PCIE V100-32G                     | CPU 72cores, Memory 503G     |
| Uploaded Date       | 2020/12/22                        | 2021/7/15                         | 2020/12/22                   |
| MindSpore Version   | 1.1.0                             | 1.3.0                             | 1.5.0                        |
| Dataset             | 200k images                       | 200k images                       | 200k images                  |
| batch_size          | 128                               | 128                               | 128                          |
| Outputs             | probability                       | probability                       | probability                  |
| Accuracy            | 8pcs Top-1: 58.7%; Top-5: 81.7%   | 8pcs Top-1: 58.9%; Top-5: 81.7%   | 1p: Top-1: 58.7%; Top-5:81.5 |
| Model for inference | 11.6M (.ckpt file)                | 10.06M (.ckpt file)               | 11.6M (.ckpt file)           |

### [Inference Performance](#contents)

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | TinyDarknet                 |
| Resource            | Ascend 310; Euler2.8        |
| Uploaded Date       | 29/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | ImageNet                    |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | Top-1: 59.07%; Top-5: 81.73%|
| Model for inference | 10.3M(.ckpt file)           |

# [ModelZoo Homepage](#contents)

 Please check the official[homepage](https://gitee.com/mindspore/models).  
