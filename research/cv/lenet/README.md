# Contents

- [Contents](#contents)
    - [LeNet Description](#lenet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Apply algorithm in MindSpore Golden Stick](#apply-algorithm-in-mindspore-golden-stick)
        - [Training Process](#training-process-1)
            - [Running on GPU](#running-on-gpu-1)
        - [Evaluation Process](#evaluation-process-1)
            - [Running on GPU](#running-on-gpu-2)
            - [Result](#result-3)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference Performance](#inference-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [LeNet Description](#contents)

LeNet was proposed in 1998, a typical convolutional neural network. It was used for digit recognition and got big success.

[Paper](https://ieeexplore.ieee.org/document/726791): Y.Lecun, L.Bottou, Y.Bengio, P.Haffner. Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*. 1998.

## [Model Architecture](#contents)

LeNet is very simple, which contains 5 layers. The layer composition consists of 2 convolutional layers and 3 fully connected layers.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Train：60,000 images  
    - Test：10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py

- The directory structure is as follows:

```bash
└─Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend, GPU, or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# enter script dir, train LeNet
bash run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]
# example: bash run_standalone_train_ascend.sh /home/DataSet/MNIST/ ./ckpt/

# enter script dir, evaluate LeNet
bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
# example: bash run_standalone_eval_ascend.sh /home/DataSet/MNIST/ /home/model/lenet/ckpt/checkpoint_lenet-1_1875.ckpt
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ckpt_path='/cache/train/'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_path='/cache/data'" on the website UI interface.
    #          Add "ckpt_path='/cache/train/'" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code.
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Upload the original mnist_data dataset to S3 bucket.
    # (5) Set the code directory to "/path/lenet" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "ckpt_file='/cache/train/checkpoint_lenet-10_1875.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_path='/cache/data'" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "ckpt_file='/cache/train/checkpoint_lenet-10_1875.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code.
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Upload the original mnist_data dataset to S3 bucket.
    # (5) Set the code directory to "/path/lenet" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. The evaluation steps using ModelArts are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='lenet'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='lenet'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/lenet" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── cv
    ├── lenet
        ├── README.md                    // descriptions about lenet
        ├── requirements.txt             // package needed
        ├── ascend310_infer              // application for 310 inference
        ├── scripts
        │   ├──run_infer_310.sh                        // infer in 310
        │   ├──run_standalone_train_cpu.sh             // train in cpu
        │   ├──run_standalone_train_gpu.sh             // train in gpu
        │   ├──run_standalone_train_ascend.sh          // train in ascend
        │   ├──run_standalone_eval_cpu.sh             //  evaluate in cpu
        │   ├──run_standalone_eval_gpu.sh             //  evaluate in gpu
        │   ├──run_standalone_eval_ascend.sh          //  evaluate in ascend
        ├── src
        │   ├──aipp.cfg               // aipp config
        │   ├──dataset.py             // creating dataset
        │   ├──lenet.py               // lenet architecture
        │   └──model_utils
        │       ├──config.py               // Processing configuration parameters
        │       ├──device_adapter.py       // Get cloud ID
        │       ├──local_adapter.py        // Get local ID
        │       └──moxing_adapter.py       // Parameter processing
        ├── default_config.yaml            // Training parameter profile(ascend)
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── postprocess.py        //  postprocess script
        ├── preprocess.py        //  preprocess script
```

## [Script Parameters](#contents)

```default_config.yaml
default_config.yaml as follows:

--data_path: The absolute full path to the train and evaluation datasets.
--epoch_size: Total training epochs.
--batch_size: Training batch size.
--image_height: Image height used as input to the model.
--image_width: Image width used as input the model.
--device_target: Device where the code will be implemented. Optional values
                 are "Ascend", "GPU", "CPU".
--checkpoint_path: The absolute full path to the checkpoint file saved
                   after training.
--data_path: Path where the dataset is saved
```

## [Training Process](#contents)

### Training

```bash
python train.py --data_path Data --ckpt_path ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
bash run_standalone_train_ascend.sh [DATA_PATH] [CKPT_SAVE_PATH]
# example: bash run_standalone_train_ascend.sh /home/DataSet/MNIST/ ./ckpt/
```

After training, the loss value will be achieved as follows:

```bash
# grep "loss is " log.txt
epoch: 1 step: 1, loss is 2.2791853
...
epoch: 1 step: 1536, loss is 1.9366643
epoch: 1 step: 1537, loss is 1.6983616
epoch: 1 step: 1538, loss is 1.0221305
...
```

The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
python eval.py --data_path Data --ckpt_path ckpt/checkpoint_lenet-1_1875.ckpt > log.txt 2>&1 &  
# or enter script dir, and run the script
bash run_standalone_eval_ascend.sh [DATA_PATH] [CKPT_NAME]
# example: bash run_standalone_eval_ascend.sh /home/DataSet/MNIST/ /home/model/lenet/ckpt/checkpoint_lenet-1_1875.ckpt
```

You can view the results through the file "log.txt". The accuracy of the test dataset will be as follows:

```bash
# grep "Accuracy: " log.txt
'Accuracy': 0.9842
```

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### Export MindIR

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_size can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [DEVICE_ID]
```

- `DVPP` is mandatory, and must choose from ["DVPP", "CPU"], it's case-insensitive.The size of the picture that Lenet performs inference is [32, 32], the DVPP hardware limits the width of divisible by 16, and the height is divisible by 2. The network conforms to the standard, and the network can pre-process the image through DVPP.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'Accuracy': 0.9843
```

# Apply algorithm in MindSpore Golden Stick

MindSpore Golden Stick is a compression algorithm set for MindSpore. We usually apply algorithm in Golden Stick before training for smaller model size, lower power consuming or faster inference process.
MindSpore Golden Stick provides SimQAT algorithm for Lenet5. SimQAT is a quantization-aware training algorithm that trains the quantization parameters of certain layers in the network by introducing fake-quantization nodes, so that the model can perform inference with less power consumption or higher performance during the deployment phase.

## Training Process

### Running on GPU

```text
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'train.py'.
bash run_standalone_train_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CKPT_TYPE](optional) [CKPT_PATH](optional)

# standalone training example, apply SimQAT and train from beginning
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/lenet_mnist_config.yaml /path/to/dataset

# standalone training example, apply SimQAT and train from full precision checkpoint
cd ./golden_stick/scripts/
bash run_standalone_train_gpu.sh ../quantization/simqat/ ../quantization/simqat/lenet_mnist_config.yaml /path/to/dataset FP32 /path/to/fp32_ckpt
```

## Evaluation Process

### Running on GPU

```text
# evaluation
cd ./golden_stick/scripts/
# PYTHON_PATH represents path to directory of 'eval.py'.
bash run_eval_gpu.sh [PYTHON_PATH] [CONFIG_FILE] [DATASET_PATH] [CHECKPOINT_PATH]

# evaluation example
cd ./golden_stick/scripts/
bash run_eval_gpu.sh ../quantization/simqat/ ../quantization/simqat/lenet_mnist_config.yaml /path/to/dataset ./checkpoint/lenet-10.ckpt
```

### Result

Evaluation result will be stored in the example path, whose folder name is "eval". Under this, you can find result like the following in log.

- Apply SimQAT on Lenet5, and evaluating with MNIST dataset

```bash
================ {'Accuracy': 0.9907852564102564} ================
```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | LeNet                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8              |
| uploaded Date              | 07/05/2021 (month/day/year)                                 |
| MindSpore Version          | 1.3.0                                                      |
| Dataset                    | MNIST                                                    |
| Training Parameters        | epoch=10, steps=1875, batch_size = 32, lr=0.01              |
| Optimizer                  | Momentum                                                         |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Loss                       | 0.002                                                      |
| Speed                      | 1.071 ms/step                          |
| Total time                 | 32.1s                          |                                       |
| Checkpoint for Fine tuning | 482k (.ckpt file)                                         |
| Scripts                    | [LeNet Script](https://gitee.com/mindspore/models/tree/r2.0/official/cv/MobileNet/mobilenetv2)s |

#### Inference Performance

| Parameters          | Ascend                      |
| ------------------- | --------------------------- |
| Model Version       | LeNet                       |
| Resource            | Ascend 310; CentOS 3.10     |
| Uploaded Date       | 07/05/2021 (month/day/year) |
| MindSpore Version   | 1.2.0                       |
| Dataset             | Mnist                       |
| batch_size          | 1                           |
| outputs             | Accuracy                    |
| Accuracy            | Accuracy=0.9843             |
| Model for inference | 482K(.ckpt file)            |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

## [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/models).  
