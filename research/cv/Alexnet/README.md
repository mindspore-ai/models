# Contents

- [Contents](#contents)
    - [AlexNet Description](#alexnet-description)
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
            - [ONNX Evaluation](#onnx-evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [Result](#result)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Evaluation Performance](#evaluation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [AlexNet Description](#contents)

AlexNet was proposed in 2012, one of the most influential neural networks. It got big success in ImageNet Dataset recognition than other models.

[Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf): Krizhevsky A, Sutskever I, Hinton G E. ImageNet Classification with Deep ConvolutionalNeural Networks. *Advances In Neural Information Processing Systems*. 2012.

## [Model Architecture](#contents)

AlexNet composition consists of 5 convolutional layers and 3 fully connected layers. Multiple convolutional kernels can extract interesting features in images and get more accurate classification.

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- Dataset size：175M，60,000 32*32 colorful images in 10 classes
    - Train：146M，50,000 images
    - Test：29.3M，10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

```cifar10
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py
- Download the dataset, the directory structure is as follows:

 ```bash
└─dataset
    ├─train                 # train dataset
    └─validation_preprocess # evaluate dataset
```

## [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train AlexNet
bash run_standalone_train_ascend.sh [cifar10|imagenet] [DATA_PATH] [DEVICE_ID] [CKPT_PATH]
# example: bash run_standalone_train_ascend.sh cifar10 /home/DataSet/Cifar10/cifar-10-batches-bin/ 0 /home/model/alexnet/ckpt/

# enter script dir, evaluate AlexNet
bash run_standalone_eval_ascend.sh [cifar10|imagenet] [DATA_PATH] [CKPT_NAME] [DEVICE_ID]
# example: bash run_standalone_eval_ascend.sh cifar10 /home/DataSet/cifar10/cifar-10-verify-bin /home/model/cv/alxnet/ckpt/checkpoint_alexnet-1_1562.ckpt 0
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 30" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "dataset_path=/cache/data" on the website UI interface.
    #          Add "epoch_size: 30" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/alexnet" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set "epoch_size: 30" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add "epoch_size: 30" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Perform a or b. (suggested option a)
    #       a. zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/alexnet" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "checkpoint_url='s3://dir_to_your_trained_model/'" on base_config.yaml file.
    #          Set "checkpoint='./alexnet/alexnet_trained.ckpt'" on default_config.yaml file.
    #          Set "dataset_path='/cache/data'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_url='s3://dir_to_your_trained_model/'" on the website UI interface.
    #          Add "checkpoint='./alexnet/alexnet_trained.ckpt'" on the website UI interface.
    #          Add "dataset_path='/cache/data'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Perform a or b. (suggested option a)
    #       a. First, zip MindRecord dataset to one zip file.
    #          Second, upload your zip dataset to S3 bucket.
    #       b. Upload the original dataset to S3 bucket.
    #           (Data set conversion occurs during training process and costs a lot of time. it happens every time you train.)
    # (5) Set the code directory to "/path/alexnet" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start evaluating as follows)

1. Export s8 multiscale and flip with voc val dataset on modelarts, evaluating steps are as follows:

    ```python
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on base_config.yaml file.
    #          Set "file_name='alexnet'" on base_config.yaml file.
    #          Set "file_format='MINDIR'" on base_config.yaml file.
    #          Set "checkpoint_url='/The path of checkpoint in S3/'" on beta_config.yaml file.
    #          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on base_config.yaml file.
    #          Set other parameters on base_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "file_name='alexnet'" on the website UI interface.
    #          Add "file_format='MINDIR'" on the website UI interface.
    #          Add "checkpoint_url='/The path of checkpoint in S3/'" on the website UI interface.
    #          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Upload or copy your trained model to S3 bucket.
    # (3) Set the code directory to "/path/alexnet" on the website UI interface.
    # (4) Set the startup file to "export.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
    ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── cv
    ├── alexnet
        ├── README.md                                // descriptions about alexnet
        ├── requirements.txt                         // package needed
        ├── scripts
        │   ├──run_standalone_train_gpu.sh           // train in gpu
        │   ├──run_standalone_train_ascend.sh        // train in ascend
        │   ├──run_standalone_eval_gpu.sh            //  evaluate in gpu
        │   ├──run_standalone_eval_ascend.sh         //  evaluate in ascend
        |   ├──run_onnx_eval.sh                      // evaluate using onnx model
        ├── src
        │   ├──dataset.py                            // creating dataset
        │   ├──alexnet.py                            // alexnet architecture
        │   └──model_utils
        │       ├──config.py                         // Processing configuration parameters
        │       ├──device_adapter.py                 // Get cloud ID
        │       ├──local_adapter.py                  // Get local ID
        │       └──moxing_adapter.py                 // Parameter processing
        ├── default_config.yaml                      // Training parameter profile(cifar10)
        ├── config_imagenet.yaml                     // Training parameter profile(imagenet)
        ├── train.py                                 // training script
        ├── eval.py                                  // evaluation script
        ├── eval_onnx.py                             // evaluation script for onnx model
```

### [Script Parameters](#contents)

```python
Major parameters in train.py and config.py as follows:

--data_path: The absolute full path to the train and evaluation datasets.
--epoch_size: Total training epochs.
--batch_size: Training batch size.
--image_height: Image height used as input to the model.
--image_width: Image width used as input the model.
--device_target: Device where the code will be implemented. Optional values are "Ascend", "GPU".
--checkpoint_path: The absolute full path to the checkpoint file saved after training.
--data_path: Path where the dataset is saved
```

### [Training Process](#contents)

#### Training

- Running on Ascend

  ```bash
  python train.py --config_path default_config.yaml --data_path cifar-10-batches-bin --ckpt_path ckpt > log 2>&1 &
  # or enter script dir, and run the script
  bash run_standalone_train_ascend.sh cifar10 /home/DataSet/Cifar10/cifar-10-batches-bin/ 0 /home/model/alexnet/ckpt/
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.2791853
  ...
  epoch: 1 step: 1536, loss is 1.9366643
  epoch: 1 step: 1537, loss is 1.6983616
  epoch: 1 step: 1538, loss is 1.0221305
  ...
  ```

  The model checkpoint will be saved in the current directory.

- running on GPU

  ```bash
  python train.py --config_path default_config.yaml --device_target "GPU" --data_path cifar-10-batches-bin --ckpt_path ckpt > log 2>&1 &
  # or enter script dir, and run the script
  bash run_standalone_train_gpu.sh cifar10 cifar-10-batches-bin ckpt
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.3125906
  ...
  epoch: 30 step: 1560, loss is 0.6687547
  epoch: 30 step: 1561, loss is 0.20055409
  epoch: 30 step: 1561, loss is 0.103845775
  ```

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  python eval.py --config_path default_config.yaml --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-1_1562.ckpt > eval_log.txt 2>&1 &
  # or enter script dir, and run the script
  bash run_standalone_eval_ascend.sh cifar10 cifar-10-verify-bin ckpt/checkpoint_alexnet-1_1562.ckpt 0
  ```

  You can view the results through the file "eval_log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "Accuracy: " eval_log
  'Accuracy': 0.8832
  ```

- running on GPU

  ```bash
  python eval.py --config_path default_config.yaml --device_target "GPU" --data_path cifar-10-verify-bin --ckpt_path ckpt/checkpoint_alexnet-30_1562.ckpt > eval_log 2>&1 &
  # or enter script dir, and run the script
  bash run_standalone_eval_gpu.sh cifar10 cifar-10-verify-bin ckpt/checkpoint_alexnet-30_1562.ckpt 0
  ```

  You can view the results through the file "eval_log". The accuracy of the test dataset will be as follows:

  ```bash
  # grep "Accuracy: " eval_log
  'Accuracy': 0.88512
  ```

#### ONNX Evaluation

- Export your model to ONNX

  ```bash
  # when using cifar10 dataset
  python export.py --config_path default_config.yaml --ckpt_file /path/to/checkpoint.ckpt --file_name /path/to/exported.onnx --file_format ONNX --device_target CPU

  # when using imagenet dataset
  python export.py --config_path config_imagenet.yaml --ckpt_file /path/to/checkpoint.ckpt --file_name /path/to/exported.onnx --file_format ONNX --device_target CPU
  ```

- Run ONNX evaluation. Specify dataset type as "cifar10" or "imagenet"

  ```bash
  # when using cifar10 dataset
  python eval_onnx.py --config_path default_config.yaml --data_path cifar-10-verify-bin --file_name /path/to/exported.onnx --device_target GPU > eval_log 2>&1 &

  # when using imagenet dataset
  python eval_onnx.py --config_path config_imagenet.yaml --data_path imagenet_val --file_name /path/to/exported.onnx --device_target GPU > output.eval_onnx.log 2>&1 &
  ```

    - The above python command will run in the background, you can view the results through the file `output.eval_onnx.log`. You will get the accuracy as following:

    ```bash
    # when using cifar10 dataset
    # grep "accuracy" output.eval_onnx.log
    accuracy: 0.8952

    # when using the imagenet dataset
    top-1 accuracy: 0.5787
    top-5 accuracy: 0.8010
    ```

## [Inference Process](#contents)

### [Export MindIR](#contents)

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### [Infer](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TYPE] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" OR "AIR" model.
- `DATASET_NAME` specifies datasets used to infer. value can be chosen between 'cifar10' and 'imagenet2012', defaulted is 'cifar10'
- `DATASET_PATH` specifies path of cifar10 datasets
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n', if you choose y, the cifar10 dataset will be processed in bin format, the imagenet2012 dataset will generate label json file.
- `DEVICE_TYPE` can choose from [Ascend, GPU, CPU].
- `DEVICE_ID` is optional, default value is 0.

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'acc': 0.88772
```

- Running on [ModelArts](https://support.huaweicloud.com/modelarts/)

    ```bash
    # Train 8p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "distribute=True" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ckpt_path='/cache/train'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "distribute=True" on the website UI interface.
    #          Add "data_path=/cache/data" on the website UI interface.
    #          Add "ckpt_path=/cache/train" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Upload the original cifar10 dataset to S3 bucket.
    # (5) Set the code directory to "/path/alexnet" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Train 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ckpt_path='/cache/train'" on default_config.yaml file.
    #          (optional)Set "checkpoint_url='s3://dir_to_your_pretrained/'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_path=/cache/data" on the website UI interface.
    #          Add "ckpt_path=/cache/train" on the website UI interface.
    #          (optional)Add "checkpoint_url='s3://dir_to_your_pretrained/'" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your pretrained model to S3 bucket if you want to finetune.
    # (4) Upload the original cifar10 dataset to S3 bucket.
    # (5) Set the code directory to "/path/alexnet" on the website UI interface.
    # (6) Set the startup file to "train.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    #
    # Eval 1p with Ascend
    # (1) Perform a or b.
    #       a. Set "enable_modelarts=True" on default_config.yaml file.
    #          Set "data_path='/cache/data'" on default_config.yaml file.
    #          Set "ckpt_file='/cache/train/checkpoint_alexnet-30_1562.ckpt'" on default_config.yaml file.
    #          Set other parameters on default_config.yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "data_path=/cache/data" on the website UI interface.
    #          Add "ckpt_file=/cache/train/checkpoint_alexnet-30_1562.ckpt" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Prepare model code
    # (3) Upload or copy your trained model to S3 bucket.
    # (4) Upload the original cifar10 dataset to S3 bucket.
    # (5) Set the code directory to "/path/alexnet" on the website UI interface.
    # (6) Set the startup file to "eval.py" on the website UI interface.
    # (7) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (8) Create your job.
    ```

## [Model Description](#contents)

### [Performance](#contents)

#### Evaluation Performance

| Parameters                 | Ascend                                                      | GPU                                              | Ascend-8P                               |
| -------------------------- | ------------------------------------------------------------| -------------------------------------------------|------------------------------------------|
| Resource                   | Ascend 910; ARM CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 | Tesla V100-PCIE-32GB; X86_64 CPU Xeon 8180 2.50GHz, 112cores                                | Ascend 910 * 8; ARM CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |
| uploaded Date              | 14/04/2022 (day/month/year)                                 | 14/04/2022 (day/month/year)                      | 14/04/2022 (day/month/year)              |
| MindSpore Version          | 1.7.0                                                       | 1.7.0                                            | 1.7.0                                |
| Dataset                    | CIFAR-10                                                    | CIFAR-10                                         | ImageNet2012                             |
| Training Parameters        | epoch=30, steps=1562, batch_size = 32, lr=0.01             | epoch=30, steps=1562, batch_size = 32, lr=0.002  | epoch=150, steps=625, batch_size=256*8, lr=0.01 |
| Optimizer                  | Momentum                                                    | Momentum                                         | Momentum                                 |
| Loss Function              | Softmax Cross Entropy                                       | Softmax Cross Entropy                            | Softmax Cross Entropy                    |
| outputs                    | probability                                                 | probability                                      | probability                              |
| Loss                       | 0.01                                                        | 0.15                                             | 1.72                                     |
| Speed                      | 7.2 ms/step                                                 | 7.4 ms/step                                     | 60.9 ms/step                              |
| Total time                 | 6 mins                                                      | 6 mins                                          | 96 mins                                 |
| Checkpoint for Fine tuning | 428M (.ckpt file)                                           | 428M (.ckpt file)                                | 459M (.ckpt file)                        |
| Scripts                    | [AlexNet Script](https://gitee.com/mindspore/models/tree/master/research/cv/Alexnet) | [AlexNet Script](https://gitee.com/mindspore/models/tree/master/research/cv/Alexnet) | [AlexNet Script](https://gitee.com/mindspore/models/tree/master/research/cv/Alexnet) |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
