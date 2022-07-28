# Contents

- [Autoslim Description](#autoslim-description)
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
    - [310 Inference Process](#310-inference-process)
        - [310 Inference](#310-inference)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Autoslim Description](#contents)

Based on the article universal slim networks and improved training technologies, Autoslim further extends universally slimmable networks to the field of neural network architecture search (NAS), and selects the optimal model under the performance evaluation standard through the search agent.

[Paper](https://arxiv.org/abs/1903.11728v1) ：Jiahui Yu, Thomas Huang."AutoSlim: Towards One-Shot Architecture Search for Channel Numbers".2019.

# [Dataset](#contents)

Dataset used: Imagenet2012, [here](https://image-net.org/download.php)

Dataset size 224*224 colorful images in 1000 classes

Train：1,281,167 images

Test： 50,000 images

Data format：jpeg

```bash
    - Note：Data will be processed in dataset.py
└─dataset
   ├─train                 # train dataset
   └─val                   # evaluate dataset
```

# [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# Install necessary package
pip install -r requirements.txt

# Ascend training
cd scripts
bash run_standalone_train_ascend.sh 0 /path/to/imagenet-1k

# Ascend training in parallel
cd scripts
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] /path/to/imagenet-1k

# Ascend evaluation
cd scripts
bash run_standalone_eval_gpu.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]

# start training on a GPU
cd scripts
bash run_standalone_train_ascend.sh 0 /path/to/imagenet-1k

# start training on GPUs in parallel
cd scripts
bash run_distribute_train_gpu.sh /path/to/imagenet-1k

# start evaling on a GPU
cd scripts
bash run_standalone_eval_gpu.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── cv
    ├── AutoSlim
        ├── README.md                    // descriptions about AutoSlim
        ├── README_CN.md                 // descriptions about AutoSlim in Chinese
        ├── requirements.txt             // package needed
        ├── ascend310_infer              // 310 inference source code
        ├── preprocess.py                // 310 inference preprocess
        ├── postprocess.py               // 310 inference postprocess
        ├── scripts
        │   ├──run_distribute_train_ascend.sh   // ascend training in parallel
        │   ├──run_distribute_train_gpu.sh  // gpu training in parallel
        │   ├──run_eval_ascend.sh           // ascend evaluation
        │   ├──run_eval_gpu.sh              // gpu evaluation
        │   ├──run_infer_310.sh             // 310 inference
        │   ├──run_standalone_train_ascend.sh   // ascend training
        │   └──run_standalone_train_gpu.sh  // gpu training
        ├── src
        │   ├──dataset.py                   // load dataset
        │   ├──lr_generator.py              // generate learning rate
        │   ├──autoslim_resnet.py           // backbone for training
        │   ├──autoslim_resnet_for_val.py   // backbone for evaling
        │   ├──slimmable_ops.py             // functions
        │   ├──config.py                    // use the parameters
        │   └──autoslim_cfg.py              // hyper parameters
        ├── train.py                     // training script
        ├── eval.py                      // evaluation script
        └── export.py                    // exportation script
```

## [Script Parameters](#contents)

```bash

# Major parameters in train.py and set_parser.py as follows:

--device_target:Device target, default is "Ascend"
--device_id:Device ID
--device_num:The number of device used
--dataset_path:Path where the dataset is saved
--batch_size:Training batch size.
--epoch_size:The number of training epoch
--save_checkpoint_path:Path where the ckpt is saved
--file_format:Model transformation format

```

## [Training Process](#contents)

### Training

```bash
python train.py --dataset_path=/path/to/imagenet-1

# or enter script dir, and run the script with single ascend
cd scripts
bash run_standalone_train_ascend.sh 0 /path/to/imagenet-1k

# or run the script with ascends in parallel
cd scripts
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] /path/to/imagenet-1k

# if you want to train model on a GPU, please run the script
cd scripts
bash run_standalone_train_ascend.sh 0 /path/to/imagenet-1k

# if you want to train model on GPUs in parallel, please run the script
cd scripts
bash run_distribute_train_ascend.sh /path/to/imagenet-1k
```

After training, the loss value will be achieved as follows:

```bash
============== Starting Training ==============
epoch: 1 step: 100, loss is 6.9037023
epoch: 1 step: 200, loss is 6.9010477
epoch: 1 step: 300, loss is 6.895539

...

epoch: 94 step: 535, loss is 1.0588518
epoch: 94 step: 635, loss is 0.923507

...
```

The model checkpoint will be saved in [SAVE_CKPT], which has been designated in the script.

## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```bash
python eval.py --dataset_path=/path/to/imagenet-1k --pretained_checkpoint_path=[PRETAINED_CHECKPOINT_PATH]
# or enter script dir, and run the script with a ascend
cd scripts
bash run_eval_ascend.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]

# if on a GPU, please run
cd scripts
bash run_eval_gpu.sh 0 /path/to/imagenet-1k [PRETAINED_CHECKPOINT_PATH]
```

The accuracy of the test dataset is as follows:

```bash
Start loading model.
Start evaluating.
Accuracy = 0.685
```

## [310 Inference Process](#contents)

### 310 Inference

Before 310 inference, exported mindir file is needed.

```bash
python export.py --pretained_checkpoint_path=[PRETAINED_CHECKPOINT_PATH] --export_model_name=[EXPORT_MODEL_NAME] --file_format=[FILE_FORMAT]
```

For 310 inference, please run this script as follows:

```bash
cd scripts
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [DEVICE_ID]
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters            | AutoSlim  |
| ------------------ | -------------------|
| Resource          | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G;OS CentOS8.2；GPU V100； |
| uploaded Date    | 01/03/2022 (month/day/year)       |
| MindSpore Version    | 1.3.0           |
| Dataset      |  Imagenet-1k   |
| Training Parameters   | epoch = 100, batch_size = 256, lr_max=0.1  momentum=0.9  weight_decay=1e-4  |
| Optimizer     | SGD  |
| Loss Function   |  SoftmaxCrossEntropy |
| outputs      | probability     |
| Speed      | 142ms/step    |

## [Description of Random Situation](#contents)

In train.py, we use "dataset.Generator(shuffle=True)" to shuffle dataset.

## [ModelZoo Homepage](#contents)  

Please check the official [homepage](https://gitee.com/mindspore/models).

## FAQ

Please refer to [ModelZoo FAQ](https://gitee.com/mindspore/models/blob/master/README.md#faq) to get some common FAQ.

- **Q**: Get "out of memory" error in PYNATIVE_MODE.

  **A**: You can set smaller batch size, e.g. 32, 16.  