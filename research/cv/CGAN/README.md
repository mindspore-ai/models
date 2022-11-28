# Contents

- [Contents](#contents)
- [CGAN Description](#cgan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
    - [Training Script Parameters](#training-script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
    - [ONNX Inference](#onnx-inference)
        - [Export ONNX](#export-onnx)
        - [Infer ONNX](#infer-onnx)
    - [Model Export](#model-export)
    - [Ascend310 Inference Process](#ascend310-inference-process)
        - [Export MINDIR file](#export-mindir-file)
        - [Ascend310 Inference](#ascend310-inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Evaluation Index](#Evaluation-Index)
- [Description of Random Situation](#description-of-random-situation)
- [Model_Zoo Homepage](#model_zoo-homepage)

# [CGAN Description](#contents)

Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

[Paper](https://arxiv.org/pdf/1411.1784.pdf): Conditional Generative Adversarial Nets.

# [Model Architecture](#contents)

Architecture guidelines for Conditional GANs

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

# [Dataset](#contents)

Train CGAN Dataset used: [MNIST](<http://yann.lecun.com/exdb/mnist/>)

- Dataset size：52.4M，60,000 28*28 in 10 classes
    - Train：60,000 images  
    - Test：10,000 images
- Data format：binary files
    - Note：Data will be processed in dataset.py

```text

└─data
  └─MNIST_Data
    └─train
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─CGAN
  ├─README.md               # README
  ├─requirements.txt        # required modules
  ├─scripts                 # shell script
    ├─run_standalone_train_ascend.sh         # training in standalone mode(1pcs)
    ├─run_standalone_train_gpu.sh            # training in standalone mode(1pcs)
    ├─run_distributed_train_ascend.sh        # training in parallel mode(8 pcs)
    ├─run_distributed_train_gpu.sh           # training in parallel mode(8 pcs)
    ├─run_eval_ascend.sh    # evaluation
    ├─run_eval_gpu.sh       # evaluation
    ├─run_infer_onnx.sh     # onnxinference
    └─run_infer_310.sh      # 310inference
  ├─ src
    ├─dataset.py            # dataset create
    ├─cell.py               # network definition
    ├─ckpt_util.py          # utility of checkpoint
    ├─model.py              # discriminator & generator structure
    ├─parzen_numpy.py       # parzen estimation
    ├─tools.py              # useful tools
    └─reporter.py           # reporter the training process
  ├─ train.py               # train cgan
  ├─ eval.py                # eval cgan
  ├─ infer_onnx.py          # infer onnx_cgan
  ├─ export.py              # export mindir
  ├─ postprocess.py         # 310 postprocess
  ├─ preprocess.py          # 310 preprocess
  ├─ ascend310_infer        # 310 main

```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training on ascend or gpu
bash run_distributed_train_ascend.sh [DATA_PATH] [OUTPUT_PATH] [RANK_TABLE_FILE] [DEVICE_NUM]
# bash run_distributed_train_gpu.sh [DATA_PATH] [OUTPUT_PATH] [CUDA_VISIBLE_DEVICES] [DEVICE_NUM]

# standalone training on ascend or gpu
bash run_standalone_train_ascend.sh [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]
# bash run_standalone_train_gpu.sh [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]

# evaluating on ascend or gpu
bash run_eval_ascend.sh [G_CKPT] [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]
# bash run_eval_gpu.sh [G_CKPT] [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]
```

## [Training Process](#contents)

### [Training](#content)

- Run `run_standalone_train_ascend.sh` for non-distributed training of CGAN model.

```bash
# standalone training on ascend or gpu
bash run_standalone_train_ascend.sh /path/MNIST_Data/train /path/to/result  0
# bash run_standalone_train_gpu.sh /path/to/MNIST_Data/train /path/to/result  0
```

### [Distributed Training](#content)

- Run `run_distributed_train_ascend.sh` for distributed training of CGAN model.

```bash
bash run_distributed_train_ascend.sh /path/to/MNIST_Data/train /path/to/result /path/to/hccl_8p_01234567_127.0.0.1.json 8
# bash run_distributed_train_gpu.sh /path/to/MNIST_Data/train /path/to/result 0,1,2,3,4,5,6,7 8
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

### [Training Result](#content)

Training result will be stored in ` /path/to/result `.

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval_ascend.sh` for evaluation.

```bash
# eval
bash run_eval_ascend.sh /path/to/ckpt /path/to/MNIST_Data/train /path/to/result 0
# bash run_eval_gpu.sh /path/to/ckpt /path/to/MNIST_Data/train /path/to/result 0
```

- Run `run_onnx_eval_gpu.sh` for onnx evaluation.

```bash
# eval
bash run_onnx_eval_gpu.sh /path/to/onnx /path/to/MNIST_Data/train /path/to/result 0
# bash run_onnx_eval_gpu.sh /path/to/onnx /path/to/MNIST_Data/train /path/to/result 0
```

### [Evaluation Result](#content)

Evaluation result will be stored in the /path/to/result/eval/random_results. Under this, you can find generator result.

## [ONNX Inference](#contents)

### [Export ONNX](#content)

```bash
python  export.py --ckpt_dir [G_CKPT] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
```

### [Infer ONNX](#content)

```bash
bash run_infer_onnx.sh [ONNX_PATH] [DATA_PATH] [OUTPUT_PATH] [DEVICE_ID]
```

## [Model Export](#content)

```bash
python  export.py --ckpt_dir [G_CKPT] --device_target [DEVICE_TARGET] --file_format [FILE_FORMAT]
```

## [Ascend310 Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MINDIR file](#content)

```bash
python  export.py --ckpt_dir /path/to/train/ckpt/G_50.ckpt --device_target Ascend
```

### [Ascend310 Inference](#content)

- Run `run_infer_310.sh` for Ascend310 inference.

```bash
# infer
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

Ascend310 inference result will be stored in the postprocess_Result path. Under this, you can find generator result in result.png.

### [Generated example](#contents)

![generated_example](imgs/generated_ example.jpg)

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                       | GPU                                                          |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Model Version              | V1                                                           | V1                                                           |
| Resource                   | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G   | Ubuntu1 18.04.1; GPU RTX 3090; CPU 2.90GHz, 64cores; Memory, 256G |
| uploaded Date              | 07/04/2021 (month/day/year)                                  | 06/01/2022 (month/day/year)                                  |
| MindSpore Version          | 1.2.0                                                        | 1.6.1                                                        |
| Dataset                    | MNIST Dataset                                                | MNIST Dataset                                                |
| Training Parameters        | epoch=50,  batch_size = 128                                  | epoch=50,  batch_size = 128                                  |
| Optimizer                  | Adam                                                         | Adam                                                         |
| Loss Function              | BCELoss                                                      | BCELoss                                                      |
| Output                     | predict class                                                | predict class                                                |
| Loss                       | g_loss: 4.9693 d_loss: 0.1540                                | g_loss: 0.98 d_loss: 0.59                                    |
| Total time                 | 7.5 mins(8p)                                                 | 36 mins(1p);11.2mins(8p)                                     |
| Checkpoint for Fine tuning | 26.2M(.ckpt file)                                            | 5.7M (G_Net); 2.6M (D_Net)                                   |
| Scripts                    | [cgan script](https://gitee.com/mindspore/models/tree/master/research/cv/CGAN) | [cgan script](https://gitee.com/mindspore/models/tree/master/research/cv/CGAN) |

### [Evaluation Index](#contents)

|                       Model                        | Gaussian Parzen window log-likelihood estimate for the MNIST dataset |
| :------------------------------------------------: | :----------------------------------------------------------: |
|            Adversarial nets (original )            |                           225 ± 2                            |
|     Conditional adversarial nets  (original )      |                          132 ± 1.8                           |
| Conditional adversarial nets  (MindSpore Version ) |                        283.03 ± 2.15                         |

Note: , The higher the value, the better. For more information about Gaussian Parzen window log-likelihood, please refer to the [blog](https://blog.csdn.net/Nianzu_Ethan_Zheng/article/details/79211861).

# [Description of Random Situation](#contents)

We use random seed in train.py and cell.py for weight initialization.

# [Model_Zoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
