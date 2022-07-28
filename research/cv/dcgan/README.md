# Contents

- [DCGAN Description](#DCGAN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DCGAN Description](#contents)

The deep convolutional generative adversarial networks (DCGANs) first introduced CNN into the GAN structure, and the strong feature extraction ability of convolution layer was used to improve the generation effect of GAN.

[Paper](https://arxiv.org/pdf/1511.06434.pdf): Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[J]. Computer ence, 2015.

# [Model Architecture](#contents)

Architecture guidelines for stable Deep Convolutional GANs

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

# [Dataset](#contents)

Train DCGAN Dataset used: [Imagenet-1k](<http://www.image-net.org/index>)

- Dataset size: ~125G, 224*224 colorful images in 1000 classes
    - Train: 120G, 1281167 images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

```path

└─imagenet_original
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

```shell
.
└─dcgan
  ├─README.md                            # README
  ├─ ascend310_infer
      ├─inc
         └─utils.h                       # 310 inference header file
      ├─src
         ├─main.cc                       # 310 inference main file
         └─utils.cc                      # 310 inference utils file
      ├─build.sh                         # 310 inference build file
      └─CMakeLists.txt                   # 310 inference cmake file
  ├─ gpu_infer
      ├─inc
         └─utils.h                       # gpu inference header file
      ├─src
         ├─main.cc                       # gpu inference main file
         └─utils.cc                      # gpu inference utils file
      ├─build.sh                         # gpu inference build file
      └─CMakeLists.txt                   # gpu inference cmake file
  ├─scripts                              # shell script
    ├─run_standalone_train_ascend.sh     # training in standalone mode(1pc)
    ├─run_standalone_train_gpu.sh        # training in standalone mode(1pc)
    ├─run_distribute_train_ascend.sh     # training in parallel mode(8pcs)
    ├─run_distribute_train_gpu.sh        # training in parallel mode(8pcs)
    ├─run_eval_ascend.sh                 # evaluation on ascend
    ├─run_eval_gpu.sh                    # evaluation on gpu
    ├─run_infer_310.sh                   # infer on 310
    └─run_infer_gpu.sh                   # infer on gpu
  ├─ src
    ├─dataset.py                         # dataset create
    ├─cell.py                            # network definition
    ├─dcgan.py                           # dcgan structure
    ├─discriminator.py                   # discriminator structure
    ├─generator.py                       # generator structure
    └─config.py                          # config
 ├─ train.py                             # train dcgan
 ├─ eval.py                              # eval dcgan
 ├─ preprocess.py                        # preprocess on 310
 ├─ export.py                            # export checkpoint file
 ├─ verify.py                            # verify on 310
 └─ requirements.txt                     # requirements
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training on ascend
Usage: bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_URL] [TRAIN_URL]

# standalone training on ascend
Usage: bash run_standalone_train_ascend.sh [DEVICE_ID] [DATA_URL] [TRAIN_URL]

# distributed training on gpu
Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES] [DATA_URL] [TRAIN_URL]

# standalone training on gpu
Usage: bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_URL] [TRAIN_URL]
```

### [Parameters Configuration](#contents)

```txt
dcgan_imagenet_cfg {
    'num_classes': 1000,
    'epoch_size': 20,
    'batch_size': 128,
    'latent_size': 100,
    'feature_size': 64,
    'channel_size': 3,
    'image_height': 32,
    'image_width': 32,
    'learning_rate': 0.0002,
    'beta1': 0.5
}

dcgan_cifar10_cfg {
    'num_classes': 10,
    'ds_length': 60000,
    'batch_size': 100,
    'latent_size': 100,
    'feature_size': 64,
    'channel_size': 3,
    'image_height': 32,
    'image_width': 32,
    'learning_rate': 0.0002,
    'beta1': 0.5
}
```

- In order to conveniently store the files of the inference process,  batch_size of cifar10 is set to 100

## [Training Process](#contents)

- Set options in `config.py`, including learning rate, output filename and network hyperparameters. Click [here](https://www.mindspore.cn/tutorials/en/master/advanced/dataset.html) for more information about dataset.

### [Training](#content)

- Run `run_standalone_train.sh` for non-distributed training of DCGAN model.

```bash
# standalone training
Usage: bash run_standalone_train_ascend.sh [DEVICE_ID] [DATA_URL] [TRAIN_URL]
```

### [Distributed Training](#content)

- Run `run_distribute_train.sh` for distributed training of DCGAN model.

```bash
bash run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_URL] [TRAIN_URL]
```

- Notes
1. hccl.json which is specified by RANK_TABLE_FILE is needed when you are running a distribute task. You can generate it by using the [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools).

### [Training Result](#content)

Training result will be stored in save_path. You can find checkpoint file.

```bash
# standalone training result(1p)
Date time:  2021-04-13 13:55:39         epoch:  0 / 20         step:  0 / 10010        Dloss:  2.2297878       Gloss:  1.1530013
Date time:  2021-04-13 13:56:01         epoch:  0 / 20         step:  50 / 10010       Dloss:  0.21959287      Gloss:  20.064941
Date time:  2021-04-13 13:56:22         epoch:  0 / 20         step:  100 / 10010      Dloss:  0.18872623      Gloss:  5.872738
Date time:  2021-04-13 13:56:44         epoch:  0 / 20         step:  150 / 10010      Dloss:  0.53905165      Gloss:  4.477289
Date time:  2021-04-13 13:57:07         epoch:  0 / 20         step:  200 / 10010      Dloss:  0.47870708      Gloss:  2.2019134
Date time:  2021-04-13 13:57:28         epoch:  0 / 20         step:  250 / 10010      Dloss:  0.3929835       Gloss:  1.8170083
```

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run  the evaluation script.

```bash
# eval on ascend or gpu
bush run_eval_ascend.sh [IMG_URL] [CKPT_URL] [DEVICE_ID]
# bush run_eval_gpu.sh [IMG_URL] [CKPT_URL] [DEVICE_ID]
```

- Implement inference at Ascend310 or GPU platform.

```bash
# infer on ascend or gpu
bash run_infer_310.sh [MINDIR_PATH] [DATA_URL] [DEVICE_ID]
# bash run_infer_gpu.sh [MINDIR_PATH] [DATA_URL] [DEVICE_ID]
```

- Notes
1. A major contribution of the dcgan paper is to verify the capability of unsupervised representation learning with CNN, so we reproduce it on run_infer_310.sh or run_infer_gpu.sh.
2. The infer process requires environment variable to be set, such as LD_PRELOAD, PYTHONPATH, LD_LIBRARY_PATH in run_infer_gpu.sh.
3. 2.If you have the problem of `undefined reference to google::FlagRegisterer`, please refer to the [issue](#https://gitee.com/mindspore/mindspore/issues/I3X1EA).

### [Evaluation result](#content)

Evaluation result will be stored in the img_url path. Under this, you can find generator result in generate.png.

## Model Export

```shell
python export.py --ckpt_file [CKPT_PATH] --device_target [DEVICE_TARGET] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be "MINDIR"

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | Ascend                                                       | GPU                                                          |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Model Version              | V1                                                           | V1                                                           |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G              | RTX 3090; CPU 2.90GHz, 64cores; Memory, 256G                 |
| uploaded Date              | 16/04/2021 (month/day/year)                                  | 06/05/2022 (month/day/year)                                  |
| MindSpore Version          | 1.1.1                                                        | 1.6.1                                                        |
| Dataset                    | ImageNet2012, cifar-10                                       | ImageNet2012, cifar-10                                       |
| Training Parameters        | epoch=20,  batch_size = 128                                  | epoch=20,  batch_size = 128                                  |
| Optimizer                  | Adam                                                         | Adam                                                         |
| Loss Function              | BCELoss                                                      | BCELoss                                                      |
| Output                     | predict class                                                | predict class                                                |
| Accuracy                   | 310: 78.2%                                                   | 1pc: 77.8% ;  8pcs:  75.1%                                   |
| Loss                       | 10.9852                                                      | 0.3325(Dloss); 4.6742(Gloss)                                 |
| Speed                      | 1pc: 420 ms/step;  8pcs:  195 ms/step                        | 1pc: 104 ms/step;  8pcs:  178 ms/step                        |
| Total time                 | 1pc: 25.32 hours                                             | 1pc: 5.79 hours;  8pcs:  1.24 hours                          |
| Checkpoint for Fine tuning | 79.05M(.ckpt file)                                           | 69.67M(.ckpt file)                                           |
| Scripts                    | [dcgan script](https://gitee.com/mindspore/models/tree/master/research/cv/dcgan) | [dcgan script](https://gitee.com/mindspore/models/tree/master/research/cv/dcgan) |

# [Description of Random Situation](#contents)

We use random seed in train.py and cell.py for weight initialization.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
