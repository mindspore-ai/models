# [Table of contents](#table-of-contents)

- [HiFaceGAN Description](#hifacegan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Dataset Features](#dataset-features)
- [Environmental Requirements](#environmental-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
- [Script Parameters](#script-parameters)
- [Training and evaluation](#training-and-evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [HiFaceGAN Description](#table-of-contents)

HiFaceGAN is a novel method for a for photo-realistic face restoration under a “dual-blind” condition, lifting the requirements
of both the degradation and structural prior for training. The network architecture is based on collaborative suppression and
replenishment (CSR) blocks. In the suppression module, content-adaptive filters are used instead of usual convolutions. In the
replenishment module, spatial adaptive denormalization (SPADE) blocks are used, where each block receives the output from the
previous block and replenishes new details following the guidance of the corresponding semantic features encoded with the suppression
module. In this way, HiFaceGAN can automatically capture the global structure and progressively fill in finer visual details at
proper locations even without the guidance of additional face parsing information.

Extensive experiments on both synthetic and real face images verified the versatility, robustness and generalization capability
of the HiFaceGAN.

[Paper](https://arxiv.org/pdf/2005.05005.pdf): Yang L, Liu Ch, Wang P, et al. HiFaceGAN: Face Renovation via Collaborative Suppression and Replenishment

# [Model Architecture](#table-of-contents)

# [Dataset](#table-of-contents)

Dataset used: [Flickr-Faces-HQ Dataset (FFHQ, FLICKR)](https://github.com/NVlabs/ffhq-dataset)

Dataset size: 19.5G，15000 1024*1024 JPG images

- Train: 10000 images (first 10000 images, folders 00000-09000)
- Test: 5000 images (image folders 65000-69000)

To download FLICKR dataset, follow these steps:

- Clone [this repository](https://github.com/NVlabs/ffhq-dataset)
- Download manually [ffhq-dataset-v2.json](https://drive.google.com/open?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA) into the repository folder (for some reason the standard script cannot download this file)
- Launch the download script:

  ```bash
  python download_ffhq.py --json --images
  ```

This command downloads full FLICKR dataset (70000 png images in total). However, we only use 15000 of them.
You can modify the script to download only these 15000 files if you like or simply remove unnecessary files.
After that, move the contents of the folder `images1024x1024` into the dataset folder with name FLICKR.
You will get the overall structure of the dataset as follows (each directory containing 1000 png images):

```text
FLICKR
├── 00000/
├── 01000/
├── ...
├── 68000/
└── 69000/
```

Before training or running evaluation, install required packages from `research/cv/HiFaceGAN/`:

```shell
pip install -r requirements.txt
```

And preprocess images using image_preprocess.py script:

```shell
# Preprocess images for train phase
python src/dataset/image_preprocess.py /path/to/FLICKR train full 0 9
# Preprocess images for test (eval) phase
python src/dataset/image_preprocess.py /path/to/FLICKR test full 65 69
```

In this example, the images from directories [00000, 09000] and [65000, 69000] will be augmented and placed into
`/path/to/FLICKR/train_full` and `/path/to/FLICKR/test_full` respectively.

The directory structure after preprocessing:

```text
FLICKR
├── 00000/
├── 01000/
├── ...
├── 68000/
├── 69000/
├── test_full/
└── train_full/
```

# [Dataset Features](#table-of-contents)

# [Environmental Requirements](#table-of-contents)

- Hardware
    - Use GPU processor to build hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#table-of-contents)

## [Script and Sample Code](#table-of-contents)

```text
HiFaceGAN
├── scripts
│   ├── run_distributed_train_gpu.sh             # training on gpu in parallel mode
│   ├── run_eval_gpu.sh                          # evaluation on gpu
│   └── run_standalone_train_gpu.sh              # training on gpu in standalone mode
├── src
│   ├── dataset
│   │   ├── dataset.py                           # create dataset for training and evaluation
│   │   └── image_preprocess.py                  # add augmentations to images
│   ├── model
│   │   ├── architecture.py                      # auxiliary classes for HiFaceGAN architecture
│   │   ├── cell.py                              # HiFaceGAN training network wrapper
│   │   ├── discriminator.py                     # discriminator network
│   │   ├── generator.py                         # generator network
│   │   ├── initializer.py                       # initializer for neural networks
│   │   ├── loss.py                              # loss function for a model
│   │   ├── reporter.py                          # class for logging
│   │   └── vgg.py                               # vgg network for vgg loss computation
│   ├── model_utils
│   │   ├── config.py                            # config
│   │   ├── device_adapter.py                    # device adapter
│   │   ├── local_adapter.py                     # local adapter
│   │   └── moxing_adapter.py                    # moxing adapter
│   └── util.py                                  # useful functions for training and evaluation
├── default_config.yaml                          # configs for training and testing
├── eval.py                                      # eval launch file
├── export.py                                    # export checkpoints into mindir model
├── README.md                                    # README in English
├── requirements.txt                             # required modules
└── train.py                                     # train launch file
```

# [Script Parameters](#table-of-contents)

Both training and evaluation parameters can be configured in `../HiFaceGAN/defalt_config.yaml`.

```yaml
degradation_type: 'full'                   # The degradation type for images. Choose from ['full', 'down', 'noise', 'blur', 'jpeg']
img_size: 512                              # Size of image
batch_size: 1                              # The batch size to be used for training and evaluation

# Generator architecture
ngf: 64                                    # The number of generator features
input_nc: 3                                # The number of channels of input image

# Discriminator architecture
ndf: 64                                    # The number of discriminator features

# Train options
num_epochs: 30                             # The number of epochs
num_epochs_decay: 20                       # The number of epochs with linear lr decay

# Loss options
use_gan_feat_loss: True                    # Whether to use gan feature loss for generator
use_vgg_loss: True                         # Whether to use vgg loss for generator
lambda_feat: 10.0                          # The scale factor for gan feature loss, default is 10
lambda_vgg: 10.0                           # The scale factor for vgg loss, default is 10

# Optimizer and learning rate options
lr_policy: 'constant'                      # Learning rate policy. Choose from ['linear', 'constant']
use_ttur: True                             # Whether to use the Two-Time scale Update Rule
lr: 0.0002                                 # Initial learning rate
beta1: 0.0                                 # Beta1 parameter for Adam optimizer
beta2: 0.9                                 # Beta2 parameter for Adam optimizer
```

# [Training and evaluation](#table-of-contents)

Go to the directory `research/cv/HiFaceGAN/`, and after installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Training on GPU

  ```shell
  # Standalone training
  bash scripts/run_standalone_train_gpu.sh [DATA_PATH] [VGG_PATH]
  # Multi-GPU training
  bash scripts/run_distributed_train_gpu.sh [RANK_SIZE] [DATA_PATH] [VGG_PATH]
  ```

  Here VGG_PATH is the location of the pretrained VGG19 network checkpoint. We used the official [checkpoint from MindSporeHub](https://download.mindspore.cn/model_zoo/r1.3/vgg19_gpu_v130_cifar10_research_cv_bs64_acc93.75/).

  Example:

  ```shell
  # Standalone training
  bash scripts/run_standalone_train_gpu.sh /path/to/FLICKR /path/to/vgg.ckpt
  # Multi-GPU training
  bash scripts/run_distributed_train_gpu.sh 8 /path/to/FLICKR /path/to/vgg.ckpt
  ```

- Evaluation on GPU

  ```shell
  bash scripts/run_eval_gpu.sh [DATA_PATH] [CKPT_PATH] [NUM_CHECKPOINTS](optional)
  ```

  Here CKPT_PATH is a path to a directory with generator checkpoints. The metric is calculated for each of the last few checkpoints.
  The number of checkpoints can be set through the optional variable NUM_CHECKPOINTS (default is 5).
  As a result of running the script, the images for the best checkpoint will be saved.

  Example:

  ```shell
  bash scripts/run_eval_gpu.sh /path/to/FLICKR /path/to/checkpoints/
  ```

  If you want to validate only one checkpoint, place it into an individual directory. Then run the above command specifying the path to this directory.

- Training results will have the following form:

  ```text
  Epoch[50] [900/1250] step cost: 622.28 ms, G_loss: 7.85, D_loss: 0.13, G_vgg_loss: 2.27, G_gan_loss: 0.61, G_gan_feat_loss: 4.96, D_fake_loss: 0.11, D_real_loss: 0.02
  Epoch[50] [1000/1250] step cost: 621.89 ms, G_loss: 10.84, D_loss: 0.04, G_vgg_loss: 3.10, G_gan_loss: 0.93, G_gan_feat_loss: 6.80, D_fake_loss: 0.02, D_real_loss: 0.02
  Epoch[50] [1100/1250] step cost: 621.97 ms, G_loss: 10.26, D_loss: 0.18, G_vgg_loss: 3.54, G_gan_loss: 1.06, G_gan_feat_loss: 5.66, D_fake_loss: 0.02, D_real_loss: 0.16
  Epoch[50] [1200/1250] step cost: 622.82 ms, G_loss: 7.76, D_loss: 0.38, G_vgg_loss: 3.00, G_gan_loss: 0.27, G_gan_feat_loss: 4.49, D_fake_loss: 0.26, D_real_loss: 0.12
  Epoch [50] total cost: 778947.18 ms, per step: 623.16 ms, G_loss: 9.04, D_loss: 0.23
  ========== end training ===============
  ```

- Evaluation results will have the following form:

  ```text
  Metrics for the best checkpoint: PSNR = 24.1509, SSIM = 0.6486
  Start generating images for the best checkpoint...
  Save generated images at ./results/predict
  total cost 59.76 min
  ========== end predict ===============
  ```

# [Model Description](#table-of-contents)

## [Performance](#table-of-contents)

### [Training Performance](#table-of-contents)

| Parameters          | 8P GPU                                                                                  |
|---------------------|-----------------------------------------------------------------------------------------|
| Model               | HiFaceGAN                                                                               |
| Hardware            | GPU: 8 * GeForce RTX 3090 <br /> CPU 2.90GHz, 64 cores <br /> RAM:252G                  |
| Upload date         | 12.04.2022                                                                              |
| MindSpore version   | 1.5.2                                                                                   |
| Dataset             | FFHQ                                                                                    |
| Training parameters | num_epoch=30, num_epochs_decay=20, batch_size=1, lr=0.0002, <br /> beta1=0.0, beta2=0.9 |
| Optimizer           | Adam                                                                                    |
| Loss function       | Custom loss function                                                                    |
| Output              | Image                                                                                   |
| Speed               | 624 ms/step                                                                             |
| Checkpoint          | 500 MB, ckpt file                                                                       |
| Total time          | 10 h 50 min                                                                             |

### [Evaluation Performance](#table-of-contents)

| Parameters        | 1P GPU                                                                                                                                                                      |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model name        | HiFaceGAN                                                                                                                                                                   |
| Resource          | GPU: GeForce RTX 3090 <br /> CPU 2.90GHz, 64 cores <br /> RAM:252G                                                                                                          |
| Upload date       | 12.04.2022                                                                                                                                                                  |
| MindSpore version | 1.5.2                                                                                                                                                                       |
| Dataset           | FFHQ                                                                                                                                                                        |
| Batch size        | 1                                                                                                                                                                           |
| Output            | Image                                                                                                                                                                       |
| Metrics           | Peak signal-to-noise ratio (PSNR), typical results are in range (23.80, 24.25) <br /> Structural similarity index measure (SSIM), typical results are in range (0.63, 0.68) |
| PSNR              | 24.1509                                                                                                                                                                     |
| SSIM              | 0.6486                                                                                                                                                                      |

# [Description of Random Situation](#table-of-contents)

 Use set_global_seed function to set a random state. Note that the result of training and evaluation depends
 on dataset augmentations performed by imgaug library and is therefore device dependent.

# [ModelZoo Homepage](#table-of-contents)

 Please visit the [official website](https://gitee.com/mindspore/models).
