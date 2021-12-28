# Contents

- [Contents](#contents)
- [SinGAN Description](#singan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
    - [Dependences](#dependences)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
    - [Model Export](#model-export)
    - [inference Process](#inference-process)
        - [Export MINDIR](#export-mindir)
        - [Ascend310 Inference](#ascend310-inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [Model_Zoo Homepage](#model_zoo-homepage)

# [SinGAN Description](#contents)

SinGAN is an unconditional generative model that can be learned from a single natural image.It is trained to capture the internal distribution of patches within the image, and is then able to generate high quality, diverse samples that carry the same visual content as the image. SinGAN contains a pyramid of fully convolutional GANs, each responsible for learning the patch distribution at a different scale of the image. This allows generating new samples of arbitrary size and aspect ratio, that have significant variability, yet maintain both the global structure and the fine textures of the training image. In contrast to previous single image GAN schemes, SinGAN is not limited to texture images, and is not conditional (i.e. it generates samples from noise). User studies confirm that the generated samples are commonly confused to be real images. The utility of SinGAN is in a wide range of image manipulation tasks.

[Paper](https://arxiv.org/abs/1905.01164): SinGAN: Learning a Generative Model from a Single Natural Image.

# [Model Architecture](#contents)

Architecture guidelines for SinGAN

- achieved by a pyramid of fully convolutional light-weight GANs, each is responsible for capturing the distribution of patches at a different scale.
- contains one generation network and one discriminant network  at each different scale.
- generation and discriminant networks are both a fully convolutional net with 5 conv-blocks of the form Conv(3 × 3)-BatchNorm-LeakyReLU.

# [Dataset](#contents)

A single natural image, for example, 'thunder.jpg'

![](data/thunder.jpg)

```text

└─data
  └─thunder.jpg
```

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Dependences](#contents)

- Python==3.7.5
- Mindspore==1.3.0

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─CGAN
  ├─ README.md              # descriptions about SinGAN
  ├─ requirements.txt       # required modules
  ├─ scripts
    ├─ run_train_ascend.sh  # Ascend training(1 pcs)
    ├─ run_eval_ascend.sh   # Ascend eval
    └─ run_infer_310.sh     # Ascend-310 infer
  ├─ data
    └─ thunder.jpg          # A single nature image
  ├─ ascend310_infer
    ├─ src
      ├─ main.cc            # Ascend-310 inference source code
      └─ utils.cc           # Ascend-310 inference source code
    ├─ inc
      └─ utils.h            # Ascend-310 inference source code
    ├─ build.sh             # Ascend-310 inference source code
    └─ CMakeLists.txt       # CMakeLists of Ascend-310 inference
  ├─ src
    ├─ __init__.py          # init file
    ├─ block.py             # SinGAN block define
    ├─ cell.py              # SinGAN cell define
    ├─ config.py            # SinGAN config define
    ├─ functions.py         # SinGAN function define
    ├─ imresize.py          # SinGAN image resize define
    ├─ loss.py              # SinGAN loss define
    ├─ manipulate.py        # SinGAN generate images define
    └─ model.py             # SinGAN networks define
  ├─ train.py               # SinGAN training
  ├─ train_modelarts.py     # SinGAN training on modelarts
  ├─ eval.py                # SinGAN evaluation
  ├─ export.py              # SinGAN export
  ├─ postprocess.py         # Ascend-310 inference postprocess
  └─ preprocess.py          # Ascend-310 inference preprocess

```

## [Script Parameters](#contents)

Major parameters in train.py and config.py as follows:

```python
"device_target": Ascend  # run platform, only support Ascend.
"device_id": 0           # device id, default is 0.
"n_gen": 50              # number of images to generate at final scale, default is 50.
"batch_size": 1          # batch_size, default is 1.
"nc_im": 3               # image channels, default is 3.
"nc_z": 3                # noise channels, default is 3.
"min_nfc": 32            # minimum model filter numbers, default is 32.
"ker_size": 3            # kernel size for convolution blocks, default is 3.
"stride": 1              # stride for convolution blocks, default is 1.
"padd_size": 0           # padding size for convolution blocks, default is 0.
"num_layer": 5           # ench generator or discriminator model convolution block numbers, default is 5.
"scale_factor": 0.75     # pyramid scale factor, default is 0.75.
"min_size": 25           # image minimal size at the coarser scale, default is 25.
"max_size": 250          # image maximal size at the finer scale, default is 250.
"niter": 2000            # epochs to train per scale, default is 2000.
"gamma": 0.1             # scheduler gamma, default is 0.1.
"lr_g": 0.0005           # learning rate for genarators, default is 0.0005.
"lr_d": 0.0005           # learning rate for discriminators, default is 0.0005.
"Gsteps": 3              # Generator inner steps, default is 250.
"Dsteps": 3              # Discriminator inner steps, default is 250.
"lambda_grad": 0.1       # gradient penelty weight, default is 0.1.
"alpha": 10              # reconstruction loss weight, default is 10.
```

## [Training Process](#contents)

### [Training](#content)

- Run `run_train_ascend.sh` for non-distributed training of SinGAN model.

```bash
# standalone training
bash scripts/run_train_ascend.sh [INPUT_DIR] [INPUT_NAME] [DEVICE_ID]
```

### [Training Result](#content)

Training result will be stored in `TrainedModels` and produced images at final scale will be stored in  `train_Output` .

## [Evaluation Process](#contents)

### [Evaluation](#content)

- Run `run_eval_ascend.sh` for evaluation.

```bash
# eval
bash scripts/run_eval_ascend.sh [INPUT_DIR] [INPUT_NAME] [DEVICE_ID]
```

### [Evaluation Result](#content)

Evaluation result will be stored in `eval_Output`. Under this, you can find generator results at final scale.

## Model Export

```bash
python eval.py --input_dir [INPUT_DIR] --input_name [INPUT_NAME] --device_id [DEVICE_ID]
```

## [Inference Process](#contents)

### [Export MINDIR](#content)

Before performing inference, the mindir files must be exported by `export.py`.

```bash
python export.py --input_dir [INPUT_DIR] --input_name [INPUT_NAME] --device_id [DEVICE_ID]
```

### [Ascend310 Inference](#content)

- Run `run_infer_310.sh` for Ascend310 inference.

```bash
# infer
bash scripts/run_infer_310.sh [MINDIR_PATH] [INPUT_DIR] [INPUT_NAME] [NOISE_AMP] [STOP_SCALE] [DEVICE_ID]
```

- [MINDIR_PATH]：The path of the exported mindir files.
- [INPUT_DIR]：The relative path of the directory where the image is located.
- [INPUT_NAME]：The name of the input image.
- [NOISE_AMP]：The relative path of the directory where trainable noise_amp.npy files are located.
- [STOP_SCALE]：There are 8 scales for singan, you can choose 1~8 to generate an image at different scale.

Ascend310 inference result will be stored in the postprocess_Result path. Under this, you can find generator results.

# Model Description

## Performance

### Evaluation Performance

| Parameters                 | single Ascend                                              |
| -------------------------- | ---------------------------------------------------------- |
| Model Version              | SinGAN                                                     |
| Resource                   | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G |
| uploaded Date              | 12/14/2021 (month/day/year)                                |
| MindSpore Version          | 1.3.0                                                      |
| Dataset                    | A single nature image                                      |
| Training Parameters        | epoch=2000,  batch_size = 1, learning rate=0.0005          |
| Optimizer                  | Adam                                                       |
| Loss Function              | Mean Sqare Loss & WGAN-GP                                  |
| Output                     | generates samples from noise                               |
| Loss                       | scale 8: d_loss = -0.0204 , g_loss = 0.1911                |
| Speed                      | scale 8:  0.058ms per step                                 |
| Total time                 | 1pc(Ascend): 52.9 mins                                     |
| Checkpoint for Fine tuning | scale 1\~4: 119KB; scale 5\~8: 453KB                       |

# [Description of Random Situation](#contents)

We use random seed in train.py and cell.py for weight initialization.

# [Model_Zoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
