# Contents

- [Contents](#contents)
- [CycleGAN Description](#cyclegan-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [ONNX evaluation](#onnx-evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer](#infer)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [CycleGAN Description](#contents)

Image-to-image translation is a visual and image problem. Its goal is to use paired images as a training set and (let the machine) learn the mapping from input images to output images. However, in many tasks, paired training data cannot be obtained. CycleGAN does not require the training data to be paired. It only needs to provide images of different domains to successfully train the image mapping between different domains. CycleGAN shares two generators, and then each has a discriminator.

[Paper](https://arxiv.org/abs/1703.10593): Zhu J Y , Park T , Isola P , et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. 2017.

![CycleGAN Imgs](imgs/objects-transfiguration.jpg)

# [Model Architecture](#contents)

The CycleGAN contains two generation networks and two discriminant networks.

# [Dataset](#contents)

Download CycleGAN datasets and create your own datasets. We provide data/download_cyclegan_dataset.sh to download the datasets.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend or GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```markdown
.CycleGAN
├─ README.md                            # descriptions about CycleGAN
├─ data
  └─ download_cyclegan_dataset.sh.py    # download dataset
├── scripts
  └─ run_train_ascend.sh                # launch ascend training(1 pcs)
  └─ run_train_standalone_gpu.sh        # launch gpu training(1 pcs)
  └─ run_train_distributed_gpu.sh       # launch gpu training(8 pcs)
  └─ run_infer_310.sh                   # launch 310 infer
├─ imgs
  └─ objects-transfiguration.jpg        # CycleGAN Imgs
├─ ascend310_infer
  ├─ src
    ├─ main.cc                         # Ascend-310 inference source code
    └─ utils.cc                        # Ascend-310 inference source code
  ├─ inc
    └─ utils.h                         # Ascend-310 inference source code
  ├─ build.sh                          # Ascend-310 inference source code
  ├─ CMakeLists.txt                    # CMakeLists of Ascend-310 inference program
  └─ fusion_switch.cfg                 # Use BatchNorm2d instead of InstanceNorm2d
├─ src
  ├─ __init__.py                       # init file
  ├─ dataset
    ├─ __init__.py                     # init file
    ├─ cyclegan_dataset.py             # create cyclegan dataset
    └─ distributed_sampler.py          # iterator of dataset
  ├─ models
    ├─ __init__.py                     # init file
    ├─ cycle_gan.py                    # cyclegan model define
    ├─ losses.py                       # cyclegan losses function define
    ├─ networks.py                     # cyclegan sub networks define
    ├─ resnet.py                       # resnet generate network
    └─ depth_resnet.py                 # better generate network
  └─ utils
    ├─ __init__.py                     # init file
    ├─ args.py                         # parse args
    ├─ reporter.py                     # Reporter class
    └─ tools.py                        # utils for cyclegan
├─ eval.py                             # generate images from A->B and B->A
├─ train.py                            # train script
├─ export.py                           # export mindir script
├─ preprocess.py                       # data preprocessing script for scend-310 inference
└─ postprocess.py                      # data post-processing script for scend-310 inference
```

## [Script Parameters](#contents)

Major parameters in train.py and config.py as follows:

```python
"platform": Ascend       # run platform, support GPU and Ascend and CPU.
"device_id": 0           # device id, default is 0.
"model": "resnet"        # generator model.
"pool_size": 50          # the size of image buffer that stores previously generated images, default is 50.
"lr_policy": "linear"    # learning rate policy, default is linear.
"image_size": 256        # input image_size, default is 256.
"batch_size": 1          # batch_size, default is 1.
"max_epoch": 200         # epoch size for training, default is 200.
"in_planes": 3           # input channels, default is 3.
"ngf": 64                # generator model filter numbers, default is 64.
"gl_num": 9              # generator model residual block numbers, default is 9.
"ndf": 64                # discriminator model filter numbers, default is 64.
"dl_num": 3              # discriminator model residual block numbers, default is 3.
"outputs_dir": "outputs" # models are saved here, default is ./outputs.
"dataroot": None         # path of images (should have subfolders trainA, trainB, testA, testB, etc).
"load_ckpt": False       # whether load pretrained ckpt.
"G_A_ckpt": None         # pretrained checkpoint file path of G_A.
"G_B_ckpt": None         # pretrained checkpoint file path of G_B.
"D_A_ckpt": None         # pretrained checkpoint file path of D_A.
"D_B_ckpt": None         # pretrained checkpoint file path of D_B.
```

## [Training](#contents)

- running on Ascend with default parameters

    ```bash
    bash scripts/run_train_ascend.sh [DATA_PATH] [EPOCH_SIZE]
    # epoch_size is recommended 200
    ```

- running on GPU with default parameters

    ```bash
    bash scripts/run_train_standalone_gpu.sh [DATA_PATH] [EPOCH_SIZE]
    # epoch_size is recommended 200
    ```

- running on 8 GPUs with default parameters

    ```bash
    bash scripts/run_train_distributed_gpu.sh [DATA_PATH] [EPOCH_SIZE]
    # epoch_size is recommended 600
    ```

- running on CPU with default parameters

    ```bath
    python train.py --platform CPU --dataroot [DATA_PATH] --use_random False --max_epoch [EPOCH_SIZE] --print_iter 1 pool_size 0
    ```

## [Evaluation](#contents)

```bash
python eval.py --platform [PLATFORM] --dataroot [DATA_PATH] --G_A_ckpt [G_A_CKPT] --G_B_ckpt [G_B_CKPT]
```

**Note: You will get the result as following in "./outputs_dir/predict".**

## [ONNX evaluation](#contents)

First, export your models:

```bash
python export.py --platform GPU --model ResNet --G_A_ckpt /path/to/GA.ckpt --G_B_ckpt /path/to/GB.ckpt --export_file_name /path/to/<prefix> --export_file_format ONNX
```

You will get two `.onnx` files: `/path/to/<prefix>_AtoB.onnx` and `/path/to/<prefix>_BtoA.onnx`

Next, run ONNX eval with the same `export_file_name` as in export:

```bash
python eval_onnx.py --platform [PLATFORM] --dataroot [DATA_PATH] --export_file_name /path/to/<prefix>
```

**Note: You will get the result as following in "./outputs_dir/predict".**

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```bash
python export.py --G_A_ckpt [CKPT_PATH_A] --G_B_ckpt [CKPT_PATH_A] --export_batch_size 1 --export_file_name [FILE_NAME] --export_file_format [FILE_FORMAT]
```

### [Infer](#contents)

Before performing inference, the mindir file must be exported by `export.py`.Current batch_Size can only be set to 1.

```shell
bash run_infer_cpp.sh [MINDIR_PATH] [DATASET_PATH] [DATASET_MODE] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]
```

- `DATASET_PATH` is mandatory, and must specify original data path.
- `DATASET_MODE` is the translation direction of CycleGAN, it's value is 'AtoB' or 'BtoA'.
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_TARGET` can choose from [Ascend, GPU, CPU].
- `DEVICE_ID` is optional, default value is 0.

for example, on Ascend:

```bash
bash ./scripts/run_infer_cpp.sh ./cpp_infer/CycleGAN_AtoB.mindir ./data/horse2zebra AtoB y Ascend 0
```

### [Result](#contents)

Inference result is saved in current path, you can find result in infer_output_img file.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

We use Depth Resnet Generator on Ascend and Resnet Generator on GPU.

| Parameters                 | single Ascend/GPU                                           | 8 GPUs                                                      |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version              | CycleGAN                                                    | CycleGAN                                                    |
| Resource                   | Ascend 910/NV SMX2 V100-32G                                 | NV SMX2 V100-32G x 8                                        |
| MindSpore Version          | 1.2                                                         | 1.2                                                         |
| Dataset                    | horse2zebra                                                 | horse2zebra                                                 |
| Training Parameters        | epoch=200, steps=1334, batch_size=1, lr=0.0002              | epoch=600, steps=166, batch_size=8, lr=0.0002               |
| Optimizer                  | Adam                                                        | Adam                                                        |
| Loss Function              | Mean Sqare Loss & L1 Loss                                   | Mean Sqare Loss & L1 Loss                                   |
| outputs                    | probability                                                 | probability                                                 |
| Speed                      | 1pc(Ascend): 123 ms/step; 1pc(GPU): 190 ms/step             | 190 ms/step                                                 |
| Total time                 | 1pc(Ascend): 9.6h; 1pc(GPU): 14.9h;                         | 5.7h                                                        |
| Checkpoint for Fine tuning | 44M (.ckpt file)                                            | 44M (.ckpt file)                                            |

### Evaluation Performance

| Parameters          | single Ascend/GPU           |
| ------------------- | --------------------------- |
| Model Version       | CycleGAN                    |
| Resource            | Ascend 910/NV SMX2 V100-32G |
| MindSpore Version   | 1.2                         |
| Dataset             | horse2zebra                 |
| batch_size          | 1                           |
| outputs             | transferred images          |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
