# Contents

- [Contents](#contents)
    - [IndexNet Description](#indexnet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Standalone Training](#standalone-training)
            - [Distribute Training](#distribute-training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
        - [Model Export](#model-export)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [IndexNet Description](#contents)

Upsampling is an essential stage for most dense prediction tasks using deep convolutional neural networks (CNNs).
The frequently used upsampling operators include transposed convolution, unpooling, periodic shuffling
(also known as depth-to-space), and naive interpolation followed by convolution.
These operators, however, are not general-purpose designs and often exhibit different behaviors in different tasks.
Instead of using maxpooling and unpooling, IndexNet is based on two novel operations: indexed pooling and indexed upsampling
where downsampling and upsampling are guided by learned indices. The indices are generated dynamically conditioned
on the feature map and are learned using a fully convolutional network, termed IndexNet, without supervision.

[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lu_Indices_Matter_Learning_to_Index_for_Deep_Image_Matting_ICCV_2019_paper.pdf):  Indices Matter: Learning to Index for Deep Image Matting. Hau Lu, Yutong Dai, Chunhua Shen.

## [Model Architecture](#contents)

IndexNet bases on the UNet architecture and uses mobilenetv2 as backbone.
Mobilenetv2 was chosen because it is lightweight and allows the use of higher-resolution images on the same GPU as high capacity backbones.
All 2-stride convolutions were changed by 1-stride convolutions and 2-stride 2x2 max poolings after each encoding stage for downsampling, which allows the extraction of indices.
If applying the IndexNet idea, max pooling and unpooling layers can be replaced with IndexedPooling and IndexedUnpooling, respectively.

## [Dataset](#contents)

Paper uses the Adobe Image Matting dataset, but it is in close access.
Thus, we use AIM-500 (Automatic Image Matting - 500) dataset, which is in open access, and anyone can download it.

Every image from AIM-500 dataset cuts out by mask and N (96 train part, 20 test part) times placed as foreground
over the unique image from the COCO-2014 dataset (train part), which is used as background.

Datasets used: AIM-500, COCO-2014 (train).

|               | AIM-500                                          | COCO-2014             | Merged (after processing) |
| --------------|------------------------------------------------- |---------------------- |-------------------------- |
| Dataset size  | ~0.35 Gb                                         | ~13.0 Gb              | ~86.0 Gb                  |
| Train         | 0.35 Gb, 3 * 500 images (mask, original, trimap) | 13.0 Gb, 82783 images | 84 Gb, 43200 images       |
| Test          | -                                                | -                     | 2 Gb, 1000 images         |
| Data format   | .png, .jpg images                                | .jpg images           | .png images               |

Note: We manually split AIM-500 for the train/test parts (450/50).

Download [AIM-500](https://drive.google.com/drive/folders/1IyPiYJUp-KtOoa-Hsm922VU3aCcidjjz) dataset
(3 folders: original, mask, trimap), unzip them, move folders from unzipped archives into one folder named AIM-500.
Download [COCO-2014 train](http://images.cocodataset.org/zips/train2014.zip) and unzip.

The structure of the datasets will be as follows:

```text
.
└─AIM-500      <- data_dir
  ├─mask
  │ └─***.png
  ├─original
  │ └─***.jpg
  └─trimap
    └─***.png

.
└─train2014    <- bg_dir
  └─***.jpg

Where *** is the image file name
```

To process dataset use the command below.

```bash
python -m data.process_dataset --data_dir /path/to/AIM-500 --bg_dir /path/to/coco/train2014
```

- DATA_DIR - path to image matting dataset (AIM-500 folder, in this case).
- BG_DIR - path to backgrounds dataset (COCO/train2014 folder, in this case).

Note: Before data processing requirements will be installed. Make sure that you have ~100 Gb free space at disk,
which corresponds to --data_dir path. It can take about 20 hours to prepare dataset, depends on hardware.

During processing the data_dir structure will be automatically changed and
the merged images saved into data_dir/train/merged, data_dir/validation/merged. The bg_dir will remain unchanged.
Processed dataset will have the following structure:

```text
.
└─AIM-500      <- data_dir
  ├─train
  │ ├─data.txt
  │ ├─mask
  │ ├─merged
  │ └─original
  └─validation
    ├─data.txt
    ├─mask
    ├─merged
    ├─original
    └─trimap

.
└─train2014    <- bg_dir
```

## [Environment Requirements](#contents)

- Hardware（GPU）
- Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

Note: We use MindSpore 1.6.1 GPU, thus make sure that you install >= 1.6.1 version.

## [Quick Start](#contents)

After installing MindSpore through the official website, you can follow the steps below for training and evaluation,
in particular, before training, you need to install `requirements.txt` by following command `pip install -r requirements.txt`
and [download](https://mindspore.cn/resources/hub/details/en?MindSpore/ascend/1.2/mobilenetv2_v1.2_imagenet2012)
the pre-trained on ImageNet mobilenetv2 backbone.

```bash
# Run standalone training example
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [MOBILENET_CKPT] [DATA_DIR] [BG_DIR]

# Run distribute training example
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [MOBILENET_CKPT] [DATA_DIR] [BG_DIR]
```

- DEVICE_ID - process device ID.
- DEVICE_NUM - number of distribute training devices.
- LOGS_CKPT_DIR - path to the directory, where the training results (ckpts, logs) will be stored.
- MOBILENET_CKPT - path to the pre-trained mobilenetv2 backbone ([link](https://mindspore.cn/resources/hub/details/en?MindSpore/ascend/1.2/mobilenetv2_v1.2_imagenet2012)).
- DATA_DIR - path to image matting dataset (AIM-500 folder, in this case).
- BG_DIR - path to backgrounds dataset (COCO/train2014 folder, in this case).

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```text
.
└─IndexNet
  ├─README.md
  ├─requirements.txt
  ├─data
  │ └─process_dataset.py               # data preparation script
  ├─scripts
  │ ├─run_distribute_train_gpu.sh      # launch distribute train on GPU
  │ ├─run_eval_gpu.sh                  # launch evaluation on GPU
  │ └─run_standalone_train_gpu.sh      # launch standalone train on GPU
  ├─src
  │ ├─cfg
  │ │ ├─__init__.py
  │ │ └─config.py                      # parameter parser
  │ ├─dataset.py                       # dataset script and utils
  │ ├─layers.py                        # model layers
  │ ├─model.py                         # model script
  │ ├─modules.py                       # model modules
  │ └─utils.py                         # utilities used in other scripts
  ├─default_config.yaml                # default configs
  ├─eval.py                            # evaluation script
  ├─export.py                          # export to MINDIR script
  └─train.py                           # training script
```

### [Script Parameters](#contents)

```yaml
# Main arguments:

# training params
batch_size: 16          # Batch size for training
epochs: 30              # Number of training epochs
learning_rate: 0.01     # Learning rate init
backbone_lr_mult: 100   # Learning rate scaling (division) for backbone params
lr_decay: 0.1           # Learning rate scaling at milestone
milestones: [20, 26]    # Milestones for learning rate scheduler
input_size: 320         # Input crop size for training
```

### [Training Process](#contents)

#### Standalone Training

Note: For all trainings necessary to use pretrained modilenetv2 as backbone.

```bash
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [MOBILENET_CKPT] [DATA_DIR] [BG_DIR]
```

- DEVICE_ID - process device ID.
- LOGS_CKPT_DIR - path to the directory, where the training results (ckpts, logs) will be stored.
- MOBILENET_CKPT - path to the pre-trained mobilenetv2 backbone ([link](https://mindspore.cn/resources/hub/details/en?MindSpore/ascend/1.2/mobilenetv2_v1.2_imagenet2012)).
- DATA_DIR - path to image matting dataset (AIM-500 folder, in this case).
- BG_DIR - path to backgrounds dataset (COCO/train2014 folder, in this case).

The above command will run in the background, you can view the result through the generated standalone_train.log file.
After training, you can get the training loss and time logs in chosen logs dir.

The model checkpoints will be saved in `[LOGS_CKPT_DIR]` directory.

#### Distribute Training

```bash
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [LOGS_CKPT_DIR] [MOBILENET_CKPT] [DATA_DIR] [BG_DIR]
```

- DEVICE_NUM - number of distribute training devices.
- LOGS_CKPT_DIR - path to the directory, where the training results (ckpts, logs) will be stored.
- MOBILENET_CKPT - path to the pre-trained mobilenetv2 backbone ([link](https://mindspore.cn/resources/hub/details/en?MindSpore/ascend/1.2/mobilenetv2_v1.2_imagenet2012)).
- DATA_DIR - path to image matting dataset (AIM-500 folder, in this case).
- BG_DIR - path to backgrounds dataset (COCO/train2014 folder, in this case).

The above command will run in the background, you can view the result through the generated distribute_train.log file.
After training, you can get the training loss and time logs in chosen logs dir.

The model checkpoints will be saved in `[LOGS_CKPT_DIR]` directory.

### [Evaluation Process](#contents)

#### Evaluation

To start evaluation run the command below.

```bash
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CKPT_URL] [DATA_DIR] [LOGS_DIR]
```

- DEVICE_ID - process device ID.
- CKPT_URL - path to the trained IndexNet model.
- DATA_DIR - path to image matting dataset (AIM-500 folder, in this case).
- LOGS_DIR - path to the directory, where the eval results (outputs, logs) will be stored.

The above python command will run in the background. Predicted masks (.png) will be stored into chosen `[LOGS_DIR]`.
And there you can view the results through the file "eval.log".

### [Model Export](#contents)

To export the model to mindir format, run the following command:

```bash
python export.py --ckpt_url [CKPT_URL]
```

- CKPT_URL - path to the trained IndexNet model.

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters                 | GPU (1p)                                                    | GPU (8p)                                                       |
| -------------------------- |------------------------------------------------------------ |--------------------------------------------------------------- |
| Model                      | IndexNet                                                    | IndexNet                                                       |
| Hardware                   | 1 Nvidia Tesla V100-PCIE, CPU @ 3.40GHz                     | 8 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz         |
| Upload Date                | 07/04/2022 (day/month/year)                                 | 07/04/2022 (day/month/year)                                    |
| MindSpore Version          | 1.6.1                                                       | 1.6.1                                                          |
| Dataset                    | AIM-500, COCO-2014 (composition of datasets)                | AIM-500, COCO-2014 (composition of datasets)                   |
| Training Parameters        | epochs=30, lr=0.01, batch_size=16, num_workers=12           | epochs=30, lr=0.01, batch_size=16 (each device), num_workers=4 |
| Optimizer                  | Adam, beta1=0.9, beta2=0.999, eps=1e-8                      | Adam, beta1=0.9, beta2=0.999, eps=1e-8                         |
| Loss Function              | Weighted loss (alpha predictions loss and composition loss) | Weighted loss (alpha predictions loss and composition loss)    |
| Speed                      | ~ 516 ms/step                                               | ~ 2670 ms/step                                                 |
| Total time                 | ~ 11.6 hours                                                | ~ 7.5 hours                                                    |

#### Evaluation Performance

| Parameters             | GPU (1p)                                               | GPU (8p)                                               |
| -----------------------|--------------------------------------------------------|--------------------------------------------------------|
| Model                  | IndexNet                                               | IndexNet                                               |
| Resource               | 1 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz | 1 Nvidia RTX 3090, Intel Xeon Gold 6226R CPU @ 2.90GHz |
| Upload Date            | 07/04/2022 (day/month/year)                            | 07/04/2022 (day/month/year)                            |
| MindSpore Version      | 1.6.1                                                  | 1.6.1                                                  |
| Dataset                | AIM-500, COCO-2014 (composition of datasets)           | AIM-500, COCO-2014 (composition of datasets)           |
| Batch_size             | 1                                                      | 1                                                      |
| Outputs                | .png images of alpha masks                             | .png images of alpha masks                             |
| Metrics                | 21.51 SAD, 0.0096 MSE, 13.43 Grad, 20.43 Conn          |  22.06 SAD, 0.0134 MSE, 12.84 Grad, 21.32 Conn         |
| Metrics expected range | < 24.00 SAD, < 0.0120 MSE, < 13.70 Grad, < 23.20 Conn  |  < 24.20 SAD, < 0.0145 MSE, < 13.40 Grad, < 22.70 Conn |

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
