# Contents

- [Contents](#contents)
- [PWCnet Description](#pwcnet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [pretrained](#pretrained)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
        - [Train](#train)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PWCnet Description](#contents)

PWC-Net has been designed according to simple and well-established principles.

[Paper](https://arxiv.org/pdf/1709.02371.pdf):  Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"

# [Model Architecture](#contents)

PWCnet uses pyramidal processing, warping, and the use of a cost volume.

# [Dataset](#contents)

Train Dataset used: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip)
evaluating Dataset used:  [MPI-Sintel](https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip)

We use about 22872 image pairs and corresponding flow fields as training dataset and 23 sequences as evaluating dataset in this example, and you can also use your own datasets or open source datasets.

- The directory structure of train dataset is as follows:

```bash
.
└─training
  ├── 00001_img1.ppm        // img1 file
  ├── 00001_img2.ppm        // img2 file
  ├── 00001_flow.flo       // flo file
  │    ...
  ...
```

- The directory structure of evaluating dataset is as follows:

```bash
.
└─training
  ├── albedo
  ├── clean
  ├── final
     ├── alley_1
       ├── frame_0001.png
       ├── frame_0002.png
       ├── frame_0003.png
       ├── ....
       ├── frame_0050.png
    ├── ....
    ├── ....
  ├── flow
     ├── alley_1
       ├── frame_0001.flo
       ├── frame_0002.flo
       ├── ....
       ├── frame_0049.flo
  ├── flow_viz
  ├── invalid
  ├── occlusions
  ├── occlusions1
  ├── occlusions-clean
  │    ...
  ...
```

# [pretrained](#contents)

- Download pretrained model

```bash
mkdir ./pretrained_model
# download PWCNet pretrained file
wget -O ./pretrained_model/pwcnet-pretrained.pth https://github.com/visinf/irr/blob/master/saved_check_point/pwcnet/PWCNet/checkpoint_best.ckpt
```

- Convert pretrained model (from pytorch to mindspore, both Pytorch and Mindspore must be installed.)

```bash
# Convert pytorch pretrained model file to mindspore file.
bash scripts/run_ckpt_convert.sh [PYTORCH_FILE_PATH] [MINDSPORE_FILE_PATH]
```

# [Environment Requirements](#contents)

- Hardware(Ascend)
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```bash
.
└─ PWCnet
  ├─ README.md
  ├─ model_utils
    ├─ __init__.py                          # module init file
    ├─ config.py                            # Parse arguments
    ├─ device_adapter.py                    # Device adapter for ModelArts
    ├─ local_adapter.py                     # Local adapter
    └─ moxing_adapter.py                    # Moxing adapter for ModelArts
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    ├─ run_ckpt_convert.sh                  # launch converting pytorch ckpt file to pickle file on GPU
  ├─ src
    ├─ sintel.py                            # preprocessing evaluating dataset for eval
    ├─ common.py                            # handle dataset
    ├─ transforms.py                        # handle dataset
    ├─ flyingchairs.py                      # preprocessing training dataset for train
    ├─ pwcnet_model.py                      # network backbone
    ├─ pwc_modules.py                       # network backbone
    ├─ log.py                               # log function
    ├─ loss.py                              # loss function
    └─ lr_generator.py                      # generate learning rate
    ├─ utils
        ├─ ckpt_convert.py                  # convert pytorch ckpt file to pickle file
  ├─ default_config.yaml                    # Configurations
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
```

## [Running Example](#contents)

### Train

- Stand alone mode(recommended)

    ```bash
    Ascend
    bash scripts/run_standalone_train.sh [TRAIN_LABEL_FILE] [EVAL_DIR] [DEVCIE_ID] [PRETRAINED_BACKBONE]
    ```

    for example, on Ascend:

    ```bash
    bash scripts/run_standalone_train.sh ./data/FlyingChairs/ ./data/MPI-Sintel/ 0 ./pretrain.ckpt
    ```

- Distribute mode

    ```bash
    Ascend
    bash scripts/run_distribute_train.sh [TRAIN_LABEL_FILE] [EVAL_DIR] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./device0/train.log":

```bash
epoch[0], iter[0],  0.01 imgs/sec   Loss 639.3672 639.3672
epoch[0], iter[20], 0.38 imgs/sec   Loss 68.1912 179.4218
epoch[0], iter[40], 25.73 imgs/sec   Loss 33.6643 50.4679
INFO:epoch[0], iter[80], 26.97 imgs/sec Loss 18.4107 23.8088

...
epoch[9], iter[55460], 27.81 imgs/sec   Loss 3.4625 2.4753
epoch[9], iter[55480], 29.96 imgs/sec   Loss 1.9749 2.3819
epoch[9], iter[55500], 29.11 imgs/sec   Loss 2.8970 2.7417
epoch[9], iter[55520], 27.98 imgs/sec   Loss 4.3935 2.9107
epoch[9], iter[55540], 27.83 imgs/sec   Loss 2.7637 2.3774
epoch[9], iter[55560], 28.15 imgs/sec   Loss 1.8070 2.4678
```

### Evaluation

```bash
Ascend

bash scripts/run_eval.sh [EVAL_DIR] [DEVCIE_ID] [PRETRAINED_BACKBONE]
```

for example, on Ascend:

```bash
bash scripts/run_eval.sh ./data/MPI-Sintel/ 0 ./0-1_10000.ckpt
```

You will get the result as following in "./device0/eval.log":

```bash
EPE: 6.9049
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend
| -------------------------- | ----------------------------------------------------------
| Model Version              | V1
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8
| Uploaded Date              | 01/04/2022 (month/day/year)
| MindSpore Version          | 1.5.0
| Dataset                    | 22872 images pairs
| Training Parameters        | epoch=10, batch_size=4, momentum=0.9, lr=0.0001
| Optimizer                  | Adam
| Loss Function              | MultiScaleEPE_PWC
| Outputs                    | EPE
| Total time                 | 1ps: 2.5 hours; 8pcs: 0.6 hours

### Evaluation Performance

| Parameters          | Ascend
| ------------------- | -----------------------------
| Model Version       | V1
| Resource            | Ascend 910; OS Euler2.8
| Uploaded Date       | 01/04/2022 (month/day/year)
| MindSpore Version   | 1.5.0
| Dataset             | 133
| batch_size          | 1
| EPE                 | 6.9049

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
