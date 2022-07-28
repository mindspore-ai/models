# Few-Shot Meta-Baseline

# Contents

- [Few-Shot Meta-Baseline](#few-shot-meta-baseline)
    - [Datasets](#datasets)
    - [Environment](#environment)
    - [Script and sample code](#script-and-sample-code)
    - [Script parameters](#script-parameters)
    - [Quick start](#quick-start)
    - [Main Results](#main-results)
        - [5-way accuracy (%) on *miniImageNet*](#5-way-accuracy-----on--miniimagenet-)
    - [Running the code](#running-the-code)
        - [1. Training Classifier-Baseline](#1-training-classifier-baseline)
        - [2. Training Meta-Baseline](#2-training-meta-baseline)
        - [3. Test](#3-test)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
    - [Performance](#Performance)  
    - [Citation](#citation)

# [Few-Shot Meta-Baseline](#Contents)

Mindspore implementation for ***Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning***.

Original Pytorch implementation can be seen
Meta-Baseline [here](https://github.com/cyvius96/few-shot-meta-baseline).

<img src="https://user-images.githubusercontent.com/10364424/76388735-bfb02580-63a4-11ea-8540-4021961a4fbe.png" width="600">

## [Datasets](#Contents)

- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (
  courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))

Download the datasets and link the folders into `dataset/` with names `mini-imagenet`.
Note `imagenet` refers to ILSVRC-2012 1K dataset with two directories `train` and `val` with class
folders.

- Directory structure of the dataset:

```markdown
  .dataset(root_path)
  ├── mini-imagenet
      ├── miniImageNet_category_split_val.pickle
      ├── miniImageNet_category_split_train_phase_val.pickle
      ├── miniImageNet_category_split_train_phase_train.pickle
      ├── miniImageNet_category_split_train_phase_test.pickle
      ├── miniImageNet_category_split_test.pickle
```

## [Environment](#Contents)

- Hardware (Ascend/GPU)
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Script and sample code](#Contents)

```markdown
  . meta-baseline
  ├── .dataset(root_path)                       # dataset see section of dataset
  ├── scripts
  │   ├──run_eval.sh                            # script to eval
  │   ├──run_standalone_train_classifier.sh     # script to train_classifier
  │   ├──run_standalone_train_meta.sh           # script to train_meta
  ├── src
  │   ├──data
  │      ├──InerSamplers.py                     # sampler
  │      ├──mini_Imagenet.py                    # mini_Imagenet
  │   ├──model
  │      ├──classifier.py                       # train_classifier
  │      ├──meta_baseline.py                    # train meta_baseline
  │      ├──meta_eval.py                        # evaluation
  │      ├──resnet12.py                         # backbone
  │   ├──util
  │      ├──_init_.py                           # util
  ├── eval.py                                   # evaluation script
  ├── export.py                                 # export
  ├── README.md                                 # descriptions about meta-baseline
  ├── train_classifier.py                       # train_classifier script
  └── train_meta.py                             # train_meta script
```

## [Script parameters](#Contents)

Parameters for both train_classifier and train_meta can be set in the follow:

- Parameters:

```text
  # base setting
  "root_path": "../dataset",            # dataset root path
  "device_target": "GPU",               # device GPU or Ascend
  "run_offline": False,                 # run on line or offline  
  "dataset": "mini-imagenet",           # dataset mini_imagenet
  "ep_per_batch": 4,                    # nums of batch episode
  "max_epoch": 25,                      # epoch
  "lr": 0.1,                            # lr
  "n_classes": 64,                      # base classes 64
  "batch_size": 128,                    # batchsize
  "weight_decay": 5.e-4,                # weight_decay
  "num_ways": 5,                        # way 5
  "num_shots": 1,                       # shot 1 or 5
```

## [Quick start](#Contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- run on Ascend or GPU

  ```bash
  # standalone training classifier
  bash scripts/run_standalone_train_classifier.sh [GPU] [./dataset]
  # standalone training meta
  bash scripts/run_standalone_train_meta.sh [./save/classifier_mini-imagenet_resnet12/max-va.ckpt]
  [GPU] [./dataset] [1 or 5]
  # standalone evaluation
  bash scripts/run_eval.sh [./save/classifier_mini-imagenet_resnet12/max-va.ckpt] [./dataset]
  [GPU] [1 or 5]

## [Main Results](#Contents)

*The models on *miniImageNet*  use ResNet-12 as backbone, the channels in each block are **
64-128-256-512**, the backbone does **NOT** introduce any additional trick (e.g. DropBlock or wider
channel in some recent work).*

### 5-way accuracy (%) on *miniImageNet*

|method                                         |1-shot |5-shot|
|-----------------------------------------------|------ |------|
|[Baseline++](https://arxiv.org/abs/1904.04232) |51.87  |75.68 |
|[MetaOptNet](https://arxiv.org/abs/1904.03758) |62.64  |78.63 |
|Classifier-Baseline |58.91|77.76|
|Meta-Baseline        |63.17|79.26|
|Classifier-Baseline* |60.83|78.12|
|Meta-Baseline*       |62.37|78.28|

## [Running the code](#Contents)

### [1. Training Classifier-Baseline](#Contents)

``` python

python train_classifier.py --root_path ./dataset/ --device_id 0 --device_target GPU --run_offline True

```

```text

...
epoch 16, 1-shot, val acc 0.6024
epoch 16, 5-shot, val acc 0.7720
2.0m 53.8m/1.4h
train loss 0.3114, train acc 0.9227
epoch 17, 1-shot, val acc 0.6006
epoch 17, 5-shot, val acc 0.7745
2.0m 55.9m/1.4h
...

```

note:After each training epoch is completed, we have done inferences, so there is no need to
execute eval.py separately to view the results.

### [2. Training Meta-Baseline](#Contents)

``` python

python train_meta.py --num_shots 1 --load_encoder (dir) --root_path ./dataset/ --device_id 0 --device_target GPU --run_offline True

```

load_encoder is saved checkpoint of classifier-baseline.
The loss value and acc will be achieved as follows:

```text

...
epoch 5, train 0.3933|0.8947, val 1.0961|0.6113, 2.7m 13.5m/40.6m (@-1)
epoch 6, train 0.3977|0.8903, val 1.0951|0.6103, 2.7m 16.3m/40.7m (@-1)
epoch 7, train 0.3882|0.8931, val 1.0818|0.6219, 2.7m 19.0m/40.7m (@-1)
epoch 8, train 0.3752|0.8989, val 1.0839|0.6075, 2.7m 21.7m/40.7m (@-1)
epoch 9, train 0.3764|0.8967, val 1.0724|0.6116, 2.7m 24.5m/40.8m (@-1)
...

```

note:After each training epoch is completed, we have done inferences, so there is no need to
execute eval.py separately to view the results.

### [3. Test](#Contents)

``` python

python eval.py --load_encoder (dir) --num_shots 1 --root_path ./dataset/ --device_target GPU

```

## Inference Process

### [Export MindIR](#contents)

```shell

python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]

```

- The `ckpt_file` parameter is required.
- `file_format` should be in ["AIR", "MINDIR"].

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell

# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [ROOT_PATH] [NEED_PREPROCESS] [DEVICE_ID]

```

- `ROOT_PATH` the root path of validation dataset.
- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

## [Performance](#Contents)

### Training Performance

| Parameters                 | Ascend 910                                                   | GPU(RTX Titan) |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | Meta-Baseline | Meta-Baseline |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  NVIDIA RTX Titan-24G        |
| uploaded Date              | 12/28/2021 (month/day/year)                 | 12/28/2021 (month/day/year)          |
| MindSpore Version          | 1.3.0, 1.5.0                                             | 1.3.0, 1.5.0                                  |
| Dataset                    | mini-imagenet                              | mini-imagenet           |
| Training Parameters        | Epochs=20, steps per epoch=300, batch_size = 4 lr=0.001 | epoch=20, steps per epoch=300 batch_size = 4 lr=0.001|
| Optimizer                  | SGD                                                 | SGD                                  |
| Loss Function              | Softmax Cross Entropy                         | Softmax Cross Entropy          |
| outputs                    |  probability                               | probability               |
| Speed                      | 2.7 ms/step（1pcs, PyNative Mode）               | 2.7ms/step（1pcs, PyNative Mode） |
| Total time                 | about  50min |about 50min |
| Parameters (M)             | 8.7M                                     | 8.7M                        |
| Checkpoint for Fine tuning | 31.4M (.ckpt file)                                     | 31.4M (.ckpt file)                     |
| Scripts                    | [link](https://gitee.com/mindspore/models/tree/master/research/cv/meta-baseline) ||

### Inference Performance

| Parameters        | Ascend                                                      | GPU(RTX Titan)                                              |
| ----------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| Model Version     | Meta-Baseline | Meta-Baseline |
| Resource          | Ascend 910; OS Euler2.8                                     | NVIDIA RTX Titan-24G                                        |
| Uploaded Date     | 12/28/2021 (month/day/year)                                 | 12/28/2021 (month/day/year)                                 |
| MindSpore Version | 1.5.0, 1.3.0                                                | 1.5.0, 1.3.0                                                |
| Dataset           | mini-imagenet                                            | mini-imagenet                                           |
| batch_size        | 4                                                          | 4                                                          |
| outputs           | probability                                                     | probability                                                     |
| Accuracy          | See the table                                     |                                                             |

## [Citation](#Contents)

``` text

@misc{chen2020new,
    title={A New Meta-Baseline for Few-Shot Learning},
    author={Yinbo Chen and Xiaolong Wang and Zhuang Liu and Huijuan Xu and Trevor Darrell},
    year={2020},
    eprint={2003.04390},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```
