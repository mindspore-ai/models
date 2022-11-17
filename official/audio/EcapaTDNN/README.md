# Contents

[查看中文](./README_CN.md)

<!-- TOC -->

- [Contents](#contents)
- [ECAPA-TDNN Description](#ecapa-tdnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Export Process](#export-process)
        - [Export](#export)
    - [Inference Process](#inference-process)
        - [Inference](#inference)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# [ECAPA-TDNN Description](#contents)

ECAPA-TDNN is introduced in 2020，achieves best results in voxceleb1 evaluation trials. Comparing to vanilla tdnn，ECAPA-TDNN appends SE-block + Res2Block + Attentive Stat Pooling. By increasing the channels and enlarge the model size, the performance improve a lot.

[paper](https://arxiv.org/abs/2005.07143)：Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck. Interspeech 2020.

# [Model Architecture](#contents)

ECAPA-TDNN consists of several SE-Res2Blocks。The 1d convolution component of SE-Res2Block and its dilation parameters are same as conventional tdnn。The SE-Res2Block consists of **1×1 Conv**, **3×1 Conv**, **SE-block** and **res2net**。

# [Dataset](#contents)

## Dataset used: [voxceleb](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/>)

- Dataset size：7,000+ persons，more than 1 million wavs, total duration more than 2000 hours
    - [Voxceleb1](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html>) 100, 000+ wavs，1251 persons
        - train set：1211 persons
        - test set：40 persons
    - [Voxceleb2](<https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>) 1 Million+ wavs，6112 persons
        - train set：5994 persons
        - test set：118 persons
- Dataset format：voxceleb1 is wav，voxceleb2 is m4a
    - Note: we need to convert m4a to wav
- Prepare data：please follow the steps below
    - Download voxceleb1 and voxceleb2 dataset
    - Convert m4a to wav, you can use the script：https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830
    - Merge the train set of voxceleb1 and voxceleb2 to wav/ folder. i.e. voxceleb12/wav/id*/*.wav
    - The directory structure is as follows:

      ``` bash
      voxceleb12
      ├── meta
      └── wav
          ├── id10001
          │   ├── 1zcIwhmdeo4
          │   ├── 7gWzIy6yIIk
          │   └── ...
          ├── id10002
          │   ├── 0_laIeN-Q44
          │   ├── 6WO410QOeuo
          │   └── ...
          ├── ...
      ```

    - test on voxceleb1, copy [trials](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt) to voxceleb1/veri_test2.txt
    - The directory structure of voxceleb1 is as follows:

      ``` bash
      voxceleb1
      ├──veri_test2.txt
      └── wav
          ├── id10001
          │   ├── 1zcIwhmdeo4
          │   ├── 7gWzIy6yIIk
          │   └── ...
          ├── id10002
          │   ├── 0_laIeN-Q44
          │   ├── 6WO410QOeuo
          │   └── ...
          ├── ...
      ```

## Generate training and test data

As mindspore do not support fbank feature extraction online, we need to do it offline. We augment the raw wav data and extract fbank feature out as the training data of mindspore script. We borrow feature extraction script from speechbrain, and make a bit of edition.

- Run data_prepare.sh, it will cost several hours to do 5x augmentation, consume 1.3T disk space. To achieve target precision, we need 50x augmentation, which will cost 13T disk space.

  ``` bash
  bash data_prepare.sh
  ```

- Then run script below, if you want to accelerate dataload io

  ``` bash
  python3 merge_data.py hparams/prepare_train.yaml
  ```

- Note: you can see how to set parameters in [Quick Start](#quick-start)

# Features

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

# [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend processor.
- Framework
    - python3 and its dependent packages
        - run `pip3 install -r requirements.txt` after install python3
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below.
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore and python3, you can start training and evaluation as follows:

  ```text
  # Change data set path on prepare_train.yaml file
  data_folder: /home/abc000/data/voxceleb12  # path to train set
  feat_folder: /home/abc000/data/feat_train/ # path to store traing data for mindspore

  # Change data set path on prepare_eval.yaml file
  data_folder: /home/abc000/data/voxceleb1/ # path to test set
  feat_eval_folder: /home/abc000/data/feat_eval/ # path to store test data for mindspore
  feat_norm_folder:  /home/abc000/data/feat_norm/ # path to store norm data for mindspore

  # Change data set path on edapa-tdnn_config.yaml file
  train_data_path: /home/abc000/data/feat_train/

  ```

  ```bash
  # run training example
  python train.py > train.log 2>&1 &

  # For Ascend device, standalone training example(1p) by shell script
  bash run_standalone_train_ascend.sh DEVICE_ID

  # For Ascend device, distributed training example(8p) by shell script
  bash run_distribute_train.sh RANK_TABLE_FILE

  # run evaluation example
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  For distributed training, a hccl configuration file with JSON format needs to be created in advance.

  Please follow the instructions in the link below:

  <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools>.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
    ModelZoo_ECAPA-TDNN
    ├── ascend310_infer                              # 310 inference code
    ├── data_prepare.sh                              # to prepare training and eval data for mindspore
    ├── ecapa-tdnn_config.yaml                       # parameter for training and eval
    ├── eval_data_prepare.py                         # script to prepare eval data
    ├── eval.py                                      # eval script
    ├── export.py                                    # export mindir 310 model
    ├── hparams                                      # parameter to prepare training and eval data for mindspore
    ├── README_CN.md                                 # description file
    ├── README.md                                    # description file
    ├── requirements.txt                             # python dependent packages
    ├── scripts                                      # script for train and eval etc.
    ├── src                                          # model related python script
    ├── train_data_prepare.py                        # script to prepare train data
    ├── merge_data.py                                # script to merge data to few batchs to accelerate data io
    └── train.py                                     # train script
```

## [Script Parameters](#contents)

change train feature extraction parameter on hparams/prepare_train.yaml

```text
  output_folder: ./augmented/                             # path to store intermediate result
  save_folder: !ref <output_folder>/save/
  feat_folder: /home/abc000/data/feat_train/              # path to store training fbank feature
  # Data files
  data_folder: /home/abc000/data/voxceleb12               # pato to raw wav train data
  train_annotation: !ref <save_folder>/train.csv          # pre generated csv file, regenerate if not exist
  valid_annotation: !ref <save_folder>/dev.csv            # pre generated csv file, regenerate if not exist
```

change eval feature extraction parameter on hparams/prepare_train.yaml

```text
  output_folder: ./augmented_eval/                     # path to store intermediate result
  feat_eval_folder: /home/abc000/data/feat_eval/       # path to store eval related fbank feature
  feat_norm_folder:  /home/abc000/data/feat_norm/      # path to store norm related fbank feature
  data_folder: /home/abc000/data/voxceleb1/            # path to voxceleb1
  save_folder: !ref <output_folder>/save/              # path to store intermediate result
```

Parameters for both training and evaluation can be set in edapa-tdnn_config.yaml

- config ECAPA-TDNN and dataset

  ```text
    inChannels: 80                                                  # input channel size, same as the dim of fbank feature
    channels: 1024                                                  # channel size of middle layer feature map
    base_lrate: 0.000001                                            # base learning rate of cyclic LR
    max_lrate: 0.0001                                               # max learning rate of cyclic LR
    momentum: 0.95                                                  # weight decay for optimizer
    weightDecay: 0.000002                                           # momentum for optimizer
    num_epochs: 3                                                   # training epoch
    minibatch_size: 192                                             # batch size
    emb_size: 192                                                   # embedding dim
    step_size: 65000                                                # steps to achieve max learning rate cyclic LR
    CLASS_NUM: 7205                                                 # speaker num pf voxceleb1&2
    pre_trained: False                                              # if pre-trained model exist

    train_data_path: "/home/abc000/data/feat_train/"                # path to fbank training data
    keep_checkpoint_max: 30                                         # max model number to save
    checkpoint_path: "/ckpt/train_ecapa_vox2_full-2_664204.ckpt"    # path to pre-trained model
    ckpt_save_dir: "./ckpt/"                                        # path to store train model

    # eval
    eval_data_path: "/home/abc000/data/feat_eval/"                  # path to eval fbank data
    veri_file_path: "veri_test_bleeched.txt"                        # trials
    model_path: "ckpt/train_ecapa_vox2_full-2_664204.ckpt"          # path of eval model
    score_norm: "s-norm"                                            # if do norm
    train_norm_path: "/data/dataset/feat_norm/"                     # fbank data for norm

  ```

more detail please refer `edapa-tdnn_config.yaml`

## [Training Process](#contents)

### Training

  ```bash
    python3 train.py > train.log 2>&1 &
    OR bash scripts/run_standalone_train_ascend.sh
  ```

The python command above will run in the background, you can view the results through the file `train.log`. After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```bash
    # grep "loss: " train.log
    2022-02-13 13:58:33.898547, epoch:0/15, iter-719000/731560, aver loss:1.5836, cur loss:1.1664, acc_aver:0.7349
    2022-02-13 14:08:44.639722, epoch:0/15, iter-720000/731560, aver loss:1.5797, cur loss:1.1057, acc_aver:0.7363
  ...
  ```

  The model checkpoint will be saved in the ckpt_save_dir.

### Distributed Training

  ```python
  bash scripts/run_distribute_train_ascend.sh
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log`. The loss value will be achieved as follows:

  ```bash
  # grep "loss: " train.log
    2022-02-13 13:58:33.898547, epoch:0/15, iter-719000/731560, aver loss:1.5836, cur loss:1.1664, acc_aver:0.7349
    2022-02-13 14:08:44.639722, epoch:0/15, iter-720000/731560, aver loss:1.5797, cur loss:1.1057, acc_aver:0.7363
  ...
  ...
  ```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on voxceleb1 dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "/username/xxx/saved_model/xxx_20-215_176000.ckpt".

  ```bash
    bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  The above python command will run in the background. You can view the result in eval.log：

  ```bash
    # grep "eer" eval.log
    eer xxx:0.0082
  ```

  we can also set the 'cut_wav' parameter in ecapa-tdnn_config.yaml to get the eer of 3s wav which is same as the 310 inference result.

  ```text
  wav_cut: true                                              # if cut the test wav to 3s(same as train data), default false
  ```

## [Export Process](#contents)

### [Export](#content)

```shell
  python3 export.py
```

## [Inference Process](#contents)

### [Inference](#content)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by export.py. Currently, only batchsize 1 is supported.

- Do inference by ascend 310 on voxceleb1.

  First set parameter veri_file_path in ecapa-tdnn_config.yaml(veri_test_bleeched.txt is stored in feat_eval folder). As 310 only support fixed length of input，we cut input wav to fixed length, such as 3s. you can get result in acc.log.

  ```shell
    # Ascend310 inference
    bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]

    cat acc.log | grep eer
    eer sub mean: 0.0248
  ```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | ECAPA-TDNN                                                   |
| Resource                   | Ascend 910；CPU 2.60GHz，56cores；Memory 755G; OS Euler2.8   |
| uploaded Date              | 3/1/2022                                                     |
| MindSpore Version          | 1.5.1                                                        |
| Dataset                    | voxceleb1&voxceleb2                                          |
| Training Parameters        | epoch=3, steps=733560\*epoch, batch\_size = 192, min\_lr=0.000001, max\_lr=0.0001          |
| Optimizer                  | Adam                                                         |
| Loss Function              | AAM-Softmax                                                  |
| outputs                    | probability                                                  |
| Speed                      | 1pc: 17 steps/sec                                            |
| Total time                 | 1pc: 264 hours                                               |
| Loss                       | 1.1                                                          |
| Parameters (M)             | 13.0                                                         |
| Checkpoint for Fine tuning | 254 (.ckpt file)                                             |
| infer model                | 76.60M(.mindir文件)

### Inference Performance

#### Evaluation ECAPA-TDNN on voxceleb1

| Parameters          | Ascend                        |
| ------------------- | ---------------------------   |
| Model Version       | ECAPA-TDNN                    |
| Resource            |  Ascend 910；OS Euler2.8      |
| uploaded Date       | 2022-03-01                    |
| MindSpore Version   | 1.5.1                         |
| Dataset             | voxceleb1-eval, 4715 wavs     |
| batch_size          | 1                             |
| outputs             | probability                   |
| accuracy            | 1p: EER=0.82%                 |
| infer model         | 76.60M(.mindir)               |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models/tree/master).
