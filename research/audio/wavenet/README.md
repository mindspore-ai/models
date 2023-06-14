# Contents

- [WaveNet Description](#WaveNet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
    - [Convert Process](#convert-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [WaveNet Description](#contents)

WaveNet is a deep neural network for generating raw audio waveforms. The model is fully probabilistic and autoregressive, with the predictive distribution for each audio sample conditioned on all previous ones. We support training and evaluation  on Ascend, GPU, and CPU.

[Paper](https://arxiv.org/pdf/1609.03499.pdf): Oord A, Dieleman S, Zen H, et al. Wavenet: A generative model for raw audio.

# [Model Architecture](#contents)

The current model consists of a pre-convolution layer, followed by several residual block which has residual and skip connection with gated activation units.
Finally, post convolution layers are added to predict the distribution.

# [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [The LJ Speech Dataset](<https://keithito.com/LJ-Speech-Dataset>)

- Dataset size：2.6G
- Data format：audio clips(13100) and transcription

- The dataset structure is as follows:

    ```path
    .
    └── LJSpeech-1.1
        ├─ wavs                  //audio clips files
        └─ metadata.csv           //transcripts
    ```

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

**Note that some of the scripts described below are not included our code**. These scripts should first be download them from [r9y9](https://github.com/r9y9/wavenet_vocoder) and added into this project.

```path
.
├── audio
    └──wavenet
        ├── scripts
        │   ├──run_distribute_train_ascend.sh    // launch distributed training with Ascend platform
        │   ├──run_distribute_train_gpu.sh       // launch distributed training with GPU platform
        │   ├──run_eval_ascend.sh                // launch evaluation with Ascend platform
        │   ├──run_eval_gpu.sh                   // launch evaluation with GPU platform
        │   ├──run_eval_cpu.sh                   // launch evaluation with CPU platform
        │   ├──run_standalone_train_ascend.sh    // launch standalone training with Ascend platform
        │   ├──run_standalone_train_gpu.sh       // launch standalone training with GPU platform
        │   └──run_standalone_train_cpu.sh       // launch standalone training with CPU platform
        ├── datasets                             // Process audio files for generating train/evaluate data
        ├── egs                                  // Note the egs folder should be downloaded from the above link  
        ├── utils                                // Note the utils folder should be downloaded from the above link  
        ├── audio.py                             // Audio utils. Note this script should be downloaded from the above link
        ├── compute-meanvar-stats.py             // Compute mean-variance Normalization stats. Note this script should be downloaded from the above link
        ├── evaluate.py                          // Evaluation
        ├── export.py                            // Convert mindspore model to air/mindir model
        ├── hparams.py                           // Hyper-parameter configuration. Note this script should be downloaded from the above link
        ├── mksubset.py                          // Make subset of dataset. Note this script should be downloaded from the above link
        ├── preprocess.py                        // Preprocess dataset. Note this script should be downloaded from the above link
        ├── preprocess_normalize.py              // Perform meanvar Normalization to preprocessed features. Note this script should be downloaded from the above link
        ├── README.md                            // Descriptions about WaveNet
        ├── train.py                             // Training scripts
        ├── train_pytorch.py                     // Note this script should be downloaded from the above link. The initial name of this script is train.py in the project from the link
        ├── src
        │   ├──__init__.py
        │   ├──dataset.py                        // Generate dataloader and data processing entry
        │   ├──callback.py                       // Callbacks to monitor the training
        │   ├──lr_generator.py                   // Learning rate generator
        │   └──loss.py                           // Loss function definition
        └── wavenet_vocoder
            ├──__init__.py
            ├──conv.py                           // Extended 1D convolution
            ├──mixture.py                        // Loss function for training and sample function for testing
            ├──modules.py                        // Modules for Wavenet construction
            ├──upsample.py                       // Upsample layer definition
            ├──util.py                           // Utils. Note this script should be downloaded from the above link
            ├──wavenet.py                        // WaveNet networks
            └──tfcompat                          // Note this script should be downloaded from the above link
               ├──__init__.py
               └──hparam.py                      // Param management tools
```

## [Script Parameters](#contents)

### Training

```text
usage: train.py  [--data_path DATA_PATH] [--preset PRESET]
                 [--checkpoint_dir CHECKPOINT_DIR] [--checkpoint CHECKPOINT]
                 [--speaker_id SPEAKER_ID] [--platform PLATFORM]
                 [--mode_name MODE] [--is_distributed IS_DISTRIBUTED]

options:
    --data_path                  dataset path
    --preset                     path of preset parameters (json)
    --checkpoint_dir             directory of saving model checkpoints
    --checkpoint                 pre-trained ckpt path, default is "./checkpoints"
    --speaker_id                 specific speaker of data in case for multi-speaker datasets, not used currently
    --platform                   specify platform to be used, defeault is "GPU"
    --mode_name                  specify graph mode, default is "GRAPH"
    --is_distributed             whether distributed training or not

```

### Evaluation

```text
usage: evaluate.py  [--data_path DATA_PATH] [--preset PRESET]
                    [--pretrain_ckpt PRETRAIN_CKPT] [--is_numpy]
                    [--output_path OUTPUT_PATH] [--speaker_id SPEAKER_ID]
                    [--platform PLATFORM]
options:
    --data_path                  dataset path
    --preset                     path of preset parameters (json)
    --pretrain_ckpt              pre-trained ckpt path
    --is_numpy                   whether using numpy for inference or not
    --output_path                path to save synthesized audio
    --speaker_id                 specific speaker of data in case for multi-speaker datasets, not used currently
    --platform                   specify platform to be used, defeault is "GPU"
```

More parameters for training and evaluation can be set in file `hparams.py`.

## [Training Process](#contents)

Before your first training, some dependency scripts should be downloaded and placed in correct directory as described in [Script and Sample Code].
After that, raw data should be pre-processed by using the scripts in `egs`. The directory of egs is as follows:

```path
.
├── egs
    ├──gaussian
    │  ├──conf
    │  │  ├──gaussian_wavenet.json
    │  │  └──gaussian_wavenet_demo.json
    │  └──run.sh
    ├──mol
    │  ├──conf
    │  │  ├──mol_wavenet.json
    │  │  └──mol_wavenet_demo.json
    │  └──run.sh
    ├──mulaw256
    │  ├──conf
    │  │  ├──mulaw_wavenet.json
    │  │  └──mulaw_wavenet_demo.json
    │  └──run.sh
    └──README.md
```

In this project, three different losses are implemented to train the network:

- mulaw256: categorical output distribution. The input is 8-bit mulaw quantized waveform.
- mol: discretized mix logistic loss. The input is 16-bit raw audio.
- gaussian: mix gaussian loss. The input is 16-bit raw audio.

The three folder gaussian, mol, mulaw is used to generate corresponding training data respectively. For example, To generate the training data for
mix gaussian loss, you should first modify the `run.sh` in line 28. Change `conf/gaussian_wavenet_demo.json` to
`conf/gaussian_wavenet.json`. We use the default parameter in `gaussian_wavenet.json`. By this setting, data will be generated to adapt to mix gaussian loss and
some parameters in `hparams.py` will be covered by that in `gaussian_wavenet.json`. You can also define your own hyper-parameter json here. After the modification,
The following command can be ran for data generation. Note that if you want to change values of some parameters, you may need to modify in `gaussian_wavenet.json` instead of  `hparams.py` since `gaussian_wavenet.json` may cover that in`hparams.py`.

```bash
bash run.sh --stage 0 --stop-stage 0 --db-root /path_to_dataset/LJSpeech-1.1/wavs
bash run.sh --stage 1 --stop-stage 1
```

After the processing, the directory of gaussian will be as follows:

```path
.
├── gaussian
    ├──conf
    ├──data
    ├──exp
    └──dump
       └──lj
          └──logmelspectrogram
             ├──org
             └──norm
                ├──train_no_dev
                ├──dev
                └──eval
```

The train_no_dev folder contains the final training data. For mol and gaussian, the process is the same. When the training data is prepared,
you can run the following command to train the network:

```bash
Standalone training
Ascend:
bash ./scripts/run_standalone_train_ascend.sh train.py [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_save_ckpt] [DEVICE_ID]

GPU:
bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID] [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_save_ckpt]

CPU:
bash ./scripts/run_standalone_train_cpu.sh [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_save_ckpt]

Distributed training
Ascend:
bash ./scripts/run_distributed_train_ascend.sh [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_save_ckpt] [path_to_hccl_config_file] [RANK_SIZE] [FIRST_DEVICE_ID]

GPU(8p):
bash ./scripts/run_distribute_train_gpu.sh [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_save_ckpt]
```

## [Evaluation Process](#contents)

WaveNet has a process of auto-regression and this process currently cannot be run in Graph mode(place the auto-regression into `construct`). Therefore, we implement the process in a common function. Here, we provide two kinds of ways to realize the function: using Numpy or using MindSpore ops. One can set `is_numpy` to determine which mode is used. We recommend using numpy on GPU since it is much faster than using MindSpore ops. This is because the auto-regression process only calls some simple operation like Matmul and Bias_add. Unlike Graph mode, there will exist some fixed cost each step and this leads to a lower speed. For more information, please refer to
this [link](https://bbs.huaweicloud.com/forum/thread-94852-1-1.html)

```bash
Evaluation
Ascend (using mindspore):
bash ./scripts/run_eval_ascend.sh [DEVICE_ID] evaluate.py [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/eval] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_load_ckpt] [path_to_save_audio]

Ascend(using numpy):
bash ./scripts/run_eval_ascend.sh [DEVICE_ID] evaluate.py [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/eval] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_load_ckpt] is_numpy [path_to_save_audio]

GPU (using mindspore):
bash ./scripts/run_eval_gpu.sh [CUDA_DEVICE_ID] [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/eval] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_load_ckpt] [path_to_save_audio]

GPU (using numpy):
bash ./scripts/run_eval_gpu.sh [CUDA_DEVICE_ID] [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/eval] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_load_ckpt] is_numpy [path_to_save_audio]

CPU:
bash ./scripts/run_eval_cpu.sh [/path_to_egs/egs/gaussian/dump/lj/logmelspectrogram/norm/eval] [/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json] [path_to_load_ckpt] [is_numpy] [path_to_save_audio]
```

## [Convert Process](#contents)

```bash
Ascend:
python export.py --preset=/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json --checkpoint_dir=path_to_dump_hparams --pretrain_ckpt=path_to_load_ckpt --platform=Ascend

GPU:
python export.py --preset=/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json --checkpoint_dir=path_to_dump_hparams --pretrain_ckpt=path_to_load_ckpt

CPU:
python export.py --preset=/path_to_egs/egs/gaussian/conf/gaussian_wavenet.json --checkpoint_dir=path_to_dump_hparams --pretrain_ckpt=path_to_load_ckpt --platform=CPU
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters           | GPU                                                          | Ascend                                                       |
| -------------------- | ------------------------------------------------------------ | :----------------------------------------------------------- |
| Resource             | NV SMX2 V100-32G                                             | Ascend 910                                                   |
| uploaded Date        | 01/14/2021 (month/day/year)                                  | 09/27/2021 (month/day/year)                                  |
| MindSpore Version    | 1.0.0                                                        | 1.3.0                                                        |
| Dataset              | LJSpeech-1.1                                                 | LJSpeech-1.1                                                 |
| Training Parameters  | 1p, epoch=600(max), steps=1635 * epoch, batch_size = 8, lr=1e-3 | 8p, epoch=2000(max), steps=1635*epoch, batch_size =24, lr=2e-3 |
| Optimizer            | Adam                                                         | Adam                                                         |
| Loss Function        | SoftmaxCrossEntropyWithLogits/discretized_mix_logistic/mix_gaussian | SoftmaxCrossEntropyWithLogits/mix_gaussian                   |
| Loss                 | around 2.0(mulaw256)/around 4.5(mol)/around -6.0(gaussian)   | around -5.0(gaussian)                                        |
| Speed                | 1p 1.467s/step                                               |                                                              |
| Total time: training | 1p(mol/gaussian): around 4 days; 2p(mulaw256):around 1 week  | 8p: around 1 week                                            |
| Checkpoint           | 59.79MM/54.87M/54.83M (.ckpt file)                           | 42.73M (.ckpt file)                                          |
| Scripts              | [WaveNet script](https://gitee.com/mindspore/models/tree/r2.0/research/audio/wavenet) | [WaveNet script](https://gitee.com/mindspore/models/tree/r2.0/research/audio/wavenet) |

### Inference Performance

Audio samples will be demonstrated online soon.

# [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).

## FAQ

Please refer to [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) to get some common FAQ.

- **Q: What third-party packages are required and how to install them?**

  **A**: nnmnkwii, librosa(preferably 0.4.0) and tqdm are needed in order to run wavenet. Instructions on how to install these packages can be found at [nnmnkwii](https://github.com/r9y9/nnmnkwii). On windows systems, specifically, installing nnmnkwii requires pysptk installed, which then requires Microsoft Visual C++ 14.0 installed previously.
