# Contents

- - [jasper Description](#CenterNet-description)
  - [Model Architecture](#Model-Architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-parameters)
    - [Script Parameters](#script-parameters)
    - [Training and eval Process](#training-process)
    - [Export](#Export)
  - [Performance](#performance)
    - [Training Performance](#training-performance)
    - [Inference Performance](#inference-performance)
  - [ModelZoo Homepage](#modelzoo-homepage)

## [Jasper Description](#contents)

Jasper is an end-to-end speech recognition models which is trained with CTC loss. Jasper model uses only 1D convolutions, batch normalization, ReLU, dropout, and residual connections. We support training and evaluation on CPU and GPU.

[Paper](https://arxiv.org/pdf/1904.03288v3.pdf): Jason Li, et al. Jasper: An End-to-End Convolutional Neural Acoustic Model.

## [Model Architecture](#contents)

Jasper is an end-to-end neural acoustic model that is based on convolutions. In the audio processing stage, each frame is transformed into mel-scale spectrogram features, which the acoustic model takes as input and outputs a probability distribution over the vocabulary for each frame. The acoustic model has a modular block structure and can be parametrized accordingly: a Jasper BxR model has B blocks, each consisting of R repeating sub-blocks.
Each sub-block applies the following operations in sequence: 1D-Convolution, Batch Normalization, ReLU activation, and Dropout.
Each block input is connected directly to the last subblock of all following blocks via a residual connection, which is referred to as dense residual in the paper. Every block differs in kernel size and number of filters, which are increasing in size from the bottom to the top layers. Irrespective of the exact block configuration parameters B and R, every Jasper model has four additional convolutional blocks: one immediately succeeding the input layer (Prologue) and three at the end of the B blocks (Epilogue).

## [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [LibriSpeech](<http://www.openslr.org/12>)

Train Data：
train-clean-100: [6.3G] (training set of 100 hours "clean" speech)
train-clean-360.tar.gz [23G] (training set of 360 hours "clean" speech)
train-other-500.tar.gz [30G] (training set of 500 hours "other" speech)
Val Data：
dev-clean.tar.gz [337M] (development set, "clean" speech)
dev-other.tar.gz [314M] (development set, "other", more challenging, speech)
Test Data:
test-clean.tar.gz [346M] (test set, "clean" speech )
test-other.tar.gz [328M] (test set, "other" speech )
Data format：wav and txt files

## [Environment Requirements](#contents)

Hardware（GPU）
  Prepare hardware environment with GPU processor.
Framework
  [MindSpore](https://www.mindspore.cn/install/en)
For more information, please check the resources below：
  [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```path
.
└─audio
    └─jasper
        │  eval.py                          //inference file
        │  labels.json                      //label file
        │  pt2mind.py                       //pth transform to ckpt file
        |  create_mindrecord.py             //transform data to mindrecord
        │  README-CN.md                     //Chinese readme
        │  README.md                        //English readme
        │  requirements.txt                 //required library file
        │  train.py                         //train file
        │
        ├─scripts
        │      download_librispeech.sh      //download data
        │      preprocess_librispeech.sh    //preprocess data
        │      run_distribute_train_gpu.sh  //8 GPU cards train
        │      run_eval_cpu.sh              //CPU evaluate
        │      run_eval_gpu.sh              //GPU evaluate
        │      run_standalone_train_cpu.sh  //one CPU train
        │      run_standalone_train_gpu.sh  //one GPU train
        │
        ├─src
        │      audio.py                     //preprocess data
        │      callback.py                  //callback
        │      cleaners.py                  //preprocess data
        │      config.py                    //jasper config
        │      dataset.py                   //preporcess data
        │      decoder.py                   //Third-party decoders
        │      eval_callback.py             //evaluate callback
        │      greedydecoder.py             //refactored greedydecoder
        │      jasper10x5dr_speca.yaml      //jasper model's config
        │      lr_generator.py              //learning rate
        │      model.py                     //training model
        │      model_test.py                //inference model
        │      number.py                    //preprocess data
        │      text.py                      //preprocess data
        │      __init__.py
        │
        └─utils
                convert_librispeech.py      //convert data
                download_librispeech.py     //download data
                download_utils.py           //download utils
                inference_librispeech.csv   //links to inference data
                librispeech.csv             //links to all data
                preprocessing_utils.py      //preprocessing utils
                __init__.py

```

### [Script Parameters](#contents)

#### Training

```text
usage: train.py  [--use_pretrained USE_PRETRAINED]
                 [--pre_trained_model_path PRE_TRAINED_MODEL_PATH]
                 [--is_distributed IS_DISTRIBUTED]
                 [--bidirectional BIDIRECTIONAL]
                 [--device_target DEVICE_TARGET]
options:
    --pre_trained_model_path    pretrained checkpoint path, default is ''
    --is_distributed            distributed training, default is False
    is True. Currently, only bidirectional model is implemented
    --device_target             device where the code will be implemented: "GPU" | "CPU", default is "GPU"
```

#### Evaluation

```text
usage: eval.py  [--bidirectional BIDIRECTIONAL]
                [--pretrain_ckpt PRETRAIN_CKPT]
                [--device_target DEVICE_TARGET]

options:
    --bidirectional              whether to use bidirectional RNN, default is True. Currently, only bidirectional model is implemented
    --pretrain_ckpt              saved checkpoint path, default is ''
    --device_target              device where the code will be implemented: "GPU" | "CPU", default is "GPU"
```

#### Options and Parameters

Parameters for training and evaluation can be set in file `config.py`

```text
config for training.
    epochs                       number of training epoch, default is 70
```

```text
config for dataloader.
    train_manifest               train manifest path, default is 'data/libri_train_manifest.json'
    val_manifest                 dev manifest path, default is 'data/libri_val_manifest.json'
    batch_size                   batch size for training, default is 8
    labels_path                  tokens json path for model output, default is "./labels.json"
    sample_rate                  sample rate for the data/model features, default is 16000
    window_size                  window size for spectrogram generation (seconds), default is 0.02
    window_stride                window stride for spectrogram generation (seconds), default is 0.01
    window                       window type for spectrogram generation, default is 'hamming'
    speed_volume_perturb         use random tempo and gain perturbations, default is False, not used in current model
    spec_augment                 use simple spectral augmentation on mel spectograms, default is False, not used in current model
    noise_dir                    directory to inject noise into audio. If default, noise Inject not added, default is '', not used in current model
    noise_prob                   probability of noise being added per sample, default is 0.4, not used in current model
    noise_min                    minimum noise level to sample from. (1.0 means all noise, not original signal), default is 0.0, not used in current model
    noise_max                    maximum noise levels to sample from. Maximum 1.0, default is 0.5, not used in current model
```

```text
config for optimizer.
    learning_rate                initial learning rate, default is 3e-4
    learning_anneal              annealing applied to learning rate after each epoch, default is 1.1
    weight_decay                 weight decay, default is 1e-5
    momentum                     momentum, default is 0.9
    eps                          Adam eps, default is 1e-8
    betas                        Adam betas, default is (0.9, 0.999)
    loss_scale                   loss scale, default is 1024
```

```text
config for checkpoint.
    ckpt_file_name_prefix        ckpt_file_name_prefix, default is 'Jasper'
    ckpt_path                    path to save ckpt, default is 'checkpoints'
    keep_checkpoint_max          max number of checkpoints to save, delete older checkpoints, default is 10
```

## [Training and Eval process](#contents)

Before training, the dataset should be processed.

``` bash
bash scripts/download_librispeech.sh
bash scripts/preprocess_librispeech.sh
python createmindrecord.py //transform data to mindrecord
```

dataset directory structure is as follows:

```path
    .
    |--LibriSpeech
    │  |--train-clean-100-wav
    │  │--train-clean-360-wav
    │  │--train-other-500-wav
    │  |--dev-clean-wav
    │  |--dev-other-wav
    │  |--test-clean-wav
    │  |--test-other-wav
    |--librispeech-train-clean-100-wav.json,librispeech-train-clean-360-wav.json,librispeech-train-other-500-wav.json,librispeech-dev-clean-wav.json,librispeech-dev-other-wav.json,librispeech-test-clean-wav.json,librispeech-test-other-wav.json
```

The three *.json file stores the absolute path of the corresponding
data. After obtaining the 3 json file, you should modify the configurations in `src/config.py`.
For training config, the train_manifest should be configured with the path of `libri_train_manifest.json` and for eval config, it should be configured
with `libri_test_other_manifest.json` or `libri_train_manifest.json`, depending on which dataset is evaluated.

```shell
...
train config
"Data_dir": '/data/dataset',
"train_manifest": ['/data/dataset/librispeech-train-clean-100-wav.json',
                   '/data/dataset/librispeech-train-clean-360-wav.json',
                   '/data/dataset/librispeech-train-other-500-wav.json'],
"mindrecord_format": "/data/jasper_tr{}.md",
"mindrecord_files": [f"/data/jasper_tr{i}.md" for i in range(8)]

eval config
"DataConfig":{
     "Data_dir": '/data/inference_datasets',
     "test_manifest": ['/data/inference_datasets/librispeech-dev-clean-wav.json'],
}

```

Before training, some requirements should be installed, including `librosa` and `Levenshtein`
After installing MindSpore via the official website and finishing dataset processing, you can start training as follows:

```shell

# standalone training gpu
bash ./scripts/run_standalone_train_gpu.sh [DEVICE_ID]

# standalone training cpu
bash ./scripts/run_standalone_train_cpu.sh

# distributed training gpu
bash ./scripts/run_distribute_train_gpu.sh

```

The following script is used to evaluate the model. Note we only support greedy decoder now and before run the script:

```shell

# eval on cpu
bash ./scripts/run_eval_cpu.sh [PATH_CHECKPOINT]

# eval on gpu
bash ./scripts/run_eval_gpu.sh [DEVICE_ID] [PATH_CHECKPOINT]

```

## [Model Description](#contents)

### [Performance](#contents)

#### Training Performance

| Parameters           | Jasper                                                       |
| -------------------- | ------------------------------------------------------------ |
| Resource             | NV SMX2 V100-32G                                             |
| uploaded Date        | 2/7/2022 (month/day/year)                                    |
| MindSpore Version    | 1.8.0                                                        |
| Dataset              | LibriSpeech                                                  |
| Training Parameters  | 8p, epoch=70, steps=1088 * epoch, batch_size = 64, lr=3e-4   |
| Optimizer            | Adam                                                         |
| Loss Function        | CTCLoss                                                      |
| outputs              | probability                                                  |
| Loss                 | 0.2-0.7                                                      |
| Speed                | 8p 2.7s/step                                                 |
| Total time: training | 8p: around 194 h;                                            |
| Checkpoint           | 991M (.ckpt file)                                            |
| Scripts              | [Jasper script](https://gitee.com/mindspore/models/tree/master/research/audio/jasper) |

#### Inference Performance

| Parameters          | Jasper                     |
| ------------------- | -------------------------- |
| Resource            | NV SMX2 V100-32G           |
| uploaded Date       | 2/7/2022 (month/day/year)  |
| MindSpore Version   | 1.8.0                      |
| Dataset             | LibriSpeech                |
| batch_size          | 64                         |
| outputs             | probability                |
| Accuracy(dev-clean) | 8p: WER: 5.754  CER: 2.151 |
| Accuracy(dev-other) | 8p: WER: 19.213 CER: 9.393 |
| Model for inference | 330M (.mindir file)        |

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
