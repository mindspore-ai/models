# Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [MelGAN Description](#melgan-description)
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
        - [Inference Performance](#inference-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MelGAN Description](#contents)

MelGAN, a GANs that can generate high quality coherent waveforms by introducing a set of architectural changes and simple training techniques. This network runs at more than 100x faster than realtime on GTX 1080Ti GPU and more than 2x faster than real-time on CPU, without any hardware specific optimization tricks.

[Paper](https://arxiv.org/abs/1910.06711):  Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville. "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis.".

# [Model Architecture](#contents)

Specifically, the MelGAN model is non-autoregressive, fully convolutional, with significantly fewer parameters than competing models and generalizes to unseen speakers for mel-spectrogram inversion. The generator consists 4 upsample layers and 4 residual stacks, and the discriminator is a multi-scale architecture. Unlike the structure of the paper design, we modify the size of some convolution kernels in the discriminator， and we use 1D-convolution instead of avgpool in the discriminator.

# [Dataset](#contents)

Dataset used: [LJ Speech](<https://keithito.com/LJ-Speech-Dataset/>)

- Dataset size：2.6GB，13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.

- Data format：Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 22050 Hz
    - The audio data needs to be processed to a mel-spectrum, and you can refer to the script in [mel-spectrogram data creation](https://github.com/seungwonpark/melgan/blob/master/preprocess.py). Non CUDA environment needs to delete `. cuda()` in `utils/stfy.py`. To save data in the `npy` format, `preprocess.py` also needs to be modified. As follows:

    ```
    # 37 - 38 lines
    melpath = wavpath.replace('.wav', '.npy').replace('wavs', 'mel')
    if not os.path.exists(os.path.dirname(melpath)):
        os.makedirs(os.path.dirname(melpath), exist_ok=True)
    np.save(melpath, mel.squeeze(0).detach().numpy())
    ```

    - The directory structure is as follows:

      ```
        ├── dataset
            ├── val
            │   ├─ wavform1.npy
            │   ├─ ...
            │   └─ wavformn.npy
            ├── train
                ├─ wav
                │    ├─wavform1.wav
                │    ├─ ...
                │    └─wavformn.wav
                └─ mel
                    ├─wavform1.npy
                    ├─ ...
                    └─wavformn.npy
      ```

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below.
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

  ```yaml
  # Change data set path on yaml file, take training LJSpeech as an example
  data_path:/home/LJspeech/dataset/

  # Add checkpoint path parameters on yaml file before continue Training
  checkpoint_path:/home/model/saved_model/melgan_20-215_176000.ckpt
  ```

  ```python
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

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

  ```bash
    # run distributed training on modelarts example
    # (1) First, Perform a or b.
    #       a. Set "enable_modelarts=True" on yaml file.
    #          Set other parameters on yaml file you need.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add other parameters on the website UI interface.
    # (2) Set the code directory to "/path/MelGAN" on the website UI interface.
    # (3) Set the startup file to "train.py" on the website UI interface.
    # (4) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (5) Create your job.

    # run evaluation on modelarts example
    # (1) Copy or upload your trained model to S3 bucket.
    # (2) Perform a or b.
    #       a. Set "enable_modelarts=True" on yaml file.
    #          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
    #          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
    #       b. Add "enable_modelarts=True" on the website UI interface.
    #          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
    #          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
    # (3) Set the code directory to "/path/MelGAN" on the website UI interface.
    # (4) Set the startup file to "eval.py" on the website UI interface.
    # (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
    # (6) Create your job.
  ```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── melgan
    ├── README.md                     // descriptions about melgan
    ├── README_CN.md                  // descriptions about melgan in Chinese
    ├── ascend310_infer               // application for 310 inference
    ├── scripts
    │   ├──run_standalone_train_ascend           // shell script for standalone training(1p)
    │   ├──run_distribute_train_ascend.sh        // shell script for distributed training(8p)
    │   ├──run_eval_ascend.sh                    // shell script for evaluation
    │   ├──run_infer_310.sh                      // shell script for 310 evaluation
    ├── src
    │   ├──dataset.py              // creating dataset
    │   ├──model.py                // generator and discriminator architecture
    │   ├──loss.py                 // loss function
    │   ├──config.py               // parameter configuration
    │   ├── model_utils
    │       ├──config.py                      // parameter configuration
    │       ├──device_adapter.py              // device adapter
    │       ├──local_adapter.py               // local adapter
    │       ├──moxing_adapter.py              // moxing adapter
    ├── train.py                   // training script
    ├── eval.py                    //  evaluation script
    ├── config.yaml                // parameter configuration
     ├── export.py                 // export checkpoint files into air/mindir
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.yaml.

- config for MelGAN, LJ Speech dataset

  ```python
  'pre_trained': 'Flase'    # whether training based on the pre-trained model
  'checkpoint_path':  './melgan_20-215_176000.ckpt'
                            # the path of the pre-trained model
  'lr_g': 0.0001            # initial learning rate for Generator
  'lr_d': 0.0001            # initial learning rate for Discriminator
  'batch_size': 4           # training batch size (change to 16 when standalone training)
  'epoch_size': 5000        # total training epochs
  'momentum': 0.9           # momentum
  'leaky_alpha': 0.2        # the alpha in leaky relu
  'train_length': 64        # frames used for each training batch (max:120)

  'beta1':0.9               # The exponential decay rate for the 1st moment estimations
  'beta2':0.999             # The exponential decay rate for the 2nd moment estimations
  'weight_decay':0.0        # Weight decay (L2 penalty)

  'hop_size': 256           # the length of a frame in mel-spectrogram
  'mel_num': 80             # number of mel-spectrogram channels
  'filter_length': 1024     # n-point Short-Time Fourier Transform
  'win_length': 1024        # the length of the window function
  'segment_length': 16000   # the minimum wav length when calculating mel-spectrogram
  'sample': 22050           # the sampling rate of the wav
  'data_path':'/home/datadisk0/voice/melgan/data/'
                            # absolute full path to the train and evaluation datasets
  'save_steps': 4000        # save checkpoint steps.
  'save_checkpoint_name': 'melgan' # name of saved model.
  'save_checkpoint_path': './saved_model'
                            # the absolute full path to save the checkpoint file
  'eval_data_path': '/home/datadisk0/voice/melgan/val_data/'
                            # the path of test mel data
  'eval_model_path': './melgan_20-215_176000.ckpt'
                            # the path of evaluation model
  'output_path': 'output/'  # the storage location of results
  'eval_length': 240        # the number of frames input to the eval_model each time (max:240)
  ```

## [Training Process](#contents)

### Training

  ```python
  python train.py > train.log 2>&1 &
  OR bash scripts/run_standalone_train_ascend.sh DEVICE_ID
  ```

  The python command above will run in the background, you can view the results through the file `train.log`. After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```python
  # grep "loss_G= " train.log
  1epoch 1iter loss_G=27.5 loss_D=27.5 0.30s/it
  1epoch 2iter loss_G=27.4 loss_D=27.4 0.30s/it
  ...
  ```

   The model checkpoint will be saved in the current directory.

### Distributed Training

  ```python
  bash scripts/run_distribute_train_ascend.sh
  ```

  The above shell script will run distribute training in the background. You can view the results through the file `train_parallel[X]/log`. The loss value will be achieved as follows:

  ```python
  # grep "result: " train_parallel*/log
  train_parallel0/log:1epoch 1iter loss_G=27.5 loss_D=27.5 0.30s/it
  train_parallel0/log:1epoch 2iter loss_G=27.4 loss_D=27.4 0.30s/it
  ...
  train_parallel1/log:1epoch 1iter loss_G=27.5 loss_D=27.5 0.30s/it
  train_parallel1/log:1epoch 2iter loss_G=27.4 loss_D=27.4 0.30s/it
  ...
  ...
  ```

## [Evaluation Process](#contents)

### Evaluation

- evaluation on LJ Speech dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "/username/melgan/saved_model/melgan_20-215_176000.ckpt".

  ```python
  bash run_eval_ascend.sh DEVICE_ID PATH_CHECKPOINT
  ```

  The above python command will run in the background. You can view the generated waveforms through the file "output"

## [Export Process](#contents)

### [Export](#content)

```shell
python export.py  --format [EXPORT_FORMAT] --checkpoint_path [CKPT_PATH]
```

## [Inference Process](#contents)

### [Inference](#content)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by export.py. Currently, only batchsize 1 is supported.

```bash
bash run_infer_cpp.sh [MODEL_PATH] [DATA_PATH] [DEVICE_TYPE] [DEVICE_ID]
```

`DEVICE_ID` is optional, default value is 0.
`DEVICE_TYPE` can choose from [Ascend, GPU, CPU]

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | MelGAN                                                       |
| Resource                   | Ascend 910；CPU 2.60GHz，56cores；Memory 755G; OS Euler2.8   |
| uploaded Date              | 10/11/2021                                                   |
| MindSpore Version          | 1.3.0                                                        |
| Dataset                    | LJ Speech                                                    |
| Training Parameters        | epoch=3000, steps=2400000, batch_size=16, lr=0.0001          |
| Optimizer                  | Adam                                                         |
| Loss Function              | L1 Loss                                                      |
| outputs                    | waveforms                                                    |
| Speed                      | 1pc: 320 ms/step; 8pc: 310 ms/step                           |
| Total time                 | 1pc: 220 hours; 8pc: 25 hours                                |
| Loss                       | loss_G=340.123449 loss_D=4.457899                            |
| Parameters (M)             | generator : 4.26; discriminator : 56.4                       |
| Checkpoint for Fine tuning | 361.490M (.ckpt file)                                        |

### Inference Performance

| Parameters          | Ascend                      | Ascend                      |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | MelGAN                      |                             |
| Resource            | Ascend 910                  | Ascend 310                  |
| Uploaded Date       | 10/11/2021                  | 10/11/2021                  |
| MindSpore Version   | 1.5.0                       | 1.5.0                       |
| Dataset             | LJ Speech                   | LJ Speech                   |
| batch_size          | 1                           | 1                           |
| outputs             | waveforms                   | waveforms                   |
| Accuracy            | 3.2(mos score)              | 3.2(mos score)              |
| Model for inference | 361.490M (.ckpt file)       | 18.550M                     |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models/tree/master).
