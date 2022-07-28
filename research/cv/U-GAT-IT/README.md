# [Table of contents](#table-of-contents)

- [U-GAT-IT Description](#u-gat-it-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Dataset Features](#dataset-features)
- [Environmental Requirements](#environmental-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
- [Script Parameters](#script-parameters)
- [Training and evaluation](#training-and-evaluation)
- [Inference on Ascend 310](#inference-on-ascend-310)
- [Model Description](#model-description)
    - [Model Performance](#model-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [U-GAT-IT Description](#table-of-contents)

U-GAT-IT proposes a new unsupervised image-to-image translation method. The method introduces a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides the model to focus on the important regions that distinguish the source and target domains, enabling shape changes. Additionally, Adaptive Layer-Instance Normalization helps the model flexibly control changes in shape and texture. The model can perform the conversion of real face selfie image to cartoon face image.

[Paper](https://openreview.net/attachment?id=BJlZ5ySKPH&name=original_pdf)：Kim J, Kim M, Kang H, et al. U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation[C]//International Conference on Learning Representations. 2019.

# [Model Architecture](#table-of-contents)

The model consists of two generators and two discriminators.

# [Dataset](#table-of-contents)

Dataset used：[selfie2anime](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view)

- Dataset size：400M，7,000 256*256 JPG images
    - Train：6800 images (3400 selfie images and 3400 anime images respectively)
    - Test：200 images (100 selfie images and 100 anime images respectively)

```text

└─dataset
  └─selfie2anime
    └─trainA
    └─trainB
    └─testA
    └─testB
```

- After the dataset is decompressed, put the `/dataset` directory in the `../U-GAT-IT/` directory

# [Dataset Features](#table-of-contents)

# [Environmental Requirements](#table-of-contents)

- Hardware (Ascend/GPU/CPU)
    - Use Ascend/GPU/CPU processor to build hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#table-of-contents)

## [Script and Sample Code](#table-of-contents)

```text
├──U-GAT-IT
    ├── README_CN.md                                 # README
    ├── README.md                                    # README in English
    ├── requirements.txt                             # required modules
    ├── scripts                                      # shell script
        ├─run_standalone_train_910.sh                # training in standalone mode
        ├─run_distributed_train_910.sh               # training in parallel mode
        ├─run_eval_910.sh                            # evaluation
        ├─run_infer_310.sh                           # 310 inference
        ├─run_standalone_train_gpu.sh                # training on gpu in standalone mode
        ├─run_distributed_train_gpu.sh               # training on gpu in parallel mode
        └─run_eval_gpu.sh                            # evaluation on gpu
    ├── src
        ├─dataset
            └─dataset.py                             # data preprocess and loader
        ├─modelarts_utils
            ├─config.py                              # config
            ├─device_adapter.py                      # device adapter
            ├─local_adapter.py                       # local adapter
            └─moxing_adapter.py                      # moxing adapter
        ├─models
            ├─cell.py                                # trainers
            ├─networks.py                            # generator and discriminator networks define
            └─UGATIT.py                              # training and testing pipline
        ├─metrics
            ├─create_inception_checkpoint.py         # script that creates inception checkpoint
            ├─inception.py                           # inception model for metrics computation
            └─metrics.py                             # metrics evaluation
        ├─utils
            └─tools.py                               # utils for U-GAT-IT
        └─default_config.yaml                        # configs for training and testing
    ├── ascend310_infer                              # 310 inference
        ├─ src
            ├─ main.cc                               # 310 inference
            └─ utils.cc                              # utils for 310 inference
        ├─ inc
            └─ utils.h                               # head file
        ├─ CMakeLists.txt                            # compile
        ├─ build.sh                                  # script of main.cc
        └─ fusion_switch.cfg                         # Use BatchNorm2d instead of InstanceNorm2d
    ├── train.py                                     # train launch file
    ├── eval.py                                      # eval launch file
    ├── export.py                                    # export checkpoints into mindir model
    ├── preprocess.py                                # preprocess for 310 inference
    └── postprocess.py                               # translate the result of 310 inference into jpg images
```

# [Script Parameters](#table-of-contents)

Both training and evaluation parameters can be configured in `../U-GAT-IT/src/defalt_config.yaml`.

- Main parameters：

```python
    '--distributed', type=str2bool, default=False                                              # Whether it is distributed training
    '--output_path', type=str, default="results"                                               # Intermediate results, model save path
    '--phase', type=str, default='train', help='[train / test]'                                # Training or evaluation phase
    '--dataset', type=str, default='selfie2anime', help='dataset_name'                         # Dataset name
    '--iteration', type=int, default=1000000, help='The number of training iterations'         # The number of training iterations
    '--batch_size', type=int, default=1, help='The size of batch size'                         # Batch size
    '--print_freq', type=int, default=1000, help='The number of image print freq'              # Image print frequency
    '--save_freq', type=int, default=100000, help='The number of model save freq'              # Model save frequency
    '--decay_flag', type=str2bool, default=True, help='The decay_flag'                         # Whether to use learning rate decay

    '--loss_scale', type=float, default=1024.0, help='The loss scale'                          # The magnification of loss to ensure gradient accuracy
    '--lr', type=float, default=0.0001, help='The learning rate'                               # Learning rate
    '--weight_decay', type=float, default=0.0001, help='The weight decay'                      # Weight decay
    '--adv_weight', type=int, default=1, help='Weight for GAN'                                 # The weight of adversarial loss
    '--cycle_weight', type=int, default=10, help='Weight for Cycle'                            # The weight of cycle loss
    '--identity_weight', type=int, default=10, help='Weight for Identity'                      # Weight for identity loss
    '--cam_weight', type=int, default=1000, help='Weight for CAM'                              # Weight for CAM loss
    '--ch', type=int, default=64, help='base channel number per layer'                         # Base channel number per layer
    '--n_res', type=int, default=4, help='The number of resblock'                              # The number of resnet blocks
    '--n_dis', type=int, default=6, help='The number of discriminator layer'                   # The number of discriminator layers
    '--img_size', type=int, default=256, help='The size of image'                              # Image size
    '--img_ch', type=int, default=3, help='The size of image channel'                          # The number of image channels

    '--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU']            # Device target
    '--device_id', type=str, default='7', help='Set target device id to run'                   # Card number
    '--resume', type=str2bool, default=False                                                   # Whether to load the existing model to continue training
    '--save_graphs', type=str2bool, default=False, help='Whether or not to save the graph'     # Whether to save the training graph
    '--graph_path', type=str, default='graph_path', help='Directory name to save the graph'    # Directory name to save the graph

    '--genA2B_ckpt', type=str, default='./results/selfie2anime_genA2B_params_latest.ckpt'      # 310 Inference only, the storage path of the generator model
    '--MINDIR_outdir', type=str, default='./mindir_result'                                     # 310 Inference only, path for exported file
    "--export_file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR"      # 310 Inference only, export file format

    '--bifile_inputdir', type=str, default='./bifile_in'                                       # 310 Inference only, input path of binary image file
    '--bifile_outputdir', type=str, default='./bifile_out'                                     # 310 Inference only, export path for binary image files
    '--eval_outputdir', type=str, default='./infer_output_img'                                 # 310 Inference only, post-processing JPG file export path
```

# [Training and evaluation](#table-of-contents)

Go to the directory `../U-GAT-IT/`, and after installing MindSpore through the official website, you can follow the steps below for training and evaluation (the decompressed dataset needs to be placed in the directory `../U-GAT-IT/`):

- Training on Ascend 910

  ```shell
  # Standalone training
  bash ./scripts/run_standalone_train_910.sh [DEVICE_ID]

  # Distributed training
  bash ./scripts/run_distributed_train_910.sh [RANK_TABLE] [RANK_SIZE] [DEVICE_START]
  ```

  Example:

  ```shell
  # Standalone training
  bash ./scripts/run_standalone_train_910.sh 0

  # Ascend multi-card training (8P)
  bash ./scripts/run_distributed_train_910.sh ./rank_table_8pcs.json 8 0
  ```

- Evaluation on Ascend 910

  ```shell
  # Evaluation
  bash ./scripts/run_eval_910.sh [DEVICE_ID]
  ```

  Example:

  ```shell
  # Evaluation
  bash ./scripts/run_eval_910.sh 0
  ```

- Train on ModelArts

  Before training on ModelArts, configure the parameters in `../U-GAT-IT/src/default_config.yaml`. Please read the official documentation of [ModelArts](https://support.huaweicloud.com/modelarts/), and then follow the instructions below:

  ```text
  # Single card training on ModelArts
  # (1) Execute a or b
  # a. Set "enable_modelarts=True" in the default_config.yaml file
  # Set "data_path='/cache/data'" in default_config.yaml file
  # Set "ckpt_path='/cache/train'" in default_config.yaml file
  # Set other parameters in default_config.yaml file
  # b. Set "enable_modelarts=True" on the web page
  # Set "data_path='/cache/data'" on the web page
  # Set "ckpt_path='/cache/train'" on the web page
  # Set other parameters on the web page
  # (3) If you choose to fine-tune your model, upload your pre-trained model to the S3 bucket
  # (4) Upload the original dataset to the S3 bucket.
  # (5) Set your code path on the web page to "/path/U-GAT-IT"
  # (6) Set the startup file as "train.py" on the web page
  # (7) Set "training data set", "training output file path", "job log path", etc. on the web page
  # (8) Create a training job
  #
  # 8-card training on ModelArts
  # (1) Execute a or b
  # a. Set "enable_modelarts=True" in the default_config.yaml file
  # Set "data_path='/cache/data'" in default_config.yaml file
  # Set "ckpt_path='/cache/train'" in default_config.yaml file
  # Set "distributed=True" in default_config.yaml file
  # Set other parameters in default_config.yaml file
  # b. Set "enable_modelarts=True" on the web page
  # Set "data_path='/cache/data'" on the web page
  # Set "ckpt_path='/cache/train'" on the web page
  # Set "distributed=True" on the page
  # Set other parameters on the web page
  # (3) If you choose to fine-tune your model, upload your pre-trained model to the S3 bucket
  # (4) Upload the original dataset to the S3 bucket.
  # (5) Set your code path on the web page to "/path/U-GAT-IT"
  # (6) Set the startup file as "train.py" on the web page
  # (7) Set "training data set", "training output file path", "job log path", etc. on the webpage
  # (8) Create a training job
  ```

- Training on GPU

  ```shell
  # Standalone training
  bash run_standalone_train_gpu.sh [DEVICE_ID] [DATA_PATH] [<LR>] [<LIGHT>] [<LOSS_SCALE>] [<USE_GLOBAL_NORM>]

  # Multi-GPU training
  bash run_distributed_train_gpu.sh [RANK_SIZE] [DATA_PATH] [<LR>] [<LIGHT>] [<LOSS_SCALE>] [<USE_GLOBAL_NORM>]
  ```

  Example:

  ```shell
  # Standalone training
  bash run_standalone_train_gpu.sh 0 /path/to/data/ 0.0001 True 1.0 False

  # Multi-GPU training
  bash run_distributed_train_gpu.sh 8 /path/to/data 0.000025 True 1.0 False
  ```

- Evaluation on GPU

  ```shell
  bash run_eval_gpu.sh [DEVICE_ID] [DATA_PATH] [OUTPUT_PATH] [LIGHT] [<INCEPTION_CKPT_PATH>]
  ```

  Example:

  ```shell
  # Evaluation without metrics computation
  bash run_eval_gpu.sh 0 /path/to/data/ /output/path True

  # Evaluation with metrics computation using an existing checkpoint
  # To compute metrics, you need to specify inception ckpt path
  bash run_eval_gpu.sh 0 /path/to/data/ /output/path True inception_for_metrics.ckpt
  ```

  To create a checkpoint for metrics computation, install tensorflow==1.13.1 and run:

  ```shell
  python create_inception_checkpoint.py inception_for_metrics.ckpt
  ```

- Training results will have the following form:

  ```text
  ...
  epoch 1:[ 1220/10200] time per iter: 0.6664
  d_loss: 3.2190554
  g_loss: 2914.113
  epoch 1:[ 1221/10200] time per iter: 0.6666
  d_loss: 3.6988006
  g_loss: 115.64786
  epoch 1:[ 1222/10200] time per iter: 0.6654
  d_loss: 3.3407924
  g_loss: 313.15454
  epoch 1:[ 1223/10200] time per iter: 0.6641
  d_loss: 2.4665039
  g_loss: 163.84
  epoch 1:[ 1224/10200] time per iter: 0.6639
  d_loss: 4.269173
  g_loss: 87.08121
  epoch 1:[ 1225/10200] time per iter: 0.6648
  d_loss: 2.621345
  g_loss: 100.11629
  ...
   [*] Training finished!
  ```

- Evaluation results will have the following form:

  ```text
  Dataset load finished
  build_model cost time: 104.3066
  these params are not loaded:  {'genA2B': []}
   [*] Load SUCCESS
  [WARNING] SESSION(6660,fffe647f41e0,python):2021-12-27-16:10:12.128.336 [mindspore/ccsrc/backend/session/ascend_session.cc:1806] SelectKernel] There are 23 node/nodes used reduce precision to selected the kernel!
   [*] Test finished!
  ```

- The result of metrics evaluation will have the following form:

  ```text
  mean_KID_mean :  10.866395338438451
  mean_KID_stddev :  0.42898761875861713
  ```

# [Inference on Ascend 310](#table-of-contents)

After training the generator model, you can export the ckpt file to MINDIR format file through export.py and perform inference on Ascend 310:

- Generate mindir file

  ```bash
  python export.py --genA2B_ckpt ./results/selfie2anime_genA2B_params_latest.ckpt --MINDIR_outdir ./mindir_result
  ```

- Execute Ascend 310 inference script

  ```bash
  bash ./scripts/run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
  ```

  Example of usage:

  ```shell
  bash scripts/run_infer_310.sh ./mindir_result/UGATIT_AtoB_graph.mindir ./dataset/selfie2anime/testA/ 0
  ```

# [Model Description](#table-of-contents)

## [Model Performance](#table-of-contents)

### [Evaluation Performance](#table-of-contents)

#### [U-GAT-IT on selfie2anime dataset](#table-of-contents)

| Parameters          | Ascend                                                                                         | GPU                                                                                                                           |
|---------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Model name          | U-GAT-IT                                                                                       | U-GAT-IT (light)                                                                                                              |
| Resource            | Ascend: 8 * Ascend-910(32GB) <br /> ARM: 192 cores 2048GB <br /> CentOS 4.18.0-193.el8.aarch64 | GPU: 8 * GeForce RTX 3090 <br /> CPU 2.90GHz, 64 cores <br /> RAM:252G                                                        |
| Upload date         | 2021-12-23                                                                                     | 2022-02-28                                                                                                                    |
| MindSpore version   | 1.5.0                                                                                          | 1.5.0                                                                                                                         |
| Datasets            | selfie2anime                                                                                   | selfie2anime                                                                                                                  |
| Training parameters | epoch=100, batch_size = 1, lr=0.0001                                                           | light=True, epoch=100, batch_size=1, lr=0.000025, <br /> loss_scale=1.0, use_global_norm=False                                |
| Optimizer           | Adam                                                                                           | Adam                                                                                                                          |
| Loss function       | Custom loss function                                                                           | Custom loss function                                                                                                          |
| Output              | Image                                                                                          | Image                                                                                                                         |
| Speed               | 640ms/step                                                                                     | 690 ms/step                                                                                                                   |
| Checkpoint          | 1.04GB, ckpt file                                                                              | 41 MB, ckpt file                                                                                                              |
| Metrics             | -                                                                                              | Kernel Inception Distance (KID) <br /> Typical good results for KID metrics are in range (10.5, 12.5), <br /> lower is better |
| KID                 | -                                                                                              | KID mean :  10.8664 <br /> KID stddev :  0.4290                                                                               |

> The checkpoint of UGATIT trained on GPU (selfie2anime_genA2B_params_0000100.ckpt) and the checkpoit for metrics computation (inception.ckpt)
> are available [here](https://disk.yandex.ru/d/_MbEh0uGG9_eyA).

# [Description of Random Situation](#table-of-contents)

 Set the random seed to 1 in main.py

# [ModelZoo Homepage](#table-of-contents)

 Please visit the [official website](https://gitee.com/mindspore/models).
