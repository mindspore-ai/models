# Contents

- [DBPN Description](#DBPN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [DBPN Description](#contents)

The feed-forward architectures of recently proposed deep super-resolution networks learn representations of
low-resolution inputs, and the non-linear mapping from those to high-resolution output. However, this approach does not
fully address the mutual dependencies of low- and high-resolution images. We propose Deep Back-Projection Networks (
DBPN), that exploit iterative up- and down-sampling layers, providing an error feedback mechanism for projection errors
at each stage. We construct mutually-connected up- and down-sampling stages each of which represents different types of
image degradation and high-resolution components. We show that extending this idea to allow concatenation of features
across up- and down-sampling stages
(Dense DBPN) allows us to reconstruct further improve super-resolution, yielding superior results and in particular
establishing new state of the art results for large scaling factors such as 8× across multiple data sets.

[Paper](https://arxiv.org/pdf/1803.02735): Muhammad Haris, Greg Shakhnarovich, Norimichi Ukita

# [Model Architecture](#contents)

The DBPN contains a generation network and a discriminator network.

# [Dataset](#contents)

Train DBPN Dataset used: [DIV2K](<https://data.vision.ee.ethz.ch/cvl/DIV2K/>)

- Note: Data will be processed in src/dataset/dataset.py

Validation and eval evaluationdataset
used: [Set5](<http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html>)

datasets of Set5, you can download from: [Set5](https://gitee.com/yzzheng/dbpn-310/tree/master/Set5)

- Note:Data will be processed in src/dataset/dataset.py

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:

```markdown
.DBPN
├─ README.md                       # descriptions about DBPN
├── scripts
  ├─ run_infer_310.sh              # launch ascend 310 inference
  └─ run_distribute_train.sh       # launch ascend training(8 pcs)
  └─ run_stranalone_train.sh       # launch ascend training(1 pcs)
  └─ run_eval.sh                   # launch ascend eval
├─ src
  ├─ dataset
    └─ dataset.py                  # dataset for training and evaling
  ├─ loss
    ├─ withlosscell.py             # DBPN Gan loss Cell define
    └─ generatorloss.py            # compute_psnr losses function define
  ├─ model
    ├─ base_network.py             # the basic unit for build neural network
    ├─ dbpns.py                    # generator DBPNS define  T = 2
    ├─ ddbpnl.py                   # generator DDBPNL define  T = 6
    ├─ ddbpn.py                    # generator DDBPN define  T = 7
    ├─ dbpn.py                     # generator DBPN define T = 10
    ├─ dbpn_iterative.py           # generator DBPNiterative define
    ├─ dicriminator.py             # discriminator define  
    └─ generator.py                # get DBPN models
  ├─ trainonestep
    ├─ trainonestepgen.py          # training process for generator such as DBPNS、DDBPNL、DBPN
    └─ trainonestepgenv2.py        # training process for generator DDBPN which add clip grad
  ├─ util
    ├─ config.py                   # parameters using for train and eval
    └─ utils.py                    # initialization for srgan
  └─ vgg19
    └─ define.py                   # this is just 4 maxpool2d operator
├─ train_dbpn.py                   # train dbpn  script
├─ train_dbpngan.py                # train gan network script
├─ eval.py                         # eval
├─ preprocess.py                   # preprocess script
├─ postprocess.py                  # postprocess scripts
└─ export.py                       # export mindir script
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training DDBPN models
Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [MODEL_TYPE] [TRAIN_GT_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [BATCHSIZE] [MODE]

eg: bash run_distribute_train.sh 8 1 ./hccl_8p.json DDBPN /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR 4 False

# distributed training DBPNGAN models
Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [MODEL_TYPE] [TRAIN_GT_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [BATCHSIZE] [MODE]

eg: bash run_distribute_train.sh 8 1 ./hccl_8p.json DBPN /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR 4 True

# standalone training DDBPN models
Usage: bash run_standalone_train.sh [DEVICE_ID] [MODEL_TYPE] [TRAIN_GT_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [BATCHSIZE] [MODE]

eg: bash run_standalone_train.sh 0 DDBPN /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR 16 False

# standalone training DBPN models
Usage: bash run_standalone_train.sh [DEVICE_ID] [MODEL_TYPE] [TRAIN_GT_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [BATCHSIZE] [MODE]

eg: bash run_standalone_train.sh 0 DBPN /data/DBPN_data/DIV2K_train_HR /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR 4 True

```

### [Training Result](#content)

Training result of the model genearated image will be stored in Results.

Training result of the model loss will be stored in result.

You can find checkpoint file in ckpt.

### [Evaluation Script Parameters](#content)

- Run `run_eval.sh` for evaluation.

```bash
#evaling DDBPN
bash run_eval.sh [DEVICE_ID] [CKPT] [MODEL_TYPE] [VAL_GT_PATH] [VAL_LR_PATH]

eg: bash run_eval.sh 0 /data/DBPN_data/dbpn_ckpt/gen_ckpt/D-DBPN-best.ckpt DDBPN /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR

#evaling DBPNGAN network
bash run_eval.sh [DEVICE_ID] [CKPT] [MODEL_TYPE] [VAL_GT_PATH] [VAL_LR_PATH]

eg: bash run_eval.sh 0 /data/DBPN_data/dbpn_ckpt/gan_ckpt/best_G.ckpt DBPN /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR
```

### [Evaluation result](#content)

Evaluation result will be stored in the result. Under this, you can find generator pictures.

## [Inference Process](#contents)

### [Export MindIR](#contents)

```bash
# export DDBPN
python export.py  [CKPT] [FILE_NAME] [FILE_FORMAT]

eg: python export.py --ckpt_path=ckpt/Set5_DDBPN_best.ckpt --file_name=ddbpn --file_format=MINDIR --model_type=DDBPN

# export DBPN GAN networks
python export.py  [CKPT] [FILE_NAME] [FILE_FORMAT]

eg: python export.py --ckpt_path=ckpt/best_G.ckpt --file_name=dbpn --file_format=MINDIR --model_type=DBPN
```

The pretrained parameter is required. EXPORT_FORMAT should be in ["AIR", "MINDIR"] Current batch_size can only be set to 1.

### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py`.Current batch_Size can only be set to 1.

```shell
# Ascend310 inference about DDBPN network
bash run_infer_310.sh [MINDIR_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]

eg: bash run_infer_310.sh ddbpn_model.mindir /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR y Ascend 0

# Ascend310 inference about DBPN network
bash run_infer_310.sh [MINDIR_PATH] [VAL_GT_PATH] [VAL_LR_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]

eg: bash run_infer_310.sh dbpn_model.mindir /data/DBPN_data/Set5/HR /data/DBPN_data/Set5/LR y Ascend 0
```

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'avg psnr': 27.52
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 |                                                             |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | DBPN                                                        |
| Resource                   | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | DIV2K                                                       |
| Training Parameters        | ddbpn:epoch=2000, batch_size = 16; dbpngan:epoch=1100,batch_size=4|
| Optimizer                  | Adam                                                        |
| Loss Function              | BCELoss  MSELoss VGGLoss                                    |
| outputs                    | super-resolution pictures                                   |
| Accuracy                   | Set5 compute_psnr 31.92(train ddbpn); Set5 compute_psnr 29.21(train dbpngan network)                       |
| Speed                      | 1pc(Ascend): 3463 ms/step(ddbpn), 800.41ms/step(dbpngan); 8pcs: 1781ms/step(ddbpn)  818.22ms/step(dbpngan) |
| Total time                 | 8pcs: 12h43m50s(ddbpn), 13h50m28s(dbpngan)                                                    |
| Checkpoint of DBPN models  | 58.8M (.ckpt file)                                           |
| Scripts                    | [DBPN script](https://gitee.com/mindspore/models/tree/master/research/cv/DBPN) |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models/tree/master/)
.