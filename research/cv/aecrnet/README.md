# Contents

- [AECRNet Description](#aecrnet-description)
    - [Abstract](#abstract)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
    - [Dependencies](#dependencies)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#train)
    - [Evaluation](#evaluation)
- [Model Description](#model-description)
    - [Training Performance](#training-performance)
    - [Evaluation Performance](#evaluation-performance)
- [ModelZoo Homepage](#modelzoo-homepage)

# [AECRNet Description](#contents)

This is the MindSpore version of [Contrastive Learning for Compact Single Image Dehazing, CVPR2021](https://arxiv.org/abs/2104.09367)

The official PyTorch implementation, pretrained models and examples are available at [https://github.com/GlassyWu/AECR-Net](https://github.com/GlassyWu/AECR-Net)

The DCN_v2 module is based on <https://gitee.com/mindspore/models/tree/master/research/cv/centernet>. We thank the authors for sharing the codes.

## Abstract

Single image dehazing is a challenging ill-posed problem due to the severe information degeneration. However, existing deep learning based dehazing methods only adopt clear images as positive samples to guide the training of dehazing network while negative information is unexploited. Moreover, most of them focus on strengthening the dehazing network with an increase of depth and width, leading to a significant requirement of computation and memory. In this paper, we propose a novel contrastive regularization (CR) built upon contrastive learning to exploit both the information of hazy images and clear images as negative and positive samples, respectively. CR ensures that the restored image is pulled to closer to the clear image and pushed to far away from the hazy image in the representation space. Furthermore, considering trade-off between performance and memory storage, we develop a compact dehazing network based on autoencoder-like (AE) framework. It involves an adaptive mixup operation and a dynamic feature enhancement module, which can benefit from preserving information flow adaptively and expanding the receptive field to improve the network’s transformation capability, respectively. We term our dehazing network with autoencoder and contrastive regularization as AECR-Net. The extensive experiments on synthetic and real-world datasets demonstrate that our AECR-Net surpass the state-of-the-art approaches.

![image-20210413200215378](https://gitee.com/wyboo/AECRNet-MindSpore/raw/main/images/model.png)

# [Dataset](#contents)

Training set: Indoor Training Set (ITS) from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-standard), 13990 hazy images with its corresponding clear images.
Please download the Indoor Training Set (ITS) and place it in `./dataset/RESIDE/train` folder. (like `./dataset/RESIDE/train/hazy` and `./dataset/RESIDE/train/clear`)

Test set: [Dense-Haze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/), which is the NTIRE2019 challenge dataset consists of dense and homogeneous hazy scenes. As the ground-truth test images are not publicly available, we use validation images (46-50) as test set.
Please download the dataset and place the corresponding test images (46-50) in `./dataset/Dense` folder. (like `./dataset/Dense/GT` and `./dataset/Dense/hazy`)

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - MindSpore
- For more information, please check the resources below：
    - [MindSpore tutorials](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fen%2Fmaster%2Findex.html)
    - [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fapi%2Fen%2Fmaster%2Findex.html)

## Dependencies

- Python == 3.7.5
- MindSpore: <https://www.mindspore.cn/install>
- numpy

# [Script Description](#contents)

## Script and Sample Code

```markdown
.AECRNet
├─ README.md                            # descriptions about AECRNet
├─ images
  └─ model.png                          # AECRNet model image
├─ script
  ├─ run_eval_ascend.sh                 # launch evaluating with ascend platform
  ├─ run_eval_gpu.sh                    # launch evaluating with gpu platform
  ├─ run_distribute_train_ascend.sh     # launch training in parallel mode with ascend platform
  ├─ run_standalone_train_ascend.sh     # launch training with ascend platform
  └─ run_standalone_train_gpu.sh        # launch training with gpu platform
├─ src
  ├─ models
    ├─ __init__.py                      # init file
    ├─ DCN.py                           # Deformable convolution
    └─ model.py                         # AECRNet model define
  ├─ utils
    ├─ __init__.py                      # init file
    └─ var_init.py                      # init functions for VGG
  ├─ __init__.py                        # init file
  ├─ args.py                            # parse args
  ├─ config.py                          # configurations for VGG
  ├─ contras_loss.py                    # contrastive loss define
  ├─ metric.py                          # evaluation utils
  └─ vgg_model.py                       # VGG model define
├─ eval.py                              # evaluation script
├─ haze_data.py                         # dataset processing
├─ export.py                            # export mindir script
└─ train_wcl.py                         # train script
```

## Script Parameters

Major parameters in scripts as follows:

```python
"rgb_range": 255        # range of image pixels, default is 255.
"patch_size": 240       # patch size for training, default is 240.
"batch_size": 16        # batch size for training, default is 16.
"epochs": 1000          # epochs for training, default is 1000.
"dir_data":             # path of train set.
"data_test": "Dense"    # name of test set, only support Dense and NHHaze.
"filename":             # model name.
"ckpt_path":            # path of the checkpoint to load.
"ckpt_save_path":       # models are saved here.
"modelArts_mode": False # whether use ModelArts.
"data_url":             # data path on OBS.
"train_url":            # output path on OBS.

"lr": 1e-5              # learning rate, default is 1e-5.
"loss_scale": 1024.0    # loss scale for LossScaleManager, default is 1024.0

"contra_lambda": 0.1    # weight for contrastive loss, default is 0.1.
"neg_num": 10           # number of negative samples, default is 10.
```

## Training Process

### Train on ModelArts

[ModelArts](https://support.huaweicloud.com/modelarts/index.html) is a one-stop AI development platform that enables developers and data scientists of any skill level to rapidly build, train, and deploy models anywhere, from the cloud to the edge.  Feel free to sign up and get hands on!

1. Create OBS bucket and prepare dataset.
2. VGG pre-trained on ImageNet is used in our contrastive loss. Please download the pre-trained model from [https://download.mindspore.cn/model_zoo/r1.3/](https://download.mindspore.cn/model_zoo/r1.3/) and place it in `./` .
3. We use PyCharm toolkit to help with the training process. You could find tutorial [here](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0021.html). Or you could start training following this [tutorial](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0080.html).

### Train

```bash
bash run_standalone_train_gpu.sh [TRAIN_DATA_PATH] [FILENAME]
bash run_standalone_train_ascend.sh [TRAIN_DATA_PATH] [FILENAME]
# distribute training
bash run_distribute_train_ascend.sh [TRAIN_DATA_DIR] [FILE_NAME] [RANK_TABLE_FILE]
```

For example:

```bash
bash run_standalone_train_ascend.sh ../dataset aecrnet_id_1
```

### Evaluation

```bash
bash run_eval_ascend.sh [DATA_PATH] [CKPT]
bash run_eval_gpu.sh [DATA_PATH] [CKPT]
```

For example:

```bash
bash run_eval_ascend.sh ./dataset ckpt/aecrnet_id_1.ckpt
```

## [Model Description](#contents)

### Training Performance

We use Depth Resnet Generator on Ascend and Resnet Generator on GPU.

| Parameters                 | single GPU                                         |
| -------------------------- | -------------------------------------------------- |
| Model Version              | AECRNet                                            |
| Resource                   | GPU/TITAN RTX                                      |
| MindSpore Version          | 1.5.0-rc1                                          |
| Dataset                    | RESIDE                                             |
| Training Parameters        | epoch=300, batch_size=16, lr=0.0001, neg_num=4     |
| Optimizer                  | Adam                                               |
| Loss Function              | L1 Loss + 20 * Contrastive Loss                    |
| outputs                    | dehazed images                                     |
| Speed                      | 1pc(GPU): 1,194 ms/step                            |
| Total time                 | 1pc(GPU): 86.97h;                                  |
| Checkpoint for Fine tuning | 9.97M (.ckpt file)                                 |

### Evaluation Performance

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Model Version       | AECRNet                     |
| Resource            | GPU/TITAN RTX               |
| MindSpore Version   | 1.5.0-rc1                   |
| Dataset             | Dense-Haze                  |
| batch_size          | 1                           |
| outputs             | dehazed images              |
| SSIM                | 0.4898                      |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
