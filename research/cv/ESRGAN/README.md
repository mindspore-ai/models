# Contents

- [ESRGAN Description](#ESRGAN-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Pretrained model](#pretrained-model)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)  
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ESRGAN Description](#contents)

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN – network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

[Paper](https://arxiv.org/pdf/1809.00219.pdf): Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang.

# [Model Architecture](#contents)

The ESRGAN contains a generation network and a discriminator network.

# [Dataset](#contents)

Train ESRGAN Dataset used: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- Note: Please run the script extract_subimages.py to crop sub-images, then use the sub-images to train the model

- Note: Data will be processed in src/dataset/traindataset.py

Validation and eval evaluationdataset used: [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)/[Set14](https://sites.google.com/site/romanzeyde/research-interests)

- Note:Data will be processed in src/dataset/testdataset.py

# [Pretrained model](#contents)

The process of training ESRGAN needs a pretrained VGG19 based on Imagenet.

[Training scripts](<https://gitee.com/mindspore/models/tree/master/official/cv/vgg16>)|
[VGG19 pretrained model](<https://download.mindspore.cn/model_zoo/>)

# [Environment Requirements](#contents)

- Hardware Ascend
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
ESRGAN

├─ README.md                             # descriptions about ESRGAN
├── scripts  
 ├─ run_distribute_train.sh              # launch ascend training(8 pcs)
 ├─ run_eval.sh                          # launch ascend eval
 └─ run_stranalone_train.sh              # launch ascend training(1 pcs)
├─ src  
 ├─ dataset
  ├─ testdataset.py                      # dataset for evaling  
  └─ traindataset.py                     # dataset for training
 ├─ loss
  ├─  gan_loss.py                        # esrgan losses function define
  └─  psnr_loss.py                       # rrdbnet losses function define
 ├─ models
  ├─ dicriminator.py                     # discriminator define  
  └─ generator.py                        # generator define  
 ├─ trainonestep
  ├─ train_gan.py                        # training process for esrgan
  └─ train_psnr.py                       # training process for rrdbnet
 └─ util
  ├─ extract_subimages.py                # crop large images to sub-images
  └─ util.py                             # Utils for model
├─ export.py                             # generate images
├─ eval.py                               # eval script
└─ train.py                              # train script
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]

# The meaning of the parameters:  DEVICE_NUM(Number of machines) DISTRIBUTE(Whether to use multiple machines) RANK_TABLE_FILE(Machine configuration file) LRPATH(LR training data set picture location) GTPATH(HR training data set picture location) VGGCKPT(VGG19 pre-training parameter position) VPSNRLRPATH(Set5 test set LR picture position) VPSNRGTPATH(Set5 test set HR picture location) VGANLRPATH(Set14 test set LR picture position) VGANGTPATH(Set14 test set HR picture location)

eg: bash run_distribute_train.sh 8 1 ./hccl_8p.json /data/DIV2K/DIV2K_train_LR_bicubic/X4_sub /data/DIV2K/DIV2K_train_HR_sub /home/HEU_535/A8/used/GAN_MD/VGG.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 /data/DIV2K/Set14/LRbicx4 /data/DIV2K/Set14/GTmod12

# standalone training
Usage: bash run_standalone_train.sh  [DEVICE_ID] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]

# The meaning of the parameters DEVICE_ID(Machine ID) LRPATH(LR training data set picture location) GTPATH(HR training data set picture location) VGGCKPT(VGG19 pre-training parameter position) VPSNRLRPATH(Set5 test set LR picture position) VPSNRGTPATH(Set5 test set HR picture position) VGANLRPATH(Set14 test set LR picture position) VGANGTPATH(Set14 test set HR picture position)

eg: bash run_standalone_train.sh 0 /data/DIV2K/DIV2K_train_LR_bicubic/X4_sub /data/DIV2K/DIV2K_train_HR_sub /home/HEU_535/A8/used/GAN_MD/VGG.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 /data/DIV2K/Set14/LRbicx4 /data/DIV2K/Set14/GTmod12
```

### [Training Result](#content)

Training result will be stored in ckpt/train_parallel0/ckpt. You can find checkpoint file.

### [Evaluation Script Parameters](#content)

- Run `run_eval.sh` for evaluation.

```bash
# evaling
Usage: bash run_eval.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]

eg: bash run_eval.sh ./ckpt/psnr_best.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 0
```

### [Evaluation result](#content)

Evaluation result will be stored in the ./result. Under this, you can find generator pictures.

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 |                                                             |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | ESRGAN                                                      |
| Resource                   | CentOs 8.2; Ascend 910; CPU 2.60GHz, 192cores; Memory 755G  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | DIV2K                                                       |
| Training Parameters        | step=1000000+400000,  batch_size = 16                       |
| Optimizer                  | Adam                                                        |
| Loss Function              | BCEWithLogitsLoss  L1Loss VGGLoss                           |
| outputs                    | super-resolution pictures                                   |
| Accuracy                   | Set5 psnr 32.56, Set14 psnr 26.23                           |
| Speed                      | 1pc(Ascend): 212,216 ms/step; 8pcs: 77,118 ms/step          |
| Total time                 | 8pcs: 36h                                                   |
| Checkpoint for Fine tuning | 64.86M (.ckpt file)                                         |
| Scripts                    | [esrgan script](https://gitee.com/mindspore/models/tree/master/research/cv/ESRGAN) |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
