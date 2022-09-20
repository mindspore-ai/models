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
- [Inference Process](#inference-process)
    - [Export MindIR](#export-mindir)
    - [Export ONNX](#export-onnx)
    - [Infer on Ascend310](#infer-on-ascend310)
        - [Result](#result)
    - [ONNX Infer](#onnx-infer)
        - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)  
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [ESRGAN Description](#contents)

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN – network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

[Paper](https://arxiv.org/pdf/1809.00219.pdf): Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang.

# [Model Architecture](#contents)

The ESRGAN contains a generation network and a discriminator network.

# [Dataset](#contents)

Train ESRGAN Dataset used: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- Note: Before training, please modify the dataset path, align DIV2K dataset path in src/util/extract_subimages.py, and run the script:

```shell
    python src/util/extract_subimages.py
```

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
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
ESRGAN

├─ README.md                             # descriptions about ESRGAN
├── scripts
 ├─ run_infer_310.sh                     # launch ascend 310 inference
 ├─ run_distribute_train_gpu.sh              # launch GPU training(8 pcs)
 ├─ run_eval_gpu.sh                          # launch GPU eval
 ├─ run_eval_onnx_gpu                        # launch ONNX inference
 ├─ run_stranalone_train_gpu.sh              # launch GPU training(1 pcs)
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
├─ export.py                             # export mindir script
├─ export_onnx.py                        # export onnx script
├─ eval.py                               # eval script
├─ eval_onnx.py                          # eval onnx script
├─ preprocess.py                         # preprocess script
├─ postprocess.py                        # postprocess scripts
└─ train.py                              # train script
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training
Ascend:

Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]

# The meaning of the parameters:  DEVICE_NUM(Number of machines) DISTRIBUTE(Whether to use multiple machines) RANK_TABLE_FILE(Machine configuration file) LRPATH(LR training data set picture location) GTPATH(HR training data set picture location) VGGCKPT(VGG19 pre-training parameter position) VPSNRLRPATH(Set5 test set LR picture position) VPSNRGTPATH(Set5 test set HR picture location) VGANLRPATH(Set14 test set LR picture position) VGANGTPATH(Set14 test set HR picture location)

eg: bash run_distribute_train.sh 8 1 ./hccl_8p.json /data/DIV2K/DIV2K_train_LR_bicubic/X4_sub /data/DIV2K/DIV2K_train_HR_sub /home/HEU_535/A8/used/GAN_MD/VGG.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 /data/DIV2K/Set14/LRbicx4 /data/DIV2K/Set14/GTmod12

GPU:

Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]

# The meaning of the parameters:  DEVICE_NUM(Number of machines) LRPATH(LR training data set picture location) GTPATH(HR training data set picture location) VGGCKPT(VGG19 pre-training parameter position) VPSNRLRPATH(Set5 test set LR picture position) VPSNRGTPATH(Set5 test set HR picture location) VGANLRPATH(Set14 test set LR picture position) VGANGTPATH(Set14 test set HR picture location)

eg: bash run_distribute_train_gpu.sh 8  /data/DIV2K/DIV2K_train_LR_bicubic/X4_sub /data/DIV2K/DIV2K_train_HR_sub /home/HEU_535/A8/used/GAN_MD/VGG.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 /data/DIV2K/Set14/LRbicx4 /data/DIV2K/Set14/GTmod12

# standalone training
Ascend:

Usage: bash run_standalone_train.sh  [DEVICE_ID] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]

# The meaning of the parameters DEVICE_ID(Machine ID) LRPATH(LR training data set picture location) GTPATH(HR training data set picture location) VGGCKPT(VGG19 pre-training parameter position) VPSNRLRPATH(Set5 test set LR picture position) VPSNRGTPATH(Set5 test set HR picture position) VGANLRPATH(Set14 test set LR picture position) VGANGTPATH(Set14 test set HR picture position)

eg: bash run_standalone_train.sh 0 /data/DIV2K/DIV2K_train_LR_bicubic/X4_sub /data/DIV2K/DIV2K_train_HR_sub /home/HEU_535/A8/used/GAN_MD/VGG.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 /data/DIV2K/Set14/LRbicx4 /data/DIV2K/Set14/GTmod12

GPU:

Usage: bash run_standalone_train_gpu.sh  [DEVICE_ID] [LRPATH] [GTPATH] [VGGCKPT] [VPSNRLRPATH] [VPSNRGTPATH] [VGANLRPATH] [VGANGTPATH]

# The meaning of the parameters DEVICE_ID(Machine ID) LRPATH(LR training data set picture location) GTPATH(HR training data set picture location) VGGCKPT(VGG19 pre-training parameter position) VPSNRLRPATH(Set5 test set LR picture position) VPSNRGTPATH(Set5 test set HR picture position) VGANLRPATH(Set14 test set LR picture position) VGANGTPATH(Set14 test set HR picture position)

eg: bash run_standalone_train_gpu.sh 0  /data/DIV2K/DIV2K_train_LR_bicubic/X4_sub /data/DIV2K/DIV2K_train_HR_sub /home/HEU_535/A8/used/GAN_MD/VGG.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 /data/DIV2K/Set14/LRbicx4 /data/DIV2K/Set14/GTmod12

```

### [Training Result](#content)

Training result will be stored in ckpt/train_parallel0/ckpt. You can find checkpoint file.

### [Evaluation Script Parameters](#content)

- Run `run_eval.sh` or `run_eval_gpu.sh` for evaluation.

```bash
# evaling
Ascend:

Usage: bash run_eval.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]

eg: bash run_eval.sh /ckpt/psnr_best.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 0

GPU:

Usage: bash run_eval_gpu.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]

eg: bash run_eval_gpu.sh /ckpt/psnr_best.ckpt /data/DIV2K/Set5/LRbicx4 /data/DIV2K/Set5/GTmod12 0
```

### [Evaluation result](#content)

Evaluation result will be stored in the ./result. Under this, you can find generator pictures.

# [Inference Process](#contents)

## [Export MindIR](#contents)

```shell
python export.py --file_name [FILE_NAME] --file_format [FILE_FORMAT] --generator_path[CKPT_PATH]

eg: python export.py ESRGAN MINDIR ./ckpt/psnr_best.ckpt
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## [Export ONNX](#contents)

The model will generate different ONNX file depending on the input size.
Need to use validation set Set5, Set14 for model export and inference.

```shell
python export.py --file_name [FILE_NAME] --file_format [FILE_FORMAT]--test_LR_path [EVALLRPATH] --test_GT_path [EVALGTPATH] --generator_path [CKPT] --device_id [DEVICE_ID]

eg: python export.py --file_name ESRGAN --file_format ONNX --test_LR_path /data/DIV2K/Set5/LRbicx4 --test_GT_path /data/DIV2K/Set5/GTmod12 --device_id 0
eg:python export.py --file_name ESRGAN --file_format ONNX --test_LR_path /data/DIV2K/Set14/LRbicx4 --test_GT_path /data/DIV2K/Set14/GTmod12 --device_id 0
```

The ckpt_file parameter is required.
`EXPORT_FORMAT` should be "ONNX".

## [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [TEST_DATASET_PATH] [NEED_PREPROCESS] [DEVICE_TARGET] [DEVICE_ID]

eg: bash run_infer_310.sh /home/stu/ESRGAN_model.mindir /home/stu/Set5 y Ascend 0  
```

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'avg psnr': 31.83
```

## [ONNX Infer](#contents)

Before performing inference, the onnx file must be exported by `export_onnx.py` script.

```shell
bash run_eval_onnx_gpu.sh [TEST_LR_PATH] [TEST_GT_PATH] [ONNX_PATH]

eg: bash run_eval_onnx_gpu.sh /data/Set5/LRbicx4 /data/Set5/GTmod12 /home/stu/ESRGAN/
```

### [Result](#contents)

Inference result is saved in `eval_onnx`, you can find result like this in `log_onnx` file.

```text
=======starting test=====
avg PSNR: 28.85
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                                                                         | NVIDIA GeForce RTX 3090               |
| -------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------- |
| Model Version              | V1                                                                                 | V1                                    |
| MindSpore Version          | 1.3.0                                                                              | 1.6.0                                 |
| Dataset                    | DIV2K                                                                              | DIV2K                                 |
| Training Parameters        | step=1000000+400000,  batch_size = 16                                              | step=1000000+400000,  batch_size = 16 |
| Optimizer                  | Adam                                                                               | Adam                                  |
| Loss Function              | BCEWithLogitsLoss  L1Loss VGGLoss                                                  | BCEWithLogitsLoss  L1Loss VGGLoss     |
| outputs                    | super-resolution pictures                                                          | super-resolution pictures             |
| Accuracy                   | Set5 psnr 32.56, Set14 psnr 26.23                                                  | Set5 psnr 30.37, Set14 psnr 26.51     |
| Speed                      | 1pc(Ascend): 212,216 ms/step; 8pcs: 77,118 ms/step                                 | 8pcs:239ms/step + 409ms/step          |
| Total time                 | 8pcs: 36h                                                                          |                                       |
| Checkpoint for Fine tuning | 64.86M (.ckpt file)                                                                | 64.86M (.ckpt file)                   |
| Scripts                    | [esrgan script](https://gitee.com/mindspore/models/tree/master/research/cv/ESRGAN) |

### Evaluation Performance

| Parameters        | Ascend 910                |
| ----------------- | ------------------------- |
| Model Version     | V1                        |
| MindSpore Version | 1.3.0                     |
| Dataset           | Set14                     |
| batch_size        | 1                         |
| outputs           | super-resolution pictures |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
