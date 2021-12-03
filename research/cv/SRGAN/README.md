# Contents

- [SRGAN Description](#SRGAN-description)
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
    - [Infer on Ascend310](#infer-on-ascend310)
    - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)  
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [SRGAN Description](#contents)

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function.Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN,a generative adversarial network (GAN) for image superresolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4× upscaling factors. To achieve this, we propose a perceptualloss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks.

[Paper](https://arxiv.org/pdf/1609.04802.pdf): Christian Ledig, Lucas thesis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi
Twitter.

# [Model Architecture](#contents)

The SRGAN contains a generation network and a discriminator network.

# [Dataset](#contents)

Train SRGAN Dataset used: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- Note: Data will be processed in src/dataset/traindataset.py

Validation and eval evaluationdataset used: [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)/[Set14](https://sites.google.com/site/romanzeyde/research-interests)

- Note:Data will be processed in src/dataset/testdataset.py

# [Pretrained model](#contents)

The process of training SRGAN needs a pretrained VGG19 based on Imagenet.

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
SRGAN

├─ README.md                   # descriptions about SRGAN
├── scripts  
 ├─ run_distribute_train.sh                # launch ascend training(8 pcs Ascend)
 ├─ run_eval.sh                   # launch ascend eval (Ascend)
 ├─ run_stranalone_train.sh             # launch ascend training(1 pcs Ascend)
 ├─ run_distribute_train_gpu.sh                # launch ascend training(8 pcs GPU)
 ├─ run_eval_gpu.sh                   # launch ascend eval(GPU)
 └─ run_stranalone_train_gpu.sh             # launch ascend training(1 pcs GPU)
├─ src  
 ├─ ckpt                       # save ckpt  
 ├─ dataset
  ├─ testdataset.py                    # dataset for evaling  
  └─ traindataset.py                   # dataset for training
├─ loss
 ├─  gan_loss.py                      #srgan losses function define
 ├─  Meanshift.py                     #operation for ganloss
 └─  gan_loss.py                      #srresnet losses function define
├─ models
 ├─ dicriminator.py                  # discriminator define  
 ├─ generator.py                     # generator define  
 └─ ops.py                           # part of network  
├─ result                              #result
├─ trainonestep
  ├─ train_gan.py                     #training process for srgan
  ├─ train_psnr.py                    #training process for srresnet
└─ util
 └─ util.py                         # initialization for srgan
├─ test.py                           # generate images
└─train.py                            # train script
```

## [Script Parameters](#contents)

### [Training Script Parameters](#contents)

```shell
# distributed training

Ascend:

Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]

eg: bash run_distribute_train.sh 8 1 ./hccl_8p.json ./DIV2K_train_LR_bicubic/X4 ./DIV2K_train_HR ./vgg.ckpt ./Set5/LR ./Set5/HR

GPU:

Usage: bash run_distribute_train_gpu.sh [DEVICE_NUM] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]

eg: bash run_distribute_train_gpu.sh 8  ./DIV2K_train_LR_bicubic/X4 ./DIV2K_train_HR ./vgg.ckpt ./Set5/LR ./Set5/HR

# standalone training

Ascend:

Usage: bash run_standalone_train_gpu.sh [DEVICE_ID] [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]

eg: bash run_standalone_train_gpu.sh 0 ./DIV2K_train_LR_bicubic/X4 ./DIV2K_train_HR ./vgg.ckpt ./Set5/LR ./Set5/HR

GPU:

Usage: Usage: bash run_standalone_train_gpu.sh  [LRPATH] [GTPATH] [VGGCKPT] [VLRPATH] [VGTPATH]

eg: bash run_standalone_train_gpu.sh   ./DIV2K_train_LR_bicubic/X4 ./DIV2K_train_HR ./vgg.ckpt ./Set5/LR ./Set5/HR
```

### [Training Result](#content)

Training result will be stored in scripts/train_parallel0/ckpt. You can find checkpoint file.

### [Evaluation Script Parameters](#content)

- Run `run_eval.sh` for evaluation.

```bash
# evaling

Ascend:

Usage: bash run_eval.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]

eg: bash run_eval.sh ./ckpt/best.ckpt ./Set14/LR ./Set14/HR 0

GPU:

Usage: bash run_eval_gpu.sh [CKPT] [EVALLRPATH] [EVALGTPATH] [DEVICE_ID]

eg: bash run_eval_gpu.sh ./ckpt/best.ckpt ./Set14/LR ./Set14/HR 0
```

### [Evaluation result](#content)

Evaluation result will be stored in the scripts/result. Under this, you can find generator pictures.

# [Inference Process](#contents)

## [Export MindIR](#contents)

```shell
python export.py --config_path [CONFIG_PATH] --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

## [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [TEST_LR_PATH] [TEST_GT_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
'avg psnr': 27.4
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | Ascend 910                       | NVIDIA GeForce RTX 3090                     |
| -------------------------- | ---------------------------------|---------------------------------------------|
| Model Version              | V1                               | V1                                          |
| MindSpore Version          | 1.2.0                            | 1.6.0                                       |
| Dataset                    | DIV2K                            | DIV2K                                       |
| Training Parameters        | epoch=2000+1000,  batch_size = 16| epoch=2000+1000,  batch_size = 16           |
| Optimizer                  | Adam                             | Adam                                        |
| Loss Function              | BCELoss  MSELoss VGGLoss         | BCELoss  MSELoss VGGLoss                    |
| outputs                    | super-resolution pictures        | super-resolution pictures                   |
| Accuracy                   | Set14 psnr 27.03                 | Set14 psnr 27.57                            |
| Speed                      | 1pc:540 ms/step;8pcs:1500 ms/step| 1pc: 260+260ms/step; 8pcs: 460+520ms/step   |
| Checkpoint for Fine tuning | 184M (.ckpt file)                | 193M (.ckpt file)                           |
| Scripts                    | [srgan script](https://gitee.com/mindspore/models/tree/master/research/cv/SRGAN)|[srgan script](https://gitee.com/mindspore/models/tree/master/research/cv/SRGAN)|

### Evaluation Performance

| Parameters          | Ascend 910               | NVIDIA GeForce RTX 3090         |
| ------------------- | -------------------------|---------------------------------|
| Model Version       | V1                       | V1                              |
| MindSpore Version   | 1.2.0                    | 1.6.0                           |
| Dataset             | Set14                    | Set14                           |
| batch_size          | 1                        | 1                               |
| outputs             | super-resolution pictures| super-resolution pictures       |

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
