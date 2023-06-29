# Content

# Model name

> Segformer

Segformer is a simple, efficient and powerful semantic segmentation framework proposed by Enze Xie et al. of Hong Kong University. This model unifies transformer and lightweight multi-layer perception (MLP) decoder.
SegFormer has two attractive features: 1) SegFormer includes a novel hierarchical transformer encoder that outputs multi-scale features. It does not need position coding, so it avoids interpolation of position coding, which leads to performance degradation when the test resolution is different from the training resolution. 2) SegFormer avoids complex decoders.

## Thesis

[Link to Paper] (https://arxiv.org/abs/2105.15203)

## Model Architecture

The overall network architecture of Segformer is as follows:
[Link] (https://arxiv.org/abs/2105.15203)

## Dataset

Dataset used: [Cityscapes](https://www.cityscapes-dataset.com/downloads/) needs to be registered and downloaded.

- Dataset size: 19 categories, 5000 1024 x 2048 color images
    - Training set: 2975 images
    - Evaluation set: 500 images
    - Test set: 1525 images
- Data format: RGB
- The directory structure is as follows:

```txt
segformer
├── src
├── scripts
├── config
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```

Note: You need to convert the **polygons.json file to the **labelTrainIds.png file before using the convert_dataset.py script developed based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts).

```shell
python tools/convert_dataset.py data/cityscapes --nproc 8
```

## Features

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data types, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

## Requirements

- Hardware (Ascend)
    - Prepare the Ascend hardware environment.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For details, see the following resources:
    - [MindSpore Tutorial] (https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## Quick Start

> Before training, you need to convert the dataset format and pretrained the model.

- Convert Dataset Format
    - If you are using the Cityspaces dataset, you need to convert the dataset format before training
    - You can use the convert_dataset.py script to convert `**polygons.json` to `**labelTrainIds.png`
    - where data/cityscapes is the data set path and nproc is the number of processes that are executed concurrently.

```shell
python tools/convert_dataset.py data/cityscapes --nproc 8
```

- Convert the pre-trained model.
    - You can use the convert_model.py script to convert the pytorch pretrained model provided by the author to the mindspore pretrained model.
    - Download the pre-trained model provided by the author of the paper([google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing)|[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw))
    - pt_model_path indicates the path of the PyTorch model, and ms_model_path indicates the path for storing the mindspore model.

```shell
python tools/convert_model.py --pt_model_path=./pretrained/mit_b0.pth --ms_model_path=./pretrained/ms_pretrained_b0.ckpt
```

- Prepare the hccl_8p.json files, before run network.
    - Genatating hccl_8p.json, Run the script of [hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py).

```shell  
# The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.
python hccl_tools.py --device_num "[0,8)"
```

- Ascend processor environment operation

```text
# Distributed training
Usage: bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE]

# Standalone training
Usage: bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE]

# Run the assessment example
Usage: bash run_eval.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH]
```

Note that if you run the preceding quick startup command, the dataset and pre-trained model must be stored in the default path. The following is an example:

```txt
segformer
├── src
├── scripts
├── config
├── data                    # Dataset directory
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
├── pretrained              # Directory of the pretrained model
```

## Script Description

### Script and sample code

```log
└──segformer
   ├── config                                    # Configuration file directory
       ├── segformer.b0.512x1024.city.yaml       # mit_b0 configuration
       ├── segformer.b0.1024x1024.city.yaml
       ├── segformer.b1.1024x1024.city.yaml
       ├── segformer.b2.1024x1024.city.yaml
       ├── segformer.b3.1024x1024.city.yaml
       ├── segformer.b4.1024x1024.city.yaml
       ├── segformer.b5.1024x1024.city.yaml
       ├── segformer.base.yaml                   # public parameter configuration file
   ├── scripts                                   # Script directory
       ├── run_distribute_train.sh               # Run the distributed training script.
       ├── run_eval.sh                           # Run the evaluation script.
       ├── run_infer.sh                          # Run the inference script.
       ├── run_standalone_train.sh               # Run the single-card training script.
   ├── src
       ├── model_utils
           ├── config.py                         # Configuration file script
       ├── base_dataset.py                       # Dataset processing script
       ├── dataset.py                            # Script for processing the Cityscapes dataset.
       ├── loss.py                               # Define the loss function.
       ├── mix_transformer.py                    # Define the backbone. b0 to b5 are supported.
       ├── optimizers.py                         # Define optimizers.
       ├── segformer.py                          # Define the entire network.
       ├── segformer_head.py                     # Define the head network.
   ├── tools
       ├── convert_dataset.py                    # Script for converting the format of a dataset.
       ├── convert_model.py                      # Convert the pretrained model script.
   ├── eval.py                                   # Evaluation script
   ├── infer.py                                  # Inference script
   ├── train.py                                  # Training script
   ├── README.md
   ├── requirement.txt                           # Third-party dependency
```

### Script parameters

```shell
run_distribute: False                                       # Indicates whether the training is distributed.
data_path: "./data/cityscapes/"                             # Dataset Path
load_ckpt: True                                             # Indicates whether to load the pretrained model.
pretrained_ckpt_path: "./pretrained/ms_pretrained_b0.ckpt"  # pretrained models.
save_checkpoint: True                                       # Indicates whether to save the model.
save_checkpoint_epoch_interval: 1                           # Interval for saving the model epoch.
save_best_ckpt: True                                        # Indicates whether to save the best model.
checkpoint_path: "./checkpoint/"                            # Path for storing the model
train_log_interval: 100                                     # Step interval for printing training logs.

epoch_size: 200                                             # Size of the training epoch.
batch_size: 2                                               # Batch size of the input tensor.
backbone: "mit_b0"                                          # Backbone network, which supports mit_b0 to mit_b5.
class_num: 19                                               # Number of dataset categories.
dataset_num_parallel_workers: 4                             # parallel dataset processing threads
momentum: 0.9                                               # Momentum optimizer
lr: 0.0001                                                  # Learning Rate
weight_decay: 0.01                                          # Weight decay
base_size: [512, 1024]                                      # Indicates the basic image size. The original image will be resized to this size first.
crop_size: [512, 1024]                                      # Crop size, which is the network input after the image is cropped.
img_norm_mean: [123.675, 116.28, 103.53]                    # image regularization
img_norm_std: [58.395, 57.12, 57.375]                       # image regularization

run_eval: True                                              # Indicates whether to evaluate the model during training.
eval_start_epoch: 0                                         # Number of epochs that start to evaluate the model.
eval_interval: 1                                            # Epoch interval for evaluating the model.
eval_ckpt_path: ""                                          # Path for storing the evaluation model.
eval_log_interval: 100                                      # Number of step intervals for printing evaluation model logs.

infer_copy_original_img: True                               # Indicates whether to copy the original image to the inference result directory during inference.
infer_save_gray_img: True                                   # Indicates whether to save the gray image during inference.
infer_save_color_img: True                                  # Indicates whether to save the color image during inference.
infer_save_overlap_img: True                                # Indicates whether to save the overlap image during inference.
infer_log_interval: 100                                     # Interval for printing inference logs.
infer_ckpt_path: ""                                         # Path for the model used for inference.
infer_output_path: "./infer_result/"                        # Inference Result Storage Path.
```

## Training

Before training, you need to convert the dataset format and the pretrained model.

- Convert Dataset Format
    - If you are using the Cityspaces dataset, you need to convert the dataset format before training
    - You can use the convert_dataset.py script to convert `**polygons.json` to `**labelTrainIds.png`
    - where data/cityscapes is the data set path and nproc is the number of processes that are executed concurrently.

```shell
python tools/convert_dataset.py data/cityscapes --nproc 8
```

- Convert the pre-trained model.
    - You can use the convert_model.py script to convert the pytorch pre-trained model provided by the author to the mindspore pre-trained model.
    - Download the pre-trained model provided by the author of the paper([google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing)|[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw))
    - pt_model_path indicates the path of the PyTorch model, and ms_model_path indicates the path for storing the mindspore model.

```shell
python tools/convert_model.py --pt_model_path=./pretrained/mit_b0.pth --ms_model_path=./pretrained/ms_pretrained_b0.ckpt
```

### Training Process

Run the following command on the single-card training

```shell
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE]
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional)
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional)
bash run_standalone_train.sh [DEVICE_ID] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional) [RUN_EVAL](optional)
For example: bash run_standalone_train.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/data/cityscapes/"
```

You can also run the Python script directly.

```shell
python train.py --config_path=[CONFIG_FILE] --pretrained_ckpt_path=[PRETRAINED_CKPT_PATH](optional) --data_path=[DATASET_PATH](optional) --run_eval=[RUN_EVAL](optional)
```

```log
Parameter description:
DEVICE_ID: ID of the card that performs the training. This parameter is mandatory. If the training is started using Python, you can run the export DEVICE_ID=X command to specify the card number. X indicates the card number.
CONFIG_FILE: indicates the configuration file path. This parameter is mandatory.
DATASET_PATH: dataset path. This parameter is optional. The default value is data/cityscapes.
PREFCCC_CKPT_PATH: (Optional) Path of the pre-trained model. The default value is pretrained/pretrained/ms_pretrained_b0.ckpt.
RUN_EVAL: indicates whether to evaluate the model during training. The default value is True. This parameter is optional.
```

Training process logs are stored in the train.log file.

```log
Epoch 12/200, step:100/1487, loss:0.08269882, overflow:False, loss_scale:32768.0, step cost:182ms
Epoch 12/200, step:200/1487, loss:0.08838322, overflow:False, loss_scale:32768.0, step cost:183ms
Epoch 12/200, step:300/1487, loss:0.06864901, overflow:False, loss_scale:32768.0, step cost:190ms
Epoch 12/200, step:400/1487, loss:0.10473227, overflow:False, loss_scale:32768.0, step cost:185ms
Epoch 12/200, step:500/1487, loss:0.07113585, overflow:False, loss_scale:32768.0, step cost:186ms
Epoch 12/200, step:600/1487, loss:0.12277897, overflow:False, loss_scale:32768.0, step cost:193ms
Epoch 12/200, step:700/1487, loss:0.07687371, overflow:False, loss_scale:32768.0, step cost:189ms
Epoch 12/200, step:800/1487, loss:0.052616842, overflow:False, loss_scale:65536.0, step cost:184ms
Epoch 12/200, step:900/1487, loss:0.057932653, overflow:False, loss_scale:32768.0, step cost:184ms
Epoch 12/200, step:1000/1487, loss:0.24258433, overflow:False, loss_scale:32768.0, step cost:183ms
Epoch 12/200, step:1100/1487, loss:0.04909046, overflow:False, loss_scale:32768.0, step cost:189ms
Epoch 12/200, step:1200/1487, loss:0.07944703, overflow:False, loss_scale:32768.0, step cost:184ms
Epoch 12/200, step:1300/1487, loss:0.08049801, overflow:False, loss_scale:32768.0, step cost:185ms
Epoch 12/200, step:1400/1487, loss:0.065799646, overflow:False, loss_scale:32768.0, step cost:174ms
eval dataset size:500
eval image 100/500 done, step cost: 593ms
eval image 200/500 done, step cost: 586ms
eval image 300/500 done, step cost: 585ms
eval image 400/500 done, step cost: 649ms
eval image 500/500 done, step cost: 580ms
====================== Evaluation Result ======================
===> class: road, IoU: 0.9600858858787784
===> class: sidewalk, IoU: 0.715323303638259
===> class: building, IoU: 0.8837955897151115
===> class: wall, IoU: 0.33827261107225753
===> class: fence, IoU: 0.41190368390165694
===> class: pole, IoU: 0.5130745390096472
===> class: traffic light, IoU: 0.53816132073092
===> class: traffic sign, IoU: 0.6545785767944123
===> class: vegetation, IoU: 0.9031885848185216
===> class: terrain, IoU: 0.5344057148054628
===> class: sky, IoU: 0.9240028381718857
===> class: person, IoU: 0.6519006647756983
===> class: rider, IoU: 0.3307419025377471
===> class: car, IoU: 0.8848571647533297
===> class: truck, IoU: 0.265923872794352
===> class: bus, IoU: 0.39119923446452287
===> class: train, IoU: 0.31510354339228125
===> class: motorcycle, IoU: 0.2662360862992992
===> class: bicycle, IoU: 0.6395381347665564
===============================================================
===> mIoU: 0.5853838553853, ckpt: segformer_mit_b1_12.ckpt
===============================================================
```

Model results are saved in the checkpoint folder by default.

```shell
# ls checkpoint/
-r-------- 1 root root 176884411 Mar 31 15:28 segformer_mit_b1_1.ckpt
-r-------- 1 root root 176884411 Mar 31 15:40 segformer_mit_b1_2.ckpt
-r-------- 1 root root 176884411 Mar 31 15:50 segformer_mit_b1_3.ckpt
-r-------- 1 root root 176884411 Mar 31 16:00 segformer_mit_b1_4.ckpt
-r-------- 1 root root 176884411 Mar 31 16:09 segformer_mit_b1_5.ckpt
-r-------- 1 root root 176884411 Mar 31 16:19 segformer_mit_b1_6.ckpt
-r-------- 1 root root 176884411 Mar 31 16:29 segformer_mit_b1_7.ckpt
-r-------- 1 root root 176884411 Mar 31 16:39 segformer_mit_b1_8.ckpt
-r-------- 1 root root 176884411 Mar 31 16:49 segformer_mit_b1_9.ckpt
-r-------- 1 root root 176884411 Mar 31 17:13 segformer_mit_b1_best.ckpt
```

### Distributed training

Prepare the hccl_8p.json files, before run network. Run the script of [hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py).

```shell  
# The following parameter "[0-8)" indicates that the hccl_8p.json file of cards 0 to 7 is generated.
python hccl_tools.py --device_num "[0,8)"
```

Run the following command on the Ascend using distributed training:

```shell
bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE]
bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional)
bash run_standalone_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional)
bash run_standalone_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] [CONFIG_FILE] [DATASET_PATH](optional) [PRETRAINED_CKPT_PATH](optional) [RUN_EVAL](optional)
For example: bash run_distribute_train.sh 8 /path/hccl_8p.json /segformer/config/segformer.b0.512x1024.city.yaml /segformer/data/cityscapes/"
```

```log
Parameter description:
DEVICE_NUM: number of cards for training. This parameter is mandatory.
RANK_TABLE_FILE: indicates the path of the networking information file. This parameter is mandatory.
CONFIG_FILE: configuration file path. This parameter is mandatory.
DATASET_PATH: Dataset path. This parameter is optional. The default value is data/cityscapes.
PRECultured_CKPT_PATH: path of the pre-trained model. This parameter is optional. The default value is pretrained/pretrained/ms_pretrained_b0.ckpt.
RUN_EVAL: indicates whether to evaluate the model during training. The default value is True. This parameter is optional.
```

## Evaluation

### Evaluation Process 910

Run the following script to perform the evaluation:

```shell
bash run_eval.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH]
For example: bash run_eval.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/checkpoint/segformer_mit_b1_10.ckpt /segformer/data/cityscapes/
```

You can also run the python script directly.

```shell
python eval.py --config_path=[CONFIG_FILE] --eval_ckpt_path=[CKPT_PATH] --data_path=[DATASET_PATH]
```

```log
Parameter description:
DEVICE_ID: ID of the card for evaluation. This parameter is mandatory. If the training is started using Python, you can run the export DEVICE_ID=X command to specify the card number. X indicates the card number.
CONFIG_FILE: indicates the configuration file path. This parameter is mandatory.
CKPT_PATH: indicates the path of the model file to be evaluated. This parameter is mandatory.
DATASET_PATH: data set path, which is mandatory.
```

### Evaluation Result 910

The preceding python commands will run in the background. You can view the results in the eval.log file. The accuracy of the test dataset is as follows:

```log
eval dataset size:500
eval image 100/500 done, step cost: 550ms
eval image 200/500 done, step cost: 572ms
eval image 300/500 done, step cost: 569ms
eval image 400/500 done, step cost: 571ms
eval image 500/500 done, step cost: 572ms
====================== Evaluation Result ======================
===> class: road, IoU: 0.9660652483316745
===> class: sidewalk, IoU: 0.7492563854504648
===> class: building, IoU: 0.8866242816349342
===> class: wall, IoU: 0.31656804931621124
===> class: fence, IoU: 0.3411113046083727
===> class: pole, IoU: 0.5398745184914306
===> class: traffic light, IoU: 0.5906606518622559
===> class: traffic sign, IoU: 0.6890628404136455
===> class: vegetation, IoU: 0.9099730088219271
===> class: terrain, IoU: 0.5330567771230307
===> class: sky, IoU: 0.9325007178024083
===> class: person, IoU: 0.7382945383844727
===> class: rider, IoU: 0.45634221493539334
===> class: car, IoU: 0.9014416024726345
===> class: truck, IoU: 0.3700774842437022
===> class: bus, IoU: 0.3706747870677765
===> class: train, IoU: 0.34248168670673496
===> class: motorcycle, IoU: 0.4306732371022434
===> class: bicycle, IoU: 0.7107199409993931
===============================================================
===> mIoU: 0.6197610145141426, ckpt: segformer_mit_b1_22.ckpt
===============================================================
all eval process done, cost:353s
```

## Export

### Export Process

To export the mindir or air model, run the following command:

```shell
python export.py --config_path [CONFIG_PATH] --export_ckpt_path [EXPORT_CKPT_PATH] --export_format [EXPORT_FORMAT]
```

The config_path and export_ckpt_path parameters are mandatory. The export_format parameter must be set to ["AIR", "MINDIR"]. The default value is AIR.

### Export Result

The name of the exported model file starts with the name of the configuration file and ends with the export type.

```shell
-r-------- 1 root root 59788012 Apr  3 15:47 segformer.b1.1024x1024.city.air
-r-------- 1 root root 59185783 Apr  3 15:49 segformer.b1.1024x1024.city.mindir
```

## Inference

### Inference process

Run the following command to perform inference:

```bash
bash run_infer.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH]
bash run_infer.sh [DEVICE_ID] [CONFIG_FILE] [CKPT_PATH] [DATASET_PATH] [OUTPUT_PATH](optional)
For example: bash run_infer.sh 0 /segformer/config/segformer.b0.512x1024.city.yaml /segformer/checkpoint/segformer_mit_b1_10.ckpt /segformer/data/cityscapes/
```

You can also run the python script directly.

```shell
python infer.py --config_path=[CONFIG_FILE] --infer_ckpt_path=[CKPT_PATH] --data_path=[DATASET_PATH] --infer_output_path=[OUTPUT_PATH](optional)
```

```log
Parameter description:
DEVICE_ID: ID of the card that performs the inference. This parameter is mandatory. If you use the Python script to start training, you can run the export DEVICE_ID=X command to specify the card number. X indicates the card number.
CONFIG_FILE: indicates the configuration file path. This parameter is mandatory.
CKPT_PATH: indicates the path of the model file for inference. This parameter is mandatory.
DATASET_PATH: data set path, which is mandatory.
OUTPUT_PATH: (Optional) Path for storing the inference result. The default value is infer_result.
```

### Inference result

For example, the preceding python command is run in the background. You can view the result in the infer.log file. The result is as follows:

```log
get image size:398, infer result will save to /segformer/infer_result
infer 100/398 done, step cost:1958ms
infer 200/398 done, step cost:1898ms
infer 300/398 done, step cost:1894ms
all infer process done, cost:819s
```

By default, the inference result is stored in the infer_result folder.

```shell
# ll infer_result/
-rw------- 1 root root   26318 Mar 31 17:35 munich_000005_000019_leftImg8bit_color.png    # color image generated from gray image
-rw------- 1 root root   21199 Mar 31 17:35 munich_000005_000019_leftImg8bit_gray.png     # inferred gray image
-rw------- 1 root root 1778554 Mar 31 17:35 munich_000005_000019_leftImg8bit_overlap.png  # overlapping image of color image and original image
-rw------- 1 root root 2279236 Mar 31 17:35 munich_000005_000019_leftImg8bit.png          # original image
```

## Performance

### Training performance

| Parameters                 | Ascend 910                                                   |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | Segformer                                                    |
| Resource                   | Ascend 910; CPU 2.70GHz, 96cores; Memory 1510G; OS Euler2.7  |
| uploaded Date              | 03/04/2023 (month/day/year)                                  |
| MindSpore Version          | master                                                       |
| Dataset                    | Cityscapes                                                   |
| Training Parameters        | epoch=300, steps per epoch=186, batch_size = 16              |
| Optimizer                  | AdamWeightDecay                                              |
| Loss Function              | Softmax Cross Entropy                                        |
| Backbone                   | mit_b0                                                       |
| Loss                       | 0.03060256                                                   |
| Speed                      | 100 ms/step                                                  |
| Total time                 | 95 mins                                                      |
| Parameters (M)             | 3.8                                                          |
| Checkpoint for Fine tuning | 43M (.ckpt file)                                             |

## Precision

| Encoder Model Size    | image size      | mIoU    |
| --------------------- | --------------- | ------- |
| MiT-B0                | 512 * 1024      | 70.88   |
| MiT-B0                | 1024 * 1024     | 71.54   |
| MiT-B1                | 1024 * 1024     | 75.34   |
| MiT-B2                | 1024 * 1024     | 78.53   |
| MiT-B3                | 1024 * 1024     | 78.97   |
| MiT-B4                | 1024 * 1024     | 79.65   |
| MiT-B5                | 1024 * 1024     | 79.90   |

## Contributions

If you want to contribute, please review the [contribution guidelines](https://gitee.com/mindspore/models/blob/master/CONTRIBUTING.md) and [how_to_contribute](https://gitee.com/mindspore/models/tree/master/how_to_contribute)

## ModelZoo Home

Please check the official [homepage](https://gitee.com/mindspore/models).
