# Contents

- [Contents](#contents)
- [MobileNetV3 Description](#mobilenetv3-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script description](#script-description)
    - [Script and sample code](#script-and-sample-code)
    - [Training process](#training-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [Eval process](#eval-process)
        - [Usage](#usage-1)
        - [Launch](#launch-1)
        - [Result](#result-1)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
            - [Result](#result-2)
        - [Infer with ONNX](#infer-with-onnx)
            - [Result](#result-3)
- [Model description](#model-description)
    - [Performance](#performance)
        - [Training Performance](#training-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [MobileNetV3 Description](#contents)

MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware- aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances.Nov 20, 2019.

[Paper](https://arxiv.org/pdf/1905.02244) Howard, Andrew, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang et al. "Searching for mobilenetv3." In Proceedings of the IEEE International Conference on Computer Vision, pp. 1314-1324. 2019.

# [Model architecture](#contents)

The overall network architecture of MobileNetV3 is show below:

[Link](https://arxiv.org/pdf/1905.02244)

# [Dataset](#contents)

Dataset used: [imagenet](http://www.image-net.org/)

- Dataset size: ~125G, 224*224 colorful images in 1000 classes
    - Train: 120G, 1281167 images
    - Test: 5G, 50000 images
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware（GPU/CPU）
    - Prepare hardware environment with GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```python
├── MobileNetV3
  ├── Readme.md              # descriptions about MobileNetV3
  ├── scripts
  │   ├──run_train.sh        # shell script for train
  │   ├──run_eval.sh         # shell script for evaluation
  │   ├──run_infer_310.sh         # shell script for inference
  │   ├──run_onnx.sh         # shell script for onnx inference
  ├── src
  │   ├──config.py           # parameter configuration
  │   ├──dataset.py          # creating dataset
  │   ├──lr_generator.py     # learning rate config
  │   ├──mobilenetV3.py      # MobileNetV3 architecture
  ├── train.py               # training script
  ├── eval.py                #  evaluation script
  ├── infer_onnx.py          #  onnx inference script
  ├── export.py              # export mindir script
  ├── preprocess.py              # inference data preprocess script
  ├── postprocess.py              # inference result calculation script  
  ├── mindspore_hub_conf.py  #  mindspore hub interface
```

## [Training process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- GPU: bash run_trian.sh GPU [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [DATASET_PATH]
- CPU: bash run_trian.sh CPU [DATASET_PATH]

### Launch

```shell
# training example
  python:
      GPU: python train.py --dataset_path ~/imagenet/train/ --device_targe GPU
      CPU: python train.py --dataset_path ~/cifar10/train/ --device_targe CPU
  shell:
      GPU: bash run_train.sh GPU 8 0,1,2,3,4,5,6,7 ~/imagenet/train/
      CPU: bash run_train.sh CPU ~/cifar10/train/
```

### Result

Training result will be stored in the example path. Checkpoints will be stored at `. /checkpoint` by default, and training log  will be redirected to `./train/train.log` like followings.

```bash
epoch: [  0/200], step:[  624/  625], loss:[5.258/5.258], time:[140412.236], lr:[0.100]
epoch time: 140522.500, per step time: 224.836, avg loss: 5.258
epoch: [  1/200], step:[  624/  625], loss:[3.917/3.917], time:[138221.250], lr:[0.200]
epoch time: 138331.250, per step time: 221.330, avg loss: 3.917
```

## [Eval process](#contents)

### Usage

You can start training using python or shell scripts. The usage of shell scripts as follows:

- GPU: bash run_infer.sh GPU [DATASET_PATH] [CHECKPOINT_PATH]
- CPU: bash run_infer.sh CPU [DATASET_PATH] [CHECKPOINT_PATH]

### Launch

```shell
# infer example
  python:
    GPU: python eval.py --dataset_path ~/imagenet/val/ --checkpoint_path mobilenet_199.ckpt --device_targe GPU
    CPU: python eval.py --dataset_path ~/cifar10/val/ --checkpoint_path mobilenet_199.ckpt --device_targe CPU

  shell:
    GPU: bash run_infer.sh GPU ~/imagenet/val/ ~/train/mobilenet-200_625.ckpt
    CPU: bash run_infer.sh CPU ~/cifar10/val/ ~/train/mobilenet-200_625.ckpt
```

> checkpoint can be produced in training process.

### Result

Inference result will be stored in the example path, you can find result like the followings in `val.log`.

```bash
result: {'acc': 0.71976314102564111} ckpt=/path/to/checkpoint/mobilenet-200_625.ckpt
```

## [Inference Process](#contents)

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```shell
python export.py --checkpoint_path [CKPT_PATH] --device_target [DEVICE] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`DEVICE` should be in ['Ascend', 'GPU', 'CPU']
`FILE_FORMAT` should be in "MINDIR"

### [Infer on Ascend310](#contents)

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.
Current batch_Size for imagenet2012 dataset can only be set to 1.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` specifies path of used "MINDIR" model.
- `DATA_PATH` specifies path of imagenet datasets.
- `DEVICE_ID` is optional, default value is 0.

#### [Result](#contents)

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
Eval: top1_correct=37051, tot=50000, acc=74.10%
```

### [Infer with ONNX](#contents)

Before Inferring, you need to export the ONNX model.

- Download the [ckpt] of the model from the [mindspore official website](https://mindspore.cn/resources/hub/details?MindSpore/1.8/mobilenetv3_imagenet2012).
- The command to export the ONNX model is as follows

```shell
python export.py --checkpoint_path[ckpt_path] --device_target "GPU" --file_name mobilenetv3.onnx --file_format  "ONNX"
```

- Infer with Python scripts

```shell
python3 infer_onnx.py --onnx_path 'mobilenetv3.onnx' --dataset_path './imagenet/val'
```

- Infer with shell scripts

```shell
bash ./scripts/run_onnx.sh [ONNX_PATH] [DATASET_PATH] [PLATFORM]
```

Infer results are output in `infer_onnx.log`

- Note 1: the above scripts need to be run in the mobilenetv3 directory.

- Note 2: the validation data set needs to be preprocessed in the form of folder. For example,

  ```reStructuredText
  imagenet
   -- val
    -- n01985128
    -- n02110063
    -- n03041632
    -- ...
  ```

- Note 3: `PLATFORM` only supports CPU and GPU, default value is GPU.

#### [Result](#contents)

The reasoning results are output on the command line, and the results are as follows

```log
ACC_TOP1 = 0.74436
ACC_TOP5 = 0.91762
```

# [Model description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | MobilenetV3               |
| -------------------------- | ------------------------- |
| Model Version              | large                     |
| Resource                   | NV SMX2 V100-32G          |
| uploaded Date              | 07/05/2021                |
| MindSpore Version          | 1.3.0                     |
| Dataset                    | ImageNet                  |
| Training Parameters        | src/config.py             |
| Optimizer                  | Momentum                  |
| Loss Function              | SoftmaxCrossEntropy       |
| outputs                    | probability               |
| Loss                       | 1.913                     |
| Accuracy                   | ACC1[77.57%] ACC5[92.51%] |
| Total time                 | 1433 min                  |
| Params (M)                 | 5.48 M                    |
| Checkpoint for Fine tuning | 44 M                      |
|  Scripts                   | [Link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv3)|

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
