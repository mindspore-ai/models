# Contents

- [ISyNet Description](#isynet-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Training Process](#training-process)
        - [Evaluation Process](#evaluation-process)
        - [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)

# [ISyNet Description](#contents)

ISyNets is a set of architectures designed to be fast on the Ascend 310 hardware and accurate at the same time. We show the advantage of the designed architectures for the NPU devices on ImageNet and the generalization ability for the downstream classification and detection tasks.

[Paper](https://arxiv.org/abs/2109.01932): Alexey Letunovskiy, Vladimir Korviakov, Vladimir Polovnikov, Anastasiia Kargapoltseva, Ivan Mazurenko, Yepan Xiong. ISyNet: Convolutional Neural Networks design for AI accelerator.

# [Model architecture](#contents)

The overall network architecture of ISyNet is described in our [paper](https://arxiv.org/abs/2109.01932).

# [Dataset](#contents)

Dataset used: [ImageNet 2012](http://image-net.org/challenges/LSVRC/2012/)

- Dataset size:
    - Train: 1.2 million images in 1,000 classes
    - Test: 50,000 validation images in 1,000 classes
- Data format: RGB images.
    - Note: Data will be processed in src/dataset.py

# [Environment Requirements](#contents)

- Hardware (GPU, Ascend)
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```markdown
├── README.md                           # descriptions about ISyNet
├── script
│   ├── run_eval_gpu.sh                 # gpu evaluation script
│   ├── run_infer_310_om.sh                # evaluation in Ascend 310 script
│   ├── run_standalone_train_gpu.sh     # training script on single GPU
│   └── run_distributed_train.sh        # distributed training script on multiple GPUs
├── ISyNet
│   ├── model.py            # architecture cell constructor
│   ├── backbone.py         # CNN backbone constructor
│   ├── head.py             # classification head constructor
│   ├── layers.py           # definition of model's layers
│   └── json_parser_backbone.py     # parser of architecture's json files
├── src
│   ├── CrossEntropySmooth.py         # cross entropy with label smooth
│   ├── KLLossAscend.py               # KL loss for Ascend
│   ├── autoaugment.py                # auto augmentation
│   ├── dataset.py                    # dataset
│   ├── ema_callback.py               # callback for EMA(exponential moving average)
│   ├── eval_callback.py              # eval callback
│   ├── lr_generator.py               # learning rate scheduling
│   ├── metric.py                     # metric
│   ├── momentum.py                   # SGD momentum
│   └── model_utils                   # utils
├── utils
│   ├── count_acc.py                  # count accuracy of the model executed on the Ascend 310
│   ├── export.py                     # export mindir script
│   └── preprocess_310.py             # preprocess imagenet dataset and convert it from jpeg to bin files for Ascend 310 inference
├── config                            # yml configs
├── eval.py                 # evaluation script
└── train.py                # training script
```

### [Training process](#contents)

#### Launch

```bash
# training on single GPU
  bash run_standalone_train_gpu.sh  [DATA_PATH] [CONFIG_PATH]
# training on multiple GPUs
  bash run_distributed_train_gpu.sh [DATA_PATH] [CONFIG_PATH]
# training on multiple Ascends
  bash run_distributed_train.sh [RANK_TABLE] [DATA_PATH] [CONFIG_PATH]
```

> checkpoints will be saved in the ./train/output folder (single GPU)
./train_parallel/output/ folder (multiple GPUs)
./train_parallel0-7/output/ folder (multiple Ascends)

### [Evaluation Process](#contents)

#### Launch

```bash
# infer example

bash run_eval_gpu.sh [DATA_PATH] [CHECKPOINT_PATH] [CONFIG_PATH]
```

Checkpoint can be produced in training process.

### [Inference Process](#contents)

### [Export MindIR](#contents)

Export MindIR on local

```shell
python utils/export.py --jsonFile [JSON_FILE] --file_name [FILE_NAME] --file_format [FILE_FORMAT] --checkpoint_file_path [CHECKPOINT_PATH]
```

The checkpoint_file_path parameter is required,
`EXPORT_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Overall procedure of running on the Ascend 310 consists of the following steps:

1. Conversion of the ImageNet validation set to bin files
2. Conversion of the CKPT MindSpore model to AIR format
3. Conversion of the AIR model to OM format
4. Building the inference executable program
5. Running OM model and dumping the inference results
6. Computing the validation accuracy

We only provide an example of inference using OM model.
Current batch_size can only be set to 1 for the accuracy measurement.

Step 1 should be done only once with the following command:

```shell

# ImageNet files conversion

python3.7 preprocess_310.py --data_path [IMAGENET_ORIGINAL_VAL_PATH] --output_path [IMAGENET_PREPROCESSED_VAL_PATH]
```

- `IMAGENET_ORIGINAL_VAL_PATH` is an input path to the original ImageNet validation folder.
- `IMAGENET_PREPROCESSED_VAL_PATH` is an output path where the converted ImageNet files will be saved.

Steps 2 to 6 are fully automated with the following script:

```shell
# Ascend310 inference
cd scripts
export ASCEND_PATH=/usr/local/Ascend/ # set another path to the Ascend toolkit if needed
bash run_infer_310_om.sh [MODEL_JSON_FILE] [MODEL_CKPT_FILE] [IMAGENET_PREPROCESSED_VAL_PATH] [BATCH_SIZE] [MODE]
```

- `MODEL_JSON_FILE` is a path to model JSON description file.
- `MODEL_CKPT_FILE` is a path to pretrained model CKPT file.
- `IMAGENET_PREPROCESSED_VAL_PATH` is a path to the converted ImageNet files prepared by preprocess_310.py script.
- `BATCH_SIZE` is a batch size. Computing the validation accuracy is supported only for batch size 1.
               Inference and profiling is supported for any size of batch, but input files should be concatenated correspondingly.
- `MODE` is a inference regime, can be "inference" or "profile"
    - "inference" means simple running the model, saving the outputs and measuring the average latency.
    - "profile" means profiling the model and saving the detailed analysis of each operation in the model graph.

After the validation

## result

Inference result is saved in current path, you can find result like this in acc.log file.

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Model               | ImageNet Top-1 mindspore | Latency, ms* | Params, x10^6 | MACs, x10^9 | Checkpoint |
| ------------------- | ------------------------ | ------------ | ------------- |-------------|------------|
| ISyNet-N0           | 75.03                    |0.43          | 9.59          | 1.13        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N0_ascend_v160_imagenet2012_research_cv_top1acc75.00.ckpt) |
| ResNet-18+          | 74.3                     |0.63          | 11.69         | 2.28        |  |
| ISyNet-N1           | 76.41                    |0.72          | 7.42          | 2.85        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N1_ascend_v160_imagenet2012_research_cv_top1acc76.00.ckpt) |
| ISyNet-N1-S1        | 76.78                    |0.74          | 7.82          | 2.88        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N1S1_ascend_v160_imagenet2012_research_cv_top1acc76.00.ckpt) |
| ISyNet-N1-S2        | 77.45                    |0.83          | 8.86          | 3.34        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N1S2_ascend_v160_imagenet2012_research_cv_top1acc77.45_top5acc93.68.ckpt) |
| ISyNet-N1-S3        | 78.25                    |0.97          | 10.81         | 4.12        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N1S3_ascend_v160_imagenet2012_research_cv_top1acc78.00.ckpt) |
| ResNet-34+          | 77.95                    |1.05          | 21.8          | 4.63        |  |
| ISyNet-N2           | 79.07                    |1.10          | 19.43         | 4.93        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N2_ascend_v160_imagenet2012_research_cv_top1acc79.00.ckpt) |
| ISyNet-N3           | 80.43                    |1.55          | 20.47         | 7.32        | [Link](https://download.mindspore.cn/models/r1.6/isynet_N3_ascend_v160_imagenet2012_research_cv_top1acc80.00.ckpt) |
| ResNet-50+          | 80.18                    |1.64          | 25.56         | 5.19        |  |

Latency is measured on Ascend 310 NPU accelerator with batch size 16 in fp16 precision.

### IsyNet-N3 on ImageNet

| Parameter                  | Value
| -------------------------- | -----------------------------------------------
| Model Version              | IsyNet-N3                                                   |
| Resource                   | Ascend 910, 8 NPU; CPU 2.60GHz, 96 cores; Memory 1500G; OS Euler2.5 |
| uploaded Date              | 03/01/2022 (month/day/year)                                 |
| MindSpore Version          | 1.5.0                                                       |
| Dataset                    | ImageNet                                                    |
| Training Parameters        | epoch=550, steps_per_epoch=1251, batch_size=128; Deep Mutual Learning, RandAugmentation, Last BatchNorm; lr=0.001, warmup=40epochs, cosine scheduler  |
| Optimizer                  | AdamW                                                       |
| Loss Function              | Softmax Cross Entropy and KL Divergence                     |
| outputs                    | probability                                                 |
| Speed                      | 6 min / epoch                                               |
| Total time                 | 55 hours                                                    |
| Parameters (M)             | 20.47                                                       |
| config                     | config/IsyNet-N1-S3_imagenet2012_config_MA_v7.yaml          |

See more details in the [Paper](https://arxiv.org/abs/2109.01932).

# [Description of Random Situation](#contents)

We set the seed inside dataset.py. We also use random seed in train.py.
