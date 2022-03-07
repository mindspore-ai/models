## Contents

[查看中文](./README_CN.md)

- [Contents](#contents)
- [Wukong Dataset](#wukong-dataset)
- [Environment requirements](#environment-requirements)
- [Quick Start](#quick-start)
    - [Prepare Dataset](#prepare-dataset)
    - [Prepare files required for tokenizer](#prepare-files-required-for-tokenizer)
    - [Propare prompt files](#propare-prompt-files)
    - [Prepare pretrained model checkpoint](#prepare-pretrained-model-checkpoint)
    - [Zero-shot Classification](#zero-shot-classification)

## Wukong Dataset

This project provides the zero-shot classification task on ILSVRC dataset using multi-modality large-scale model pretrained on Noah-Wukong dataset. Model structure as follows:

|Model|Wukong_Vit|
|:----|:----|
|Embedding dimension|256|
|Input image resolution|224x224|
|Image encoder| |
|patch_size|14|
|width|1024|
|#layers|24|
|#heads|16|
|Input text token length|32|
|Text encoder| |
|#layers|12|
|#width|768|
|#heads|12|

More benchmark of the multi-modality modal please refer to [Noah-Wukong Benchmark](https://wukong-dataset.github.io/wukong-dataset/benchmark.html)

## Environment requirements

- Hardware
    - Ascend processor
- Framework
    - [Mindspore](https://www.mindspore.cn/ "Mindspore")
- Tutorial
    - [Mindspore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [Mindspore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## Quick Start

### Prepare Dataset

- Download ILSVRC dataset and organize the file as follows:

```text
.
└── data_root
     ├── class1
     │    ├── 000000000001.jpg
     │    ├── 000000000002.jpg
     │    ├── ...
     ├── class2
     │    ├── 000000000001.jpg
     │    ├── 000000000002.jpg
     │    ├── ...
     ├── class3
     │    ├── 000000000001.jpg
     │    ├── 000000000002.jpg
     │    ├── ...
     ├── classN
     ├── ...
```

- Download corresponding Chinese class name file [imagenet_class_name_zh.json](https://drive.google.com/file/d/1LL0GygtD-ob19EwRuSTfm43ZuFqqy4Q_/view?usp=sharing) and place it the same folder with main.py .

### Prepare files required for tokenizer

Download following files and place them under src/tools/

- English: [bpe_simple_vocab_16e6.txt.gz](https://drive.google.com/file/d/1SCrD7wewUhxljCggEQxQr1khCfT6mGnj/view?usp=sharing)
- Chinese: [vocab_zh.txt](https://drive.google.com/file/d/1jmbTqpnef3czYWMK2QXYm_i79FpV1bxl/view?usp=sharing)

### Propare prompt files

Download prompt file[zh_templates.txt](https://drive.google.com/file/d/1Zky3V9LYRGBaAZzGEuTNAINYHLVPn8bd/view?usp=sharing)to src/tools/.This file defines the prompts used in zero-shot classification task. The number of prompts can be modified according to time/performance balance. Custom prompts are also allowed.

### Prepare pretrained model checkpoint

Download pretrained checkpoint file [
wk100m_yfcc_vit_l_14_filip_lit.pth](https://drive.google.com/file/d/19Xx9UbDeitSoy5MB-vs9LSHa5nDNu4FX/view?usp=sharing).
Use src/tools/convert.py to convert to proper format.

```shell
python convert.py [pth_path] [pkl_path]
```

### Zero-shot Classification

Run eval.py to do zero-shot classification.

```shell
python eval.py --ckpt_path [ckpt_path] --dataset_path [/path/to/data_root] --batch_size [batch size]
```

evaluation result is

```text
INFO:main:correct @1: 51.51; correct @5: 78.33
```

Detailed zero-shot classification performance is as below:

|dataset|ResNet50|ResNet101|ViT-B/32|ViT-B/16|Wukong_ViT (global similarity)|Wukong_ViT|Wukong_ViT-500M|Wukong_Swin (global similarity)|Wukong_Swin|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|CIFAR10|49|60.3|89|89.5|93.6|90.6|90.3|95.3|95.5|
|CIFAR100|23.5|31.1|57.3|49.4|64.6|66.3|65.3|69.1|77.2|
|Caltech101|72.3|76.3|83.6|84.4|86|89.9|89.2|87.6|91.6|
|Caltech256|58.4|64.3|70.5|75.4|76.8|86.2|86|78.2|88.4|
|Sports|78|83.3|90.6|88.1|86.8|97.8|96.9|93.4|99.1|
|Flowers|29|30.9|38|42.9|55.1|69.4|71.6|54.5|75.1|
|Food101|37|43.6|42.7|49.4|53.5|70|65.2|46.6|66.1|
|Pets|41.7|43.8|44.9|51.2|46.4|61.3|67|47.7|64.5|
|SUN397|33.1|38.4|39.8|42.7|44.9|60.2|58.9|42.4|56.5|
|ImageNet|28.3|32.8|33.2|38.3|44.8|54|54.3|43.6|58.5|
|ImageNet-r|38.9|47.1|52.3|61.9|67.4|72.2|77.5|49.3|55.3|
|ImageNet-a|14.8|20.8|22.2|35.4|55.1|52.2|53.2|36.8|41.9|
|ImageNet-s|16|20.6|24|29|33.3|36.5|36.8|24.7|31.4|
|DTD|22.4|25.2|27.1|31.7|35.6|46.4|44.6|31.1|39.8|
|Dogs|12.1|12.6|17.3|23.3|21.1|29.4|35.4|22.9|40.3|
|EuroSAT|17.6|12.5|35.3|43.9|39.7|25.5|32.3|28.8|21|
|Aircraft|10.1|10|10.3|14.8|20.8|22.3|21.5|8.9|10.1|