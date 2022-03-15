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

|Models|Embedding dimension|Image encoder|similarity|# vis_token|checkpoints|
|:----|:----|:----|:----|:----|:----|
|Wukong_ViT-B^G|512|Vit-b/32|Global|/|[download](https://drive.google.com/file/d/1kDCF3rsd7Ckioag0Nzmiu2ZKVTAk7gej/view?usp=sharing)|
|Wukong_ViT-B^F|512|Vit-b/32|Token-wise|/|[download](https://drive.google.com/file/d/1xXaZ7K1E9RbboiUJCeB0kdjRaa3KJUM1/view?usp=sharing)|
|Wukong_ViT-B|512|Vit-b/32|Token-wise|12|[download](https://drive.google.com/file/d/17szMVtb_Ea1YSXgpV_bLH175I_2slOeo/view?usp=sharing)|
|Wukong_ViT-L^G|768|Vit-L/14|Global|/|[download](https://drive.google.com/file/d/1vouG2jtOvHAPlKRiWC5XMJBEPvY6F2tv/view?usp=sharing)|
|Wukong_ViT-L ^F|768|Vit-L/14|Token-wise|/|[download](https://drive.google.com/file/d/1Wbf6EbLc38c5qMDHyVcX7gTjFB-wtIfa/view?usp=sharing)|
|Wukong_ViT-L|768|Vit-L/14|Token-wise|24|[download](https://drive.google.com/file/d/1Wbf6EbLc38c5qMDHyVcX7gTjFB-wtIfa/view?usp=sharing)|

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

- Download corresponding Chinese class name file [imagenet_class_name_zh.json](https://drive.google.com/file/d/1LL0GygtD-ob19EwRuSTfm43ZuFqqy4Q_/view?usp=sharing) and place it the same folder with eval.py .

### Prepare files required for tokenizer

Download following files and place them under src/tools/

- English: [bpe_simple_vocab_16e6.txt.gz](https://drive.google.com/file/d/1SCrD7wewUhxljCggEQxQr1khCfT6mGnj/view?usp=sharing)
- Chinese: [vocab_zh.txt](https://drive.google.com/file/d/1jmbTqpnef3czYWMK2QXYm_i79FpV1bxl/view?usp=sharing)

### Propare prompt files

Download prompt file[zh_templates.txt](https://drive.google.com/file/d/1Zky3V9LYRGBaAZzGEuTNAINYHLVPn8bd/view?usp=sharing)to src/tools/.This file defines the prompts used in zero-shot classification task. The number of prompts can be modified according to time/performance balance. Custom prompts are also allowed.

### Prepare pretrained model checkpoint

Download corresponding pretrained checkpoint files following links in the [table](#wukong-dataset).

### Zero-shot Classification

Run eval.py to do zero-shot classification, each model has its config file under src/config/ folder.

```shell
python eval.py --config_paht [config_path] --ckpt_path [ckpt_path] --dataset_path [/path/to/data_root] --batch_size [batch size]
```

evaluation result is something like this

```text
INFO:main:correct @1: 51.51; correct @5: 78.33
```

Detailed zero-shot classification performance is as below:

| |single@1|single@5|embed(80)@1|embed(80)@5|
|:----|:----|:----|:----|:----|
|ViT-B-G|44.68|71.19|47.32|74.3|
|ViT-B-F|32.53|57.51|37.17|63.22|
|ViT-B|45.22|70.69|48.24|73.43|
|ViT-L-G|56.15|79.86|57.54|81.46|
|ViT-L-F|49.74|76.3|52.83|78.88|
|ViT-L|50.22|74.79|54.43|80.1|