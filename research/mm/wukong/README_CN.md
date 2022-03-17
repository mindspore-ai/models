## 目录

[View English](./README.md)

- [目录](#目录)
- [Wukong数据集](#wukong数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
    - [准备ILSVRC数据集](#准备ilsvrc数据集)
    - [准备分词器需要的文件](#准备分词器需要的文件)
    - [准备prompt文件](#准备prompt文件)
    - [准备预训练模型文件](#准备预训练模型文件)
    - [Zero-shot分类推理](#zero-shot分类推理)

## Wukong数据集

该项目提供了基于Noah-Wukong数据集进行预训练得到的多模态大模型，在ILSVRC数据集上进行zero-shot分类的方法。模型结构如下：

|Models|Embedding dimension|Image encoder|similarity|# vis_token|checkpoints|
|:----|:----|:----|:----|:----|:----|
|Wukong_ViT-B^G|512|Vit-b/32|Global|/|[download](https://drive.google.com/file/d/1kDCF3rsd7Ckioag0Nzmiu2ZKVTAk7gej/view?usp=sharing)|
|Wukong_ViT-B^F|512|Vit-b/32|Token-wise|/|[download](https://drive.google.com/file/d/1xXaZ7K1E9RbboiUJCeB0kdjRaa3KJUM1/view?usp=sharing)|
|Wukong_ViT-B|512|Vit-b/32|Token-wise|12|[download](https://drive.google.com/file/d/17szMVtb_Ea1YSXgpV_bLH175I_2slOeo/view?usp=sharing)|
|Wukong_ViT-L^G|768|Vit-L/14|Global|/|[download](https://drive.google.com/file/d/1vouG2jtOvHAPlKRiWC5XMJBEPvY6F2tv/view?usp=sharing)|
|Wukong_ViT-L ^F|768|Vit-L/14|Token-wise|/|[download](https://drive.google.com/file/d/1Wbf6EbLc38c5qMDHyVcX7gTjFB-wtIfa/view?usp=sharing)|
|Wukong_ViT-L|768|Vit-L/14|Token-wise|24|[download](https://drive.google.com/file/d/1Wbf6EbLc38c5qMDHyVcX7gTjFB-wtIfa/view?usp=sharing)|

更多benchmark可以参考[Noah-Wukong Benchmark](https://wukong-dataset.github.io/wukong-dataset/benchmark.html)

## 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境
- 框架
    - [Mindspore](https://www.mindspore.cn/ "Mindspore")
- 如需查看详情，请参考如下资源
    - [Mindspore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [Mindspore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速开始

### 准备ILSVRC数据集

- 下载ILSVRC数据集，需要满足如下文件结构：

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

- 下载对应的中文类名文件 [imagenet_class_name_zh.json](https://drive.google.com/file/d/1LL0GygtD-ob19EwRuSTfm43ZuFqqy4Q_/view?usp=sharing)，放在eval.py同级目录下。

### 准备分词器需要的文件

下载下列文件并放在src/tools/目录下

- 英文： [bpe_simple_vocab_16e6.txt.gz](https://drive.google.com/file/d/1SCrD7wewUhxljCggEQxQr1khCfT6mGnj/view?usp=sharing)
- 中文： [vocab_zh.txt](https://drive.google.com/file/d/1jmbTqpnef3czYWMK2QXYm_i79FpV1bxl/view?usp=sharing)

### 准备prompt文件

下载prompt文件[zh_templates.txt](https://drive.google.com/file/d/1bZFH0dt6hbhn80F74l7c1W0XV09l37zY/view?usp=sharing)至src/tools/目录下。文件指定了zero-shot分类时所使用的prompt形式。可以根据实际情况（运行时间、性能）调整prompt的数量，也可以根据文件中prompt格式新增自定义的prompt。

### 准备预训练模型文件

下载对应模型的预训练参数，下载链接参考[表格](#wukong数据集)

### Zero-shot分类推理

运行下列命令进行zero-shot分类推理，其中各模型的配置文件在src/config目录下。

```shell
python eval.py --config_paht [config_path] --ckpt_path [ckpt_path] --dataset_path [/path/to/data_root] --batch_size [batch size]
```

推理结果如下

```text
INFO:main:correct @1: 51.51; correct @5: 78.33
```

各模型推理性能如下，其中single与embed分别指使用单个prompt与使用zh_templates.txt中80个prompts：

| |single@1|single@5|embed(80)@1|embed(80)@5|
|:----|:----|:----|:----|:----|
|ViT-B-G|44.68|71.19|47.32|74.3|
|ViT-B-F|32.53|57.51|37.17|63.22|
|ViT-B|45.22|70.69|48.24|73.43|
|ViT-L-G|56.15|79.86|57.54|81.46|
|ViT-L-F|49.74|76.3|52.83|78.88|
|ViT-L|50.22|74.79|54.43|80.1|