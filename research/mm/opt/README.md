# OPT: Omni-Perception Pre-Trainer for Cross-Modal Understanding and Generation

> This repo. is about the implementation of the OPT model, a visual-text-audio pre-training model.

## OPT Description

OPT is constructed in an encoder-decoder framework, including three single-modal encoders to generate token-based
embeddings for each modality, a cross-modal encoder to encode the correlations among the three modalities, and two
cross-modal decoders to generate text and image respectively. For the OPT's pre-training, we design a multi-task pretext
learning scheme to model multi-modal resources from three different data granularities, \ie, token-, modality-, and
sample-level modeling, through which OPT learns to align and translate among different modalities. The pre-training task
is carried out on a large amount of image-text-audio triplets from Open Images. Experimental results show that OPT can
learn strong image-text-audio multi-modal representations and achieve promising results on a variety of cross-modal
understanding and generation tasks.

[Paper](https://arxiv.org/abs/2107.00249): Liu J, Zhu X, Liu F, et al. OPT: Omni-Perception Pre-Trainer for Cross-Modal
Understanding and Generation[J]. arXiv preprint arXiv:2107.00249, 2021.

## Model Architecture

<img src="./8159685ccda2be63fd92cb1109fe7f8.png" alt="image-20211117104252504" />

## Pretraining Dataset

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant
domain/network architecture. In the following sections, we will introduce how to run the scripts using the related
dataset below.

- [CC3M](https://github.com/google-research-datasets/conceptual-captions) provides about 3 million image-text pairs. We
  translate English captions into Chinese.
- [COCO Captions](https://cocodataset.org/#home) provides about 415K image-text pairs. We translate English captions
  into Chinese.
- AIC provides about 1 million Chinese image-text pairs.

## Environment Requirements

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## Pre-Training

```shell
bash run_distributed_ascend.sh [RANK_TABLE_FILE] [PRETRAINED_MODEL]
```

## Finetuning on Downstream Tasks

### Cross-Modal Retrieval

- Dataset: [COCO Captions](https://cocodataset.org/#home) contains about 415K image-text pairs.

- Starting training:

```shell
bash scripts/run_standalone_ascend_train_retrieval.sh
```

- Starting inference:

```shell
bash scripts/run_standalone_ascend_inf_retrieval.sh
```

### Image Captioning (i.e. Image-to-Text Generation)

- Dataset:[COCO Captions](https://cocodataset.org/#home) provides about 415K image-text pairs. We translate English
  captions into Chinese.

- Starting training:

```shell
bash scripts/train_caption.sh
```

- Starting inference:

```shell
bash scripts/test_caption.sh
```

### Result
