# Self-Supervision Can Be a Good Few-Shot Learner

This is a [MindSpore](https://www.mindspore.cn/) implementation of the ECCV2022 paper [Self-Supervision Can Be a Good Few-Shot Learner (UniSiam)](https://arxiv.org/abs/2207.09176).

## Contents

- [Contents](#contents)
    - [UniSiam Description](#UniSiam-description)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Script description](#script-description)

## [UniSiam Description](#contents)

Existing few-shot learning (FSL) methods rely on training with a large labeled dataset, which prevents them from leveraging abundant unlabeled data. From an information-theoretic perspective, we propose an effective unsupervised FSL method, learning representations with self-supervision. Following the InfoMax principle, our method learns comprehensive representations by capturing the intrinsic structure of the data. Specifically, we maximize the mutual information (MI) of instances and their representations with a low-bias MI estimator to perform self-supervised pre-training. Rather than supervised pre-training focusing on the discriminable features of the seen classes, our self-supervised model has less bias toward the seen classes, resulting in better generalization for unseen classes. We explain that supervised pre-training and selfsupervised pre-training are actually maximizing different MI objectives. Extensive experiments are further conducted to analyze their FSL performance with various training settings. Surprisingly, the results show that self-supervised pre-training can outperform supervised pre-training under the appropriate conditions. Compared with state-of-the-art FSL methods, our approach achieves comparable performance on widely used FSL benchmarks without any labels of the base classes.

```markdown
@inproceedings{Lu2022Self,
    title={Self-Supervision Can Be a Good Few-Shot Learner},
    author={Lu, Yuning and Wen, Liangjian and Liu, Jianzhuang and Liu, Yajing and Tian, Xinmei},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on?p=self-supervision-can-be-a-good-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on-1)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on-1?p=self-supervision-can-be-a-good-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on-2)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on-2?p=self-supervision-can-be-a-good-few-shot)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervision-can-be-a-good-few-shot/unsupervised-few-shot-image-classification-on-3)](https://paperswithcode.com/sota/unsupervised-few-shot-image-classification-on-3?p=self-supervision-can-be-a-good-few-shot)

## [Dataset](#contents)

- mini-ImageNet
    - download the mini-ImageNet dataset from [google drive](https://drive.google.com/file/d/1BfEBMlrf5UT4aNOoJPaa83CgbGWZAAAk/view?usp=sharing) and unzip it.
    - download the [split files](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet) of mini-ImageNet which created by [Ravi and Larochelle](https://openreview.net/pdf?id=rJY0-Kcll).
    - move the split files to the folder `./split/miniImageNet`

## [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU.
- Framework
    - [MindSpore 1.7](https://www.mindspore.cn/install/en)
- For more information, please check the resources below£º
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Script description](#contents)

Run

```python
python ./train.py --data_path [your DATA FOLDER] --dataset [DATASET NAME] --backbone [BACKBONE] [--OPTIONARG]
```

For example, to train UniSiam model with ResNet-18 backbone and strong data augmentations on mini-ImageNet (V100):

```python
python train.py \
  --dataset miniImageNet \
  --backbone resnet18 \
  --lrd_step \
  --data_path [your mini-imagenet-folder] \
  --save_path [your save-folder]
```
