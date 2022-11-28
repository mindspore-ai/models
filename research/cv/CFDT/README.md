# Contents

- [Contents](#contents)
    - [CFDT Description](#CFDT-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Script description](#script-description)
        - [Script and sample code](#script-and-sample-code)
    - [Eval process](#eval-process)
        - [Usage](#usage)
        - [Launch](#launch)
        - [Result](#result)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [CFDT Description](#contents)

CFDT is a model that can fully leverage both global **C**oarse-grained and local **F**ine-grained features
to build an efficient **D**etection **T**ransformer (CFDT) with transformer backbone and transformer
neck.

> [Paper](https://openreview.net/pdf?id=iuW96ssPQX): A Transformer-Based Object Detector with Coarse-Fine Crossing Representations.
> Zhishan Li,  Ying Nie, Kai Han, Jianyuan Guo, Lei Xie, Chao Xu, Yunhe Wang.
## [Model architecture](#contents)

the overall architecture of CFDT. PTNT Blocks is the abbreviation of
PyramidTNT Blocks. The red dotted line represents the forward propagation of det tokens.

![image-wavemlp](./fig/CFDT.png)

## [Dataset](#contents)

Dataset used: [COCO2017](https://cocodataset.org/#download)

- Dataset size：~19G
    - [Train](http://images.cocodataset.org/zips/train2017.zip) - 18G，118000 images
    - [Val](http://images.cocodataset.org/zips/val2017.zip) - 1G，5000 images
    - [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) -
      241M，instances，captions，person_keypoints etc
- Data format：image and json files
    - The directory structure is as follows:

  ```text
    .
    ├── annotations  # annotation jsons
    └── images
      ├── train2017    # train dataset
      └── val2017      # val dataset
  ```

## [Environment Requirements](#contents)

- Hardware(GPU)
    - Prepare hardware environment with GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below£º
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## [Script description](#contents)

### [Script and sample code](#contents)

```bash
CFDT
├── src
│   ├── config
│   │   ├── __init__.py                         # init file
│   │   └── config.py                           # parse arguments
│   ├── dataset
│   │   ├── __init__.py                         # init file
│   │   ├── coco_eval.py                        # coco evaluator for validate mindspore model
│   │   ├── dataset.py                          # coco dataset
│   │   └── transforms.py                       # image augmentations
│   ├── model_utils
│   │   ├── __init__.py                         # init file
│   │   ├── box_ops.py                          # bounding box ops
│   │   └── misc.py                             # model misc
│   ├── __init__.py                             # init file
│   ├── backbone.py                             # backbone model
│   ├── cfdt.py                                 # cfdt model
│   ├── deformable_transformer.py               # deformable transformer model
│   └── ms_deform_attn.py                       # multiscale deformable attention model
├── README.md                                   # readme
├── default_config.yaml                         # config file
├── eval.py                                     # evaluate mindspore model
```

## [Eval process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash
# infer example
  # python
  GPU: python eval.py --dataset_path path/to/dataset --checkpoint_path path/to/ckpt
```

> checkpoint can be downloaded at https://download.mindspore.cn/model_zoo/research/cv/CFDT/

### Result

```bash
Results of CFDT with PyramidTNT-Tiny(P-Tiny) backbone:
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.819
```

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).