- [Contents](#contents)
- [Dynamic Quantization Description](#dynamic-quantization-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
- [Eval Process](#eval-process)
    - [Usage](#usage)
    - [Launch](#launch)
    - [Result](#result)
- [ModelZoo Homepage](#modelzoo-homepage)

# [Dynamic Quantization Description](#contents)

To conduct the low-bit quantization for each image individually, we develop a dynamic quantization scheme for exploring their optimal bit-widths. Experimental results show that our method can be easily embedded with mainstream quantization frameworks and boost their performance.

[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Instance-Aware_Dynamic_Neural_Network_Quantization_CVPR_2022_paper.pdf)：Zhenhua Liu, Yunhe Wang, Kai Han, Siwei Ma and Wen Gao. "Instance-Aware Dynamic Neural Network Quantization", CVPR 2022.

# [Model Architecture](#contents)

A bit-controller is employed to generate the bit-width of each layer for different samples and the bit-controller is jointly optimized with the main network. You can find the details in the [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Instance-Aware_Dynamic_Neural_Network_Quantization_CVPR_2022_paper.pdf).

# [Dataset](#contents)

Dataset used: [ImageNet2012](http://www.image-net.org/)

- Dataset size 224\*224 colorful images in 1000 classes
    - Train: 1,281,167 images
    - Test: 50,000 images
- Data format: jpeg
    - Note: Data will be processed in dataset.py

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend/GPU/CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Script description](#contents)

```text
DynamicQuant
├── src
    └── dataset.py # dataset loader
    └── gumbelsoftmax.py # implementation of gumbel softmax
    └── quant.py # dynamic quantization
    └── resnet.py # resnet network
├── eval.py # inference entry
├── readme.md # Readme
```

# [Eval Process](#contents)

## [Usage](#contents)

After installing MindSpore via the official website, you can start evaluation as follows:

## [Launch](#contents)

  ```python
  python eval.py --dataset_path [DATASET]
  ```

## [Result](#contents)

  ```python
  result: {'acc': 0.6901} ckpt= ./resnet18_dq.ckpt
  ```

 Checkpoint can be downloaded at [https://download.mindspore.cn/model_zoo/research/cv/DynamicQuant/](https://download.mindspore.cn/model_zoo/research/cv/DynamicQuant/).

# [ModelZoo Homepage](#contents)  

 Please check the official [homepage](https://gitee.com/mindspore/models).  
