# Contents

- [Contents](#contents)
- [Model Description](#model-description)
    - [Paper](#paper)
    - [Abstract](#abstract)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
- [Performance](#performance)
    - [Platform](#platform)
    - [Results](#results)
- [Contributors](#contributors)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [ModeZoo Homepage](#modezoo-homepage)

# [Model Description](#model-description)

## [Paper](#paper)

**[CVPR 2023] Learning Steerable Function for Efficient Image Resampling**

[Jiacheng Li*](https://ddlee-cn.github.io), Chang Chen*, Wei Huang, Zhiqiang Lang, Fenglong Song, Youliang Yan, and [Zhiwei Xiong#](http://staff.ustc.edu.cn/~zwxiong)

(*Equal contribution, #Corresponding author)

[Project Page](https://lerf.pages.dev)

This is the official MindSpore evaluation code of LeRF.

## [Abstract](#abstract)

Image resampling is a basic technique that is widely employed in daily applications. Existing deep neural networks (DNNs) have made impressive progress in resampling performance. Yet these methods are still not the perfect substitute for interpolation, due to the issues of efficiency and continuous resampling. In this work, we propose a novel method of Learning Resampling Function (termed LeRF), which takes advantage of both the structural priors learned by DNNs and the locally continuous assumption of interpolation methods. Specifically, LeRF assigns spatially-varying steerable resampling functions to input image pixels and learns to predict the hyper-parameters that determine the orientations of these resampling functions with a neural network. To achieve highly efficient inference, we adopt look-up tables (LUTs) to accelerate the inference of the learned neural network. Furthermore, we design a directional ensemble strategy and edge-sensitive indexing patterns to better capture local structures. Extensive experiments show that our method runs as fast as interpolation, generalizes well to arbitrary transformations, and outperforms interpolation significantly, e.g., up to 3dB PSNR gain over bicubic for ×2 upsampling on Manga109.

# [Dataset](#dataset)

Please download the SR benchmark datasets following the instruction of [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets).

Then, put the downloaded SR benchmark datasets here as the following structure. `[testset]` can be `['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']`.

```bash
datasets/Benchmark/
                   /[testset]/HR/*.png
                             /LR_bicubic/X2/*.png
                   /...
```

LR images with non-integer downscale factors can be obtained with the `imresize` function in MATLAB or [ResizeRight](https://github.com/assafshocher/ResizeRight).

# [Environment Requirements](#environment-requirements)

- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

- Dependencies

  ```bash
  pillow>=8.3.1
  opencv-python>=3.4.1.15
  scipy>=1.1.0
  numpy>=1.19.5
  mindspore>=1.9.0
  ```

# [Quick Start](#quick-start)

Download the LUT files and the model checkpoint from our [project page](https://lerf.pages.dev), and put them under `models/lerf-lut` and `models/lerf-net`, respectively.

Evaluation with the following script.

```bash
# lerf-lut
python lerf_sr/eval_lerf_lut.py --expDir ./models/lerf-lut \
                                --testDir ./datasets/Benchmark \
                                --resultRoot ./results \
                                --lutName LUT_ft

# lerf-net
python lerf_sr/eval_lerf_net.py --expDir ./models/lerf-net \
                                --testDir ./datasets/Benchmark \
                                --resultRoot ./results
```

# [Script Description](#script-description)

## [Script and Sample Code](#script-and-sample-code)

```bash
└──lerf
  ├── README.md
  ├── common
    ├── network.py                         # Basic blocks
    ├── option.py                          # Options
    ├── resize2d.py                        # Resize function
    └── utils.py                           # Utility functions
  ├── lerf-sr
    ├── model.py                           # LeRFNet model definition
    ├── eval_lerf_lut.py                   # Evaluation of LeRF-LUT
    └── eval_lerf_net.py                   # Evaluation of LeRFNet
  ├── models
    ├── lerf-lut
      ├── LUT_ft_4bit_int8_*.npy           # LUTs
    ├── lerf-net
      ├── Model_050000.ckpt                # Model checkpoint
  ├── eval.sh                              # Evaluation script
  └── requirements.txt                     # Dependencies
```

# [Performance](#performance)

## [Platform](#platform)

| Parameters          | GPU                         |
| ------------------- | --------------------------- |
| Resource            | CUDA 11.0                   |
| Uploaded Date       | 04/20/2023                  |
| MindSpore Version   | 1.9.0                       |

## [Results](#results)

The reference results on Set5 Super-Resolution benchmark are listed below.

| PSNR/SSIM     | 2.0x2.0         | 3.0x3.0         | 4.0x4.0         |
| ------------------- | ----------------| ----------------| ----------------|
| LeRF-LUT            | 35.71/0.9474    | 32.02/0.8980    | 30.15/0.8548    |
| LeRFNet             | 36.03/0.9517    | 32.17/0.9035    | 30.26/0.8608    |

# [Contributors](#contributor)

[ddlee](https://ddlee-cn.github.io) (jclee@mail.ustc.edu.cn)

# [Citation](#citation)

If you find our work helpful, please cite the following papers.

```bibtex
@InProceedings{Li_2022_MuLUT,
      author    = {Li, Jiacheng and Chen, Chang and Cheng, Zhen and Xiong, Zhiwei},
      title     = {{MuLUT}: Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution},
      booktitle = {ECCV},
      year      = {2022},
  }
@arxiv{Li_2023_DNN_LUT,
      author    = {Li, Jiacheng and Chen, Chang and Cheng, Zhen and Xiong, Zhiwei},
      title     = {Toward {DNN} of {LUTs}: Learning Efficient Image Restoration with Multiple Look-Up Tables},
      booktitle = {arxiv},
      year      = {2023},
  }
@InProceedings{Li_2023_LeRF,
      author    = {Li, Jiacheng and Chen, Chang and Huang, Wei and Lang, Zhiqiang and Song, Fenglong and Yan, Youliang and Xiong, Zhiwei},
      title     = {Learning Steerable Function for Efficient Image Resampling},
      booktitle = {CVPR},
      year      = {2023},
  }
```

# [Acknowledgement](#acknowledgement)

The resize part of our code is modified based on [ResizeRight](https://github.com/assafshocher/ResizeRight).

# [ModeZoo Homepage](#modelzoo)

Please check the official [homepage](https://gitee.com/mindspore/models).
