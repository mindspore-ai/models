
# Content

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requiments](#requirements)
- [Quick Start](#quick-start)
- [ModelZoo Homepage](#modelzoo-homepage)

## Introduction

<img src="assets/teaser.gif" width="100%">

*(This testing video is from the 3DPW testset, and processed frame by frame without temporal smoothing.)*

This repo contains the CLIFF demo code (Implemented in MindSpore) for the following paper.

> CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation. \
> Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan ⋆ \
> ECCV 2022 Oral

<img src="assets/arch.png" width="100%">

## Dataset

Not relevant.

## Requirements

```bash
conda create -n cliff python=3.9
pip install -r requirements.txt
```

Download the pretrained checkpoints and the testing sample to run the demo.
[[Baidu Pan](https://pan.baidu.com/s/15v0jnoyEpKIXWhh2AjAZeQ?pwd=7777)]
[[Google Drive](https://drive.google.com/drive/folders/1_d12Q8Yj13TEvB_4vopAbMdwJ1-KVR0R?usp=sharing)]

Finally put these data following the directory structure as below:

```text
${ROOT}
|-- ckpt
    |-- cliff-hr48-PA43.0_MJE69.0_MVE81.2_3dpw.ckpt
    |-- cliff-res50-PA45.7_MJE72.0_MVE85.3_3dpw.ckpt
|-- data
    |-- data/im07937.png
    |-- data/smpl_mean_params.npz
```

## Quick Start

```bash
python demo.py --input_path PATH --ckpt CKPT
```

<img src="assets/im08036/im08036.png" width="24%">
<img src="assets/im08036/im08036_bbox.jpg" width="24%">
<img src="assets/im08036/im08036_front_view_cliff_hr48.jpg" width="24%">
<img src="assets/im08036/im08036_side_view_cliff_hr48.jpg" width="24%">

<img src="assets/im00492/im00492.png" width="24%">
<img src="assets/im00492/im00492_bbox.jpg" width="24%">
<img src="assets/im00492/im00492_front_view_cliff_hr48.jpg" width="24%">
<img src="assets/im00492/im00492_side_view_cliff_hr48.jpg" width="24%">

One can change the demo options in the script. Please see the option description in the bottom lines of `demo.py`.

## ModelZoo Homepage

Please check the official [homepage](https://gitee.com/mindspore/models).