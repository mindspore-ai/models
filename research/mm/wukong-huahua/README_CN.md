## 目录

[查看英文](./README.md)

- [目录](#目录)
- [Wukong-Huahua](#wukong-huahua模型)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
    - [准备checkpoint](#准备checkpoint)
    - [准备分词器相关文件](#准备分词器相关文件)
    - [文图生成](#文图生成)
    - [生成样例](#生成样例)

## Wukong-Huahua模型

Wukong-Huahua模型是基于[Wukong数据集](https://wukong-dataset.github.io/wukong-dataset/)训练得到的一个文图生成模型。

## 环境要求

- 硬件
    - 准备Ascend处理器搭建硬件环境
- 框架
    - [Mindspore 1.9+](https://www.mindspore.cn/ "Mindspore")
    - 其他Python包需求请参考wukong-huahua/requirements.txt
- 如需查看详情，请参考如下资源
    - [Mindspore 教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [Mindspore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速开始

### 准备checkpoint

下载Wukong-Huahua预训练参数 [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) 至 wukong-huahua/models/ 目录.

### 准备分词器相关文件

下载[vocab_zh.txt](https://drive.google.com/file/d/1jmbTqpnef3czYWMK2QXYm_i79FpV1bxl/view?usp=sharing)并放到wukong-huahua/ldm/models/clip_zh/目录下。

### 文图生成

要进行文图生成，可以运行txt2img.py 或者直接使用默认参数运行 infer.sh.

```shell
python txt2img.py --prompt [input text] --ckpt_path [ckpt_path] --H [image_height] --W [image_width] --outdir [image save folder] --n_samples [number of images to generate] --plms --skip_grid
```

```shell
bash infer.sh
```

更高的分辨率需要更大的显存. 对于 Ascend 910 卡, 我们可以同时生成2张1024x768的图片或者16张512x512的图片。

### 生成样例

下面是我们的Wukong-Huahua模型生成的一些样例。

![城市夜景 赛博朋克 格雷格·鲁特科夫斯基](demo/1.png)
![来自深渊 风景 绘画 写实风格](demo/2.png)
![莫奈 撑阳伞的女人 月亮 梦幻](demo/3.png)
![海上日出时候的奔跑者](demo/4.png)
![诺亚方舟在世界末日起航 科幻插画](demo/5.png)
![时空 黑洞 辐射](demo/6.png)
![乡村 田野 屏保](demo/7.png)