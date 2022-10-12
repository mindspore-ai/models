# Mesh Movement Networks for PDE Solvers

该项目旨在实现[M2N: Mesh Movement Networks for PDE Solvers](https://arxiv.org/abs/2204.11188)的[MindSpore](https://www.mindspore.cn/)版本。M2N是据我们所知的第一个基于学习的端到端的网格移动算法框架，可以在达到与传统sota方法（Monge–Ampere）的PDE数值误差相当水平的情况下，实现网格优化过程3-4个数量级的加速。

我们的原始代码是用PyTorch实现的。由于通知比较紧急，我们计划在2022年底之前实现并开源MindSpore版本。

## 引用

```latex
@article{song2022m2n,
  title={M2N: Mesh Movement Networks for PDE Solvers},
  author={Song, Wenbin and Zhang, Mingrui and Wallwork, Joseph G and Gao, Junpeng and Tian, Zheng and Sun, Fanglei and Piggott, Matthew D and Chen, Junqing and Shi, Zuoqiang and Chen, Xiang and others},
  journal={arXiv preprint arXiv:2204.11188},
  year={2022}
}
```
