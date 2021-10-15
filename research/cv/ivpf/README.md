# iVPF: Numerical Invertible Volume Preserving Flow for Efficient Lossless Compression

Shifeng Zhang, Chen Zhang, Ning Kang, Zhenguo Li. iVPF: Numerical Invertible Volume Preserving Flow for Efficient Lossless Compression. In CVPR 2021.

The [MindSpore](https://www.mindspore.cn/) implementation of iVPF.

## Dependencies

Python 3.7.5 (Highly recommend version `3.7.5`, or unexpected error of Mindspore installation may occur)

Mindspore GPU 1.5 (GPU CUDA 10.1/11.1)

The installation of Mindspore can be found [here](https://www.mindspore.cn/install). The Mindspore version MUST be the newest, or unexperted error on running the code may occur.

## Usage

Download model weights [here](), and put it in folder `model_weights`.

Inference CIFAR-10 with iVPF

`python eval_coding_cifar10.py --data_dir [CIFAR10_DATA_DIR] --no_code`

Encoding and Decoding CIFAR-10 with iVPF

`python eval_coding_cifar10.py --data_dir [CIFAR10_DATA_DIR]`

## Reference

`
@inproceedings{zhang2021ivpf,
  title={iVPF: Numerical Invertible Volume Preserving Flow for Efficient Lossless Compression},
  author={Zhang, Shifeng and Zhang, Chen and Kang, Ning and Li, Zhenguo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={620--629},
  year={2021}
}
`