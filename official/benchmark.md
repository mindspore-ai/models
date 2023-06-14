## Computer Vision

Currently all results are tested on Ascend 910, RTX 3090 version is coming soon

### Image Classification

Accuracies are reported on ImageNet-1K

| model | acc@1 | bs | cards | ms/step | amp | config
:-: | :-: | :-: | :-: | :-: | :-: | :-: |
| vgg11| 71.86 | 32 | 8 |  61.63  |  O2 |  [mindcv_vgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg) |
| vgg13| 72.87 | 32 | 8 |  66.47  |  O2 |   [mindcv_vgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg) |
| vgg16| 74.61 | 32 | 8 |  73.68  |  O2 |   [mindcv_vgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg) |
| vgg19| 75.21 | 32 | 8 |  81.13  |  O2 |   [mindcv_vgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg) |
| resnet18| 70.21 | 32 | 8 |  23.98  |  O2 |   [mindcv_resnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet) |
| resnet34| 74.15 | 32 | 8 |  23.98  |  O2 |   [mindcv_resnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet) |
| resnet50| 76.69 | 32 | 8 |  31.97  |  O2 |   [mindcv_resnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet) |
| resnet101| 78.24 | 32 | 8 | 50.76   |  O2 |   [mindcv_resnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet) |
| resnet152| 78.72 | 32 | 8 |  70.94  |  O2 |   [mindcv_resnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet) |
| resnetv2_50| 76.90 | 32 | 8 | 35.72   |  O2 |   [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2) |
| resnetv2_101| 78.48 | 32 | 8 |  56.02  |  O2 |   [mindcv_resnetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2) |
| dpn92  | 79.46 | 32 | 8 | 79.89   |  O2 |   [mindcv_dpn](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn) |
| dpn98  | 79.94 | 32 | 8 | 106.60  |  O2 |   [mindcv_dpn](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn) |
| dpn107 | 80.05 | 32 | 8 | 107.60  |  O2 |   [mindcv_dpn](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn) |
| dpn131 | 80.07 | 32 | 8 | 143.57  |  O2 |   [mindcv_dpn](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn) |
| densenet121  | 75.64 | 32 | 8 | 48.07   |  O2 |   [mindcv_densenet](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet) |
| densenet161  | 79.09 | 32 | 8 | 115.11  |  O2 |   [mindcv_densenet](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet) |
| densenet169 | 77.26 | 32 | 8 | 73.14  |  O2 |   [mindcv_densenet](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet) |
| densenet201 | 78.14 | 32 | 8 | 96.12  |  O2 |   [mindcv_densenet](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet) |
| seresnet18 | 71.81 | 64 | 8 | 50.39  |  O2 |   [mindcv_senet](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet) |
| seresnet34 | 75.36 | 64 | 8 | 50.54 |  O2 |   [mindcv_senet](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet) |
| seresnet50 | 78.31 | 64 | 8 | 98.37  |  O2 |   [mindcv_senet](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet) |
| seresnext26 | 77.18 | 64 | 8 | 73.72  |  O2 |   [mindcv_senet](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet) |
| seresnext50 | 78.71 | 64 | 8 | 113.82  |  O2 |   [mindcv_senet](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet) |
| skresnet18 | 73.09 | 64 | 8 | 65.95  |  O2 |   [mindcv_sknet](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet) |
| skresnet34 | 76.71 | 32 | 8 | 43.96  |  O2 |   [mindcv_sknet](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet) |
| skresnet50_32x4d | 79.08 | 64 | 8 | 65.95  |  O2 |   [mindcv_sknet](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet) |
| resnext50_32x4d | 78.53 | 32 | 8 | 50.25  |  O2 |   [mindcv_resnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext) |
| resnext101_32x4d | 79.83 | 32 | 8 | 68.85  |  O2 |   [mindcv_resnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext) |
| resnext101_64x4d | 80.30 | 32 | 8 | 112.48  |  O2 |   [mindcv_resnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext) |
| resnext152_64x4d | 80.52 | 32 | 8 | 157.06  |  O2 |   [mindcv_resnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext) |
| rexnet_x09 | 77.07 | 64 | 8 | 145.08 |  O2 |   [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| rexnet_x10 | 77.38 | 64 | 8 | 156.67 |  O2 |   [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| rexnet_x13 | 79.06 | 64 | 8 | 203.04 |  O2 |   [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| rexnet_x15 | 79.94 | 64 | 8 | 231.41 |  O2 |   [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| rexnet_x20 | 80.64 | 64 | 8 | 308.15 |  O2 |   [mindcv_rexnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| resnest50 | 80.81 | 128 | 8 | 376.18 |  O2 |   [mindcv_resnest](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest) |
| resnest101 | 82.50 | 128 | 8 | 719.84 |  O2 |   [mindcv_resnest](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest) |
| res2net50 | 79.35 | 32 | 8 | 49.16 |  O2 |   [mindcv_res2net](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net) |
| res2net101 | 79.56 | 32 | 8 | 49.96 |  O2 |   [mindcv_res2net](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net) |
| res2net50_v1b | 80.32 | 32 | 8 | 93.33 |  O2 |   [mindcv_res2net](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net) |
| res2net101_v1b | 95.41 | 32 | 8 | 86.93 |  O2 |   [mindcv_res2net](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net) |
| googlenet | 72.68 | 32 | 8 | 23.26 |  O0 |   [mindcv_googlenet](https://github.com/mindspore-lab/mindcv/tree/main/configs/googlenet) |
| inceptionv3 | 79.11 | 32 | 8 | 49.96 |  O0 |   [mindcv_inceptionv3](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv3) |
| inceptionv4 | 80.88 | 32 | 8 | 93.33 |  O0 |   [mindcv_inceptionv4](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv4) |
| mobilenet_v1_025 | 53.87 | 64 | 8 | 75.93 |  O2 |   [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1) |
| mobilenet_v1_050 | 65.94 | 64 | 8 | 51.96 |  O2 |   [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1) |
| mobilenet_v1_075 | 70.44 | 64 | 8 | 57.55 |  O2 |   [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1) |
| mobilenet_v1_100 | 72.95 | 64 | 8 | 44.04 |  O2 |   [mindcv_mobilenetv1](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1) |
| mobilenet_v2_075 | 69.98 | 256 | 8 | 169.81 |  O3 |   [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2) |
| mobilenet_v2_100 | 72.27 | 256 | 8 | 195.06 |  O3 |   [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2) |
| mobilenet_v2_140 | 75.56 | 256 | 8 | 230.06 |  O3 |   [mindcv_mobilenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2) |
| mobilenet_v3_small | 68.10 | 75 | 8 | 67.19 |  O3 |   [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3) |
| mobilenet_v3_large | 75.23 | 75 | 8 | 85.61 |  O3 |   [mindcv_mobilenetv3](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3) |
| shufflenet_v1_g3_x0_5 | 57.05 | 64 | 8 | 142.69 |  O0 |   [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1) |
| shufflenet_v1_g3_x1_5 | 67.77 | 64 | 8 | 267.79 |  O0 |   [mindcv_shufflenetv1](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1) |
| shufflenet_v2_x0_5 | 57.05 | 64 | 8 | 142.69 |  O0 |   [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2) |
| shufflenet_v2_x1_0 | 67.77 | 64 | 8 | 267.79 |  O0 |   [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2) |
| shufflenet_v2_x1_5 | 57.05 | 64 | 8 | 142.69 |  O0 |   [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2) |
| shufflenet_v2_x2_0 | 67.77 | 64 | 8 | 267.79 |  O0 |   [mindcv_shufflenetv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2) |
| xception | 79.01 | 32 | 8 | 98.03 |  O2 |   [mindcv_xception](https://github.com/mindspore-lab/mindcv/tree/main/configs/xception) |
| ghostnet_50 | 66.03 | 128 | 8 | 220.88 |  O3 |   [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet) |
| ghostnet_100 | 73.78 | 128 | 8 | 222.67 |  O3 |   [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet) |
| ghostnet_130 | 75.50 | 128 | 8 | 223.11 |  O3 |   [mindcv_ghostnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet) |
| nasnet_a_4x1056 | 73.65 | 256 | 8 | 1562.35 |  O0 |   [mindcv_nasnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/nasnet) |
| mnasnet_0.5 | 68.07 | 512 | 8 | 367.05 |  O3 |   [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet) |
| mnasnet_0.75 | 71.81 | 256 | 8 | 151.02 |  O0 |   [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet) |
| mnasnet_1.0 | 74.28 | 256 | 8 | 153.52 |  O0 |   [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet) |
| mnasnet_1.4 | 76.01 | 256 | 8 | 194.90 |  O0 |   [mindcv_mnasnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet) |
| efficientnet_b0 | 76.89 | 128 | 8 | 276.77 |  O2 |   [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet) |
| efficientnet_b1 | 78.95 | 128 | 8 | 435.90 |  O2 |   [mindcv_efficientnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet) |
| regnet_x_200mf| 68.74 | 64 | 8 | 47.56 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_x_400mf| 73.16 | 64 | 8 | 47.56 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_x_600mf| 73.34 | 64 | 8 | 48.36 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_x_800mf| 76.04 | 64 | 8 | 47.56 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_y_200mf| 70.30 | 64 | 8 | 58.35 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_y_400mf| 73.91 | 64 | 8 | 77.94 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_y_600mf| 75.69 | 64 | 8 | 79.94 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| regnet_y_800mf| 76.52 | 64 | 8 | 81.93 |  O2 |   [mindcv_regnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet) |
| mixnet_s | 75.52 | 128 | 8 | 340.18 |  O3 |   [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet) |
| mixnet_m | 76.64 | 128 | 8 | 384.68 |  O3 |   [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet) |
| mixnet_l | 78.73 | 128 | 8 | 389.97 |  O3 |   [mindcv_mixnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet) |
| hrnet_w32 | 80.64 | 128 | 8 | 335.73 |  O2 |   [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet) |
| hrnet_w48 | 81.19 | 128 | 8 | 463.63 |  O2 |   [mindcv_hrnet](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet) |
| bit_resnet50 | 76.81 | 32 | 8 | 130.60 |  O0 |   [mindcv_bit](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit) |
| bit_resnet50x3 | 80.63 | 32 | 8 | 533.09 |  O0 |   [mindcv_bit](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit) |
| bit_resnet101 | 77.93| 16 | 8 | 128.15 |  O0 |   [mindcv_bit](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit) |
| repvgg_a0 | 72.19 | 32 | 8 | 27.63 |  O0 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_a1 | 74.19 | 32 | 8 | 27.45 |  O0 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_a2 | 76.63 | 32 | 8 | 39.79 |  O0 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b0 | 74.99 | 32 | 8 | 33.05 |  O0 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b1 | 78.81 | 32 | 8 | 68.88 |  O0 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b2 | 79.29 | 32 | 8 | 106.90 |  O0 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b3 | 80.46 | 32 | 8 | 137.24 |  O0|   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b1g2 | 78.03 | 32 | 8 | 59.71 |  O2 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b1g4 | 77.64 | 32 | 8 | 65.83 |  O2 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repvgg_b2g4 | 78.80 | 32 | 8 | 89.57 |  O2 |   [mindcv_repvgg](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg) |
| repmlp_t224 | 76.71 | 128 | 8 | 973.88 |  O2 |   [mindcv_repmlp](https://github.com/mindspore-lab/mindcv/tree/main/configs/repmlp) |
| convnext_tiny | 81.91 | 128 | 8 | 343.21 |  O2 |   [mindcv_convnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext) |
| convnext_small | 83.40 | 128 | 8 | 405.96 |  O2 |   [mindcv_convnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext) |
| convnext_base | 83.32 | 128 | 8 | 531.10 |  O2 |   [mindcv_convnext](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext) |
| vit_b_32_224 | 75.86 | 256 | 8 | 623.09 |  O2 |   [mindcv_vit](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit) |
| vit_l_16_224 | 76.34| 48 | 8 | 613.98 |  O2 |   [mindcv_vit](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit) |
| vit_l_32_224 | 73.71 | 128 | 8 | 527.58 |  O2 |   [mindcv_vit](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit) |
| swintransformer_tiny | 80.82 | 256 | 8 | 1765.65 |  O2 |   [mindcv_swintransformer](https://github.com/mindspore-lab/mindcv/tree/main/configs/swintransformer) |
| pvt_tiny | 74.81 | 128 | 8 | 310.74 |  O2 |   [mindcv_pvt](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt) |
| pvt_small | 79.66 | 128 | 8 | 431.15 |  O2 |   [mindcv_pvt](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt) |
| pvt_medium | 81.82 | 128 | 8 | 613.08 |  O2 |   [mindcv_pvt](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt) |
| pvt_large | 81.75 | 128 | 8 | 860.41 |  O2 |   [mindcv_pvt](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt) |
| pvt_v2_b0 | 71.50 | 128 | 8 | 338.78 |  O2 |   [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2) |
| pvt_v2_b1 | 78.91 | 128 | 8 | 337.94 |  O2 |   [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2) |
| pvt_v2_b2 | 81.99 | 128 | 8 | 503.79 |  O2 |   [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2) |
| pvt_v2_b3 | 82.84 | 128 | 8 | 738.90 |  O2 |   [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2) |
| pvt_v2_b4 | 83.14 | 128 | 8 | 1030.06 |  O2 |   [mindcv_pvtv2](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2) |
| pit_ti | 72.96 | 128 | 8 | 339.44 |  O2 |   [mindcv_pit](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit) |
| pit_xs | 78.41 | 128 | 8 | 338.70 |  O2 |   [mindcv_pit](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit) |
| pit_s | 80.56 | 128 | 8 | 336.08 |  O2 |   [mindcv_pit](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit) |
| pit_b | 81.87 | 128 | 8 | 350.33 |  O2 |  [mindcv_pit](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit) |
| coat_lite_tiny | 77.35 | 64 | 8 | 258.07 |  O2 |   [mindcv_coat](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat) |
| coat_lite_mini | 78.51 | 64 | 8 | 265.44 |  O2 |   [mindcv_coat](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat) |
| coat_tiny | 79.67 | 64 | 8 | 580.54 |  O2 |   [mindcv_coat](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat) |
| convit_tiny | 73.66 | 256 | 8 | 388.80 |  O2 |   [mindcv_convit](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit) |
| convit_tiny_plus | 77.00 | 256 | 8 | 393.60 |  O2 |   [mindcv_convit](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit) |
| convit_small | 81.63 | 192 | 8 | 588.73 |  O2 |   [mindcv_convit](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit) |
| convit_small_plus | 81.80 | 192 | 8 | 665.74 |  O2 |   [mindcv_convit](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit) |
| convit_base | 82.10 | 128 | 8 | 701.84 |  O2 |   [mindcv_convit](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit) |
| convit_base_plus | 81.96 | 128 | 8 | 983.21 |  O2 |   [mindcv_convit](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit) |
| crossvit_9 | 73.56 | 256 | 8 | 685.25 |  O3 |   [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit) |
| crossvit_15 | 81.08 | 256 | 8 | 1086.00 |  O3 |   [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit) |
| crossvit_18 | 81.93 | 256 | 8 | 1137.60 |  O3 |   [mindcv_crossvit](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit) |
| mobilevit_xx_small | 68.90 | uploading  | uploading  |uploading  |  O3 |   [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit) |
| mobilevit_x_small | 74.98 | uploading |  uploading | uploading |  O3 |   [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit) |
| mobilevit_small | 78.48 | uploading | uploading  | uploading  |  O2 |   [mindcv_mobilevit](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit) |
| visformer_tiny | 78.28 | 128 | 8 | 393.29 |  O3 |   [mindcv_visformer](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer) |
| visformer_tiny_v2 | 78.82 | 256 | 8 | 627.20 |  O3 |   [mindcv_visformer](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer) |
| visformer_small | 81.76 | 64 | 8 | 155.88 |  O3 |   [mindcv_visformer](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer) |
| visformer_small_v2 | 82.17 | 64 | 8 | 158.27 |  O3 |   [mindcv_visformer](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer) |
| edgenext_xx_small | 71.02 | 256 | 8 | 1207.78 |  O2 |   [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext) |
| edgenext_x_small | 75.14 | 256 | 8 | 1961.42 |  O3 |   [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext) |
| edgenext_small | 79.15 | 256 | 8 | 882.00 |  O3 |   [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext) |
| edgenext_base | 82.24 | 256 | 8 | 1151.98 |  O2 |   [mindcv_edgenext](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext) |
| poolformer_s12 | 77.33 | 128 | 8 | 316.77 |  O3 |   [mindcv_poolformer](https://github.com/mindspore-lab/mindcv/tree/main/configs/poolformer) |
| xcit_tiny_12_p16 | 77.67 | 128 | 8 | 352.30 |  O2 |   [mindcv_xcit](https://github.com/mindspore-lab/mindcv/tree/main/configs/xcit) |
| volo_d1 | 81.82 | 128 | 8 | 575.54 |  O3 |   uploading |
| cait_s24 | 82.25 | 64 | 8 | 435.54 |  O2 |   uploading |

### Object Detection

Accuracies are reported on COCO2017

| model | map | bs | cards | ms/step | amp | config
:-: | :-: | :-: | :-: | :-: | :-: | :-: |
| yolov8_n | 37.2 | 16 | 8 |  302  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_s | 44.6 | 16 | 8 |  uploading  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_m | 50.5 | 16 | 8 |  454  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_l | 52.8 | 16 | 8 |  536  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_x | 53.7 | 16 | 8 |  636  |  O0 |   [mindyolo_yolov8](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov7_t| 37.5 | 16 | 8 |  594.91  |  O0 |   [mindyolo_yolov7](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov7_l| 50.8 | 16 | 8 |  905.26  |  O0 |   [mindyolo_yolov7](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov7_x|  52.4| 16 | 8 |  819.36  |  O0 |   [mindyolo_yolov7](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov5_n | 27.3 | 32 | 8 |  504.68  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_s | 37.6 | 32 | 8 |  535.32  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_m | 44.9 | 32 | 8 |  646.75  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_l | 48.5 | 32 | 8 |  684.81  |  O3 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_x | 50.5 | 16 | 8 |  613.81  |  O0 |   [mindyolo_yolov5](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov4_csp | 45.4 | 16 | 8 |  709.70  |  O0 |   [mindyolo_yolov4](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |
| yolov4_csp(silu) | 45.8 | 16 | 8 |  594.97  |  O0 |   [mindyolo_yolov4](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |
| yolov3_darknet53| 45.5 | 16 | 8 |  481.37  |  O0 |   [mindyolo_yolov3](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov3) |
| yolox_n | 24.1 | 8 | 8 |  201  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_t | 33.3 | 8 | 8 |  190  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_s | 40.7 | 8 | 8 |  270  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_m | 46.7 | 8 | 8 |  311  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_l | 49.2 | 8 | 8 |  535  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_x | 51.6 | 8 | 8 |  619  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_darknet53 | 47.7 | 8 | 8 |  411  |  O0 |   [mindyolo_yolox](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |

### OCR

| model | dataset |fscore | bs | cards | fps | amp | config |
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| dbnet_mobilenetv3 | icdar2015 | 77.28 | 10 | 1 |  100  |  O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| dbnet_resnet18 | icdar2015 | 83.71 | 20 | 1 |  107  |  O0 |  [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| dbnet_resnet50 | icdar2015 | 84.99 | 10 | 8 |  75.19  |  O0 | [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| dbnet_resnet50 | msra-td500 | 85.03 | 20 | 1 |  52.63 |  O0 |  [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| dbnet++_resnet50 | icdar2015 | 86.60 | 32 | 1 |  56.05  |  O0 |  [mindocr_dbnet](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet) |
| crnn_vgg7 | IC03,13,15,IIT,etc | 82.03 | 16 | 8 |  5802.71  |  O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn) |
| crnn_resnet34_vd | IC03,13,15,IIT,etc | 84.45 | 64 | 8 |  6694.84  |  O3 |  [mindocr_crnn](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn) |
| rare_resnet34_vd | IC03,13,15,IIT,etc | 85.19 | 512 | 4 |  4561.10  |  O2 |  [mindocr_rare](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare) |
