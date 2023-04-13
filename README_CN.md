# ![MindSpore Logo](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png)

## 欢迎来到MindSpore ModelZoo

MindSpore models仓中提供了不同任务领域，经典的SOTA模型实现和端到端解决方案。目的是方便MindSpore用户更加方便的利用MindSpore进行研究和产品开发。

为了让开发者更好地体验MindSpore框架优势，我们将陆续增加更多的典型网络和相关预训练模型。如果您对ModelZoo有任何需求，请通过[Gitee](https://gitee.com/mindspore/mindspore/issues)或[MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html)与我们联系，我们将及时处理。

| 目录                     | 描述                                                         |
|------------------------| ------------------------------------------------------------ |
| [official](official)   | • 官方维护，随MindSpore版本迭代更新，保证版本出口网络的精度效果<br/>• 推荐写法，使用最新的MindSpore接口和推荐使用的特性，在保证代码可读性的基础上，有更快的性能表现<br/>• 有详细的网络信息和说明文档，包含但不限于模型说明，数据集使用，规格支持，精度性能数据，网络checkpoint文件，MindIR文件等 |
| [research](research)   | • 历史支持，测试验收通过的模型，在README里标明支持的MindSpore版本<br/>• 按需维护，内容不会随版本迭代更新，只会适配对应的接口变更，由MindSpore开发人员进行维护支持，按需进行维护升级<br/>• 提供较为详细的网络信息和说明文档，包含但不限于模型说明，数据集使用，规格支持，精度数据，网络checkpoint文件，MindIR文件等 |
| [community](community) | • 生态开发者贡献模型，按需进行维护升级，在README里说明支持的MindSpore版本<br/>• 不强制提供模型文件 |

- 使用最新MindSpore API的SOTA模型

- MindSpore优势

- 官方维护和支持

## 目录

### 官方网络

|  领域 | 子领域  | 网络   | Ascend | GPU | CPU |
|:------   |:------| :-----------  |:------:   |:------:  |:-----: |
| 语音 | 声纹识别 | [ecapa_tdnn](https://gitee.com/mindspore/models/tree/master/official/audio/EcapaTDNN) |✅|   |   |
| 语音 | 语音合成 | [lpcnet](https://gitee.com/mindspore/models/tree/master/official/audio/LPCNet) |✅| ✅ |   |
| 语音 | 语音合成 | [melgan](https://gitee.com/mindspore/models/tree/master/official/audio/MELGAN) |✅| ✅ |   |
| 语音 | 语音合成 | [tacotron2](https://gitee.com/mindspore/models/tree/master/official/audio/Tacotron2) |✅|   |   |
| 图神经网络 | 文本分类 | [bgcf](https://gitee.com/mindspore/models/tree/master/research/gnn/bgcf) |✅| ✅ |   |
| 图神经网络 | 文本分类 | [gat](https://gitee.com/mindspore/models/tree/master/research/gnn/gat) |✅| ✅ |   |
| 图神经网络 | 文本分类 | [gcn](https://gitee.com/mindspore/models/tree/master/official/gnn/GCN) |✅| ✅ |   |
| 推荐 | 推荐系统 | [naml](https://gitee.com/mindspore/models/tree/master/research/recommend/naml) |✅| ✅ |   |
| 推荐 | 推荐系统 | [ncf](https://gitee.com/mindspore/models/tree/master/research/recommend/ncf) |✅| ✅ |   |
| 推荐 | 推荐系统 | [tbnet](https://gitee.com/mindspore/models/tree/master/official/recommend/Tbnet) |✅| ✅ |   |
| 图像 | 图像分类 | [alexnet](https://gitee.com/mindspore/models/tree/master/research/cv/Alexnet) |✅| ✅ |   |
| 图像 | 图像去噪 | [brdnet](https://gitee.com/mindspore/models/tree/master/research/cv/brdnet) |✅|   |   |
| 图像 | 目标检测 | [centerface](https://gitee.com/mindspore/models/tree/master/research/cv/centerface) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [cnn_direction_model](https://gitee.com/mindspore/models/tree/master/research/cv/cnn_direction_model) |✅| ✅ |   |
| 图像 | 文本识别 | [cnnctc](https://gitee.com/mindspore/models/tree/master/research/cv/cnnctc) |✅| ✅ | ✅ |
| 图像 | 文本识别 | [crnn](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN) |✅| ✅ | ✅ |
| 图像 | 文本识别 | [crnn_seq2seq_ocr](https://gitee.com/mindspore/models/tree/master/research/cv/crnn_seq2seq_ocr) |✅|   |   |
| 图像 | 图像分类 | [cspdarknet53](https://gitee.com/mindspore/models/tree/master/research/cv/cspdarknet53) |✅|   |   |
| 图像 | 目标检测 | [ctpn](https://gitee.com/mindspore/models/tree/master/official/cv/CTPN) |✅| ✅ |   |
| 图像 | 目标检测 | [darknet53](https://gitee.com/mindspore/models/tree/master/research/cv/darknet53) | | ✅ |   |
| 图像 | 语义分割 | [deeplabv3](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P) |✅| ✅ | ✅ |
| 图像 | 文本检测 | [deeptext](https://gitee.com/mindspore/models/tree/master/official/cv/DeepText) |✅| ✅ |   |
| 图像 | 图像分类 | [densenet100](https://gitee.com/mindspore/models/tree/master/research/cv/densenet) |✅| ✅ |   |
| 图像 | 图像分类 | [densenet121](https://gitee.com/mindspore/models/tree/master/research/cv/densenet) |✅| ✅ |   |
| 图像 | 深度估计 | [depthnet](https://gitee.com/mindspore/models/tree/master/official/cv/DepthNet) |✅|   |   |
| 图像 | 图像去噪 | [dncnn](https://gitee.com/mindspore/models/tree/master/research/cv/dncnn) | | ✅ |   |
| 图像 | 图像分类 | [dpn](https://gitee.com/mindspore/models/tree/master/research/cv/dpn) |✅| ✅ |   |
| 图像 | 文本检测 | [east](https://gitee.com/mindspore/models/tree/master/research/cv/east) |✅| ✅ |   |
| 图像 | 图像分类 | [efficientnet](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet) | | ✅ | ✅ |
| 图像 | 图像分类 | [erfnet](https://gitee.com/mindspore/models/tree/master/research/cv/erfnet) |✅| ✅ |   |
| 图像 | 文本识别 | [essay-recogination](https://gitee.com/mindspore/models/tree/master/research/cv/essay-recogination) | | ✅ |   |
| 图像 | 目标检测 | [FasterRCNN_Inception_Resnetv2](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| 图像 | 目标检测 | [FasterRCNN_ResNetV1.5_50](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| 图像 | 目标检测 | [FasterRCNN_ResNetV1_101](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| 图像 | 目标检测 | [FasterRCNN_ResNetV1_152](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| 图像 | 目标检测 | [FasterRCNN_ResNetV1_50](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| 图像 | 语义分割 | [fastscnn](https://gitee.com/mindspore/models/tree/master/research/cv/fastscnn) |✅|   |   |
| 图像 | 语义分割 | [FCN8s](https://gitee.com/mindspore/models/tree/master/research/cv/FCN8s) |✅| ✅ |   |
| 图像 | 图像分类 | [googlenet](https://gitee.com/mindspore/models/tree/master/research/cv/googlenet) |✅| ✅ |   |
| 图像 | 图像分类 | [inceptionv3](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv3) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [inceptionv4](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4) |✅| ✅ | ✅ |
| 图像 | 图像去噪 | [LearningToSeeInTheDark](https://gitee.com/mindspore/models/tree/master/research/cv/LearningToSeeInTheDark) |✅|   |   |
| 图像 | 图像分类 | [lenet](https://gitee.com/mindspore/models/tree/master/research/cv/lenet) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [maskrcnn_resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_resnet50) |✅| ✅ |   |
| 图像 | 目标检测 | [maskrcnn_mobilenetv1](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |✅| ✅ | ✅ |
| 图像 | 人群计数 | [MCNN](https://gitee.com/mindspore/models/tree/master/research/cv/MCNN) |✅| ✅ |   |
| 图像 | 图像分类 | [mobilenetv1](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv1) |✅| ✅ |   |
| 图像 | 图像分类 | [mobilenetv2](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv2) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [mobilenetv3](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv3) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [nasnet](https://gitee.com/mindspore/models/tree/master/research/cv/nasnet) |✅| ✅ |   |
| 图像 | 图像质量评估 | [nima](https://gitee.com/mindspore/models/tree/master/research/cv/nima) |✅| ✅ |   |
| 图像 | 点云模型 | [octsqueeze](https://gitee.com/mindspore/models/tree/master/official/cv/OctSqueeze) |✅| ✅ |   |
| 图像 | 关键点检测 | [openpose](https://gitee.com/mindspore/models/tree/master/official/cv/OpenPose) |✅|   |   |
| 图像 | 缺陷检测 | [patchcore](https://gitee.com/mindspore/models/tree/master/official/cv/PatchCore) |✅| ✅ |   |
| 图像 | 相机重定位 | [posenet](https://gitee.com/mindspore/models/tree/master/research/cv/PoseNet) |✅| ✅ |   |
| 图像 | 视频预测学习 | [predrnn++](https://gitee.com/mindspore/models/tree/master/research/cv/predrnn++) |✅|   |   |
| 图像 | 文本检测 | [psenet](https://gitee.com/mindspore/models/tree/master/research/cv/psenet) |✅| ✅ |   |
| 图像 | 姿态估计 | [pvnet](https://gitee.com/mindspore/models/tree/master/official/cv/PVNet) |✅|   |   |
| 图像 | 光流估计 | [pwcnet](https://gitee.com/mindspore/models/tree/master/official/cv/PWCNet) |✅| ✅ |   |
| 图像 | 图像超分 | [RDN](https://gitee.com/mindspore/models/tree/master/research/cv/RDN) |✅| ✅ |   |
| 图像 | 图像分类 | [resnet101](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [resnet152](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [resnet18](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [resnet34](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [resnet50_thor](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ |   |
| 图像 | 图像分类 | [resnext101](https://gitee.com/mindspore/models/tree/master/official/cv/ResNeXt) |✅| ✅ |   |
| 图像 | 图像分类 | [resnext50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNeXt) |✅| ✅ |   |
| 图像 | 目标检测 | [retinaface_resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaFace_ResNet50) | | ✅ |   |
| 图像 | 目标检测 | [retinanet](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaNet) |✅| ✅ |   |
| 图像 | 图像分类 | [se_resnext50](https://gitee.com/mindspore/models/tree/master/research/cv/SE_ResNeXt50) |✅|   |   |
| 图像 | 图像抠图 | [semantic_human_matting](https://gitee.com/mindspore/models/tree/master/official/cv/SemanticHumanMatting) |✅|   |   |
| 图像 | 图像分类 | [se-resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [shufflenetv1](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [shufflenetv2](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv2) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [simclr](https://gitee.com/mindspore/models/tree/master/research/cv/simclr) |✅| ✅ |   |
| 图像 | 关键点检测 | [simple_pose](https://gitee.com/mindspore/models/tree/master/research/cv/simple_pose) |✅| ✅ |   |
| 图像 | 目标检测 | [sphereface](https://gitee.com/mindspore/models/tree/master/research/cv/sphereface) |✅| ✅ |   |
| 图像 | 图像分类 | [squeezenet](https://gitee.com/mindspore/models/tree/master/research/cv/squeezenet) |✅| ✅ |   |
| 图像 | 图像分类 | [SqueezeNet_Residual](https://gitee.com/mindspore/models/tree/master/research/cv/squeezenet) |✅| ✅ |   |
| 图像 | 图像超分 | [srcnn](https://gitee.com/mindspore/models/tree/master/research/cv/srcnn) |✅| ✅ |   |
| 图像 | 目标检测 | [ssd_mobilenet-v1-fpn](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [ssd_mobilenet-v2](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [ssd-resnet50-fpn](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [ssd-vgg16](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| 图像 | 缺陷检测 | [ssim-ae](https://gitee.com/mindspore/models/tree/master/official/cv/SSIM-AE) |✅|   |   |
| 图像 | 图像分类 | [tinydarknet](https://gitee.com/mindspore/models/tree/master/research/cv/tinydarknet) |✅| ✅ | ✅ |
| 图像 | 语义分割 | [UNet_nested](https://gitee.com/mindspore/models/tree/master/official/cv/Unet) |✅| ✅ |   |
| 图像 | 语义分割 | [unet2d](https://gitee.com/mindspore/models/tree/master/official/cv/Unet) |✅| ✅ |   |
| 图像 | 语义分割 | [unet3d](https://gitee.com/mindspore/models/tree/master/official/cv/Unet3d) |✅| ✅ |   |
| 图像 | 图像分类 | [vgg16](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg16) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [vit](https://gitee.com/mindspore/models/tree/master/official/cv/VIT) |✅| ✅ |   |
| 图像 | 文本识别 | [warpctc](https://gitee.com/mindspore/models/tree/master/research/cv/warpctc) |✅| ✅ |   |
| 图像 | 图像分类 | [xception](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception) |✅| ✅ |   |
| 图像 | 目标检测 | [yolov3_darknet53](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3) |✅| ✅ |   |
| 图像 | 目标检测 | [yolov3_resnet18](https://gitee.com/mindspore/models/tree/master/research/cv/yolov3_resnet18) |✅|   |   |
| 图像 | 目标检测 | [yolov4](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv4) |✅|   |   |
| 图像 | 目标检测 | [yolov5s](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5) |✅| ✅ |   |
| 推荐 | 点击率预测 | [deep_and_cross](https://gitee.com/mindspore/models/tree/master/research/recommend/deep_and_cross) | | ✅ |   |
| 推荐 | 点击率预测 | [deepfm](https://gitee.com/mindspore/models/tree/master/official/recommend/DeepFM) |✅| ✅ |   |
| 推荐 | 点击率预测 | [fibinet](https://gitee.com/mindspore/models/tree/master/research/recommend/fibinet) | | ✅ |   |
| 推荐 | 点击率预测 | [wide_and_deep](https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep) |✅| ✅ |   |
| 推荐 | 点击率预测 | [wide_and_deep_multitable](https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep_Multitable) |✅| ✅ |   |
| 文本 | 自然语言理解 | [bert_base](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| 文本 | 自然语言理解 | [bert_bilstm_crf](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| 文本 | 自然语言理解 | [bert_finetuning](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| 文本 | 自然语言理解 | [bert_large](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅|   |   |
| 文本 | 自然语言理解 | [bert_nezha](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| 文本 | 自然语言理解 | [cpm](https://gitee.com/mindspore/models/tree/master/research/nlp/cpm) |✅| ✅ |   |
| 文本 | 对话 | [dgu](https://gitee.com/mindspore/models/tree/master/research/nlp/dgu) |✅| ✅ |   |
| 文本 | 对话 | [duconv](https://gitee.com/mindspore/models/tree/master/research/nlp/duconv) |✅| ✅ |   |
| 文本 | 情绪分类 | [emotect](https://gitee.com/mindspore/models/tree/master/research/nlp/emotect) |✅| ✅ |   |
| 文本 | 自然语言理解 | [ernie](https://gitee.com/mindspore/models/tree/master/research/nlp/ernie) |✅| ✅ |   |
| 文本 | 自然语言理解 | [fasttext](https://gitee.com/mindspore/models/tree/master/research/nlp/fasttext) |✅| ✅ |   |
| 文本 | 自然语言理解 | [gnmt_v2](https://gitee.com/mindspore/models/tree/master/research/nlp/gnmt_v2) |✅| ✅ |   |
| 文本 | 自然语言理解 | [gpt3](https://gitee.com/mindspore/models/tree/master/official/nlp/GPT) |✅|   |   |
| 文本 | 自然语言理解 | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) |✅| ✅ |   |
| 文本 | 情绪分类 | [lstm](https://gitee.com/mindspore/models/tree/master/official/nlp/LSTM) |✅| ✅ |   |
| 文本 | 自然语言理解 | [mass](https://gitee.com/mindspore/models/tree/master/research/nlp/mass) |✅| ✅ |   |
| 文本 | 预训练 | [pangu_alpha](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha) |✅| ✅ |   |
| 文本 | 自然语言理解 | [textcnn](https://gitee.com/mindspore/models/tree/master/research/nlp/textcnn) |✅| ✅ |   |
| 文本 | 自然语言理解 | [tinybert](https://gitee.com/mindspore/models/tree/master/research/nlp/tinybert) |✅| ✅ |   |
| 文本 | 自然语言理解 | [transformer](https://gitee.com/mindspore/models/tree/master/official/nlp/Transformer) |✅| ✅ |   |
| 视频 | 目标追踪 | [ADNet](https://gitee.com/mindspore/models/tree/master/research/cv/ADNet) |✅|   |   |
| 视频 | 视频分类 | [c3d](https://gitee.com/mindspore/models/tree/master/official/cv/C3D) |✅| ✅ |   |
| 视频 | 目标追踪 | [Deepsort](https://gitee.com/mindspore/models/tree/master/research/cv/Deepsort) |✅| ✅ |   |

### 研究网络

|  领域 | 子领域  | 网络   | Ascend | GPU | CPU |
|:------   |:------| :-----------  |:------:   |:------:  |:-----: |
| 3D | 三维重建 | [cmr](https://gitee.com/mindspore/models/tree/master/research/cv/cmr) | | ✅ |   |
| 3D | 三维重建 | [DecoMR](https://gitee.com/mindspore/models/tree/master/research/cv/DecoMR) | | ✅ |   |
| 3D | 三维重建 | [DeepLM](https://gitee.com/mindspore/models/tree/master/research/3d/DeepLM) | | ✅ |   |
| 3D | 三维重建 | [eppmvsnet](https://gitee.com/mindspore/models/tree/master/research/cv/eppmvsnet) | | ✅ |   |
| 3D | 三维物体检测 | [pointpillars](https://gitee.com/mindspore/models/tree/master/research/cv/pointpillars) |✅| ✅ |   |
| 语音 | 语音识别 | [ctcmodel](https://gitee.com/mindspore/models/tree/master/research/audio/ctcmodel) |✅|   |   |
| 语音 | 语音识别 | [deepspeech2](https://gitee.com/mindspore/models/tree/master/official/audio/DeepSpeech2) | | ✅ |   |
| 语音 | 语音唤醒 | [dscnn](https://gitee.com/mindspore/models/tree/master/research/audio/dscnn) |✅| ✅ |   |
| 语音 | 语音合成 | [FastSpeech](https://gitee.com/mindspore/models/tree/master/research/audio/FastSpeech) | | ✅ |   |
| 语音 | 语音标注 | [fcn-4](https://gitee.com/mindspore/models/tree/master/research/audio/fcn-4) |✅| ✅ |   |
| 语音 | 语音识别 | [jasper](https://gitee.com/mindspore/models/tree/master/research/audio/jasper) |✅| ✅ |   |
| 语音 | 语音合成 | [wavenet](https://gitee.com/mindspore/models/tree/master/research/audio/wavenet) |✅| ✅ |   |
| 图神经网络 | 图分类 | [dgcn](https://gitee.com/mindspore/models/tree/master/research/gnn/dgcn) |✅|   |   |
| 图神经网络 | 文本分类 | [hypertext](https://gitee.com/mindspore/models/tree/master/research/nlp/hypertext) |✅| ✅ |   |
| 图神经网络 | 图分类 | [sdne](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |✅|   |   |
| 图神经网络 | 社会和信息网络 | [sgcn](https://gitee.com/mindspore/models/tree/master/research/gnn/sgcn) |✅| ✅ |   |
| 图神经网络 | 文本分类 | [textrcnn](https://gitee.com/mindspore/models/tree/master/research/nlp/textrcnn) |✅| ✅ |   |
| 高性能计算 | 高性能计算 | [deepbsde](https://gitee.com/mindspore/models/tree/master/research/hpc/deepbsde) | | ✅ |   |
| 高性能计算 | 高性能计算 | [molecular_dynamics](https://gitee.com/mindspore/models/tree/master/research/hpc/molecular_dynamics) |✅|   |   |
| 高性能计算 | 高性能计算 | [ocean_model](https://gitee.com/mindspore/models/tree/master/research/hpc/ocean_model) | | ✅ |   |
| 高性能计算 | 高性能计算 | [pafnucy](https://gitee.com/mindspore/models/tree/master/research/hpc/pafnucy) |✅| ✅ |   |
| 高性能计算 | 高性能计算 | [pfnn](https://gitee.com/mindspore/models/tree/master/research/hpc/pfnn) | | ✅ |   |
| 高性能计算 | 高性能计算 | [pinns](https://gitee.com/mindspore/models/tree/master/research/hpc/pinns) | | ✅ |   |
| 图像 | 图像分类 | [3D_DenseNet](https://gitee.com/mindspore/models/tree/master/research/cv/3D_DenseNet) |✅| ✅ |   |
| 图像 | 语义分割 | [3dcnn](https://gitee.com/mindspore/models/tree/master/research/cv/3dcnn) |✅| ✅ |   |
| 图像 | 语义分割 | [adelaide_ea](https://gitee.com/mindspore/models/tree/master/research/cv/adelaide_ea) |✅|   |   |
| 图像 | 文本检测 | [advanced_east](https://gitee.com/mindspore/models/tree/master/research/cv/advanced_east) |✅| ✅ |   |
| 图像 | 风格转移 | [aecrnet](https://gitee.com/mindspore/models/tree/master/research/cv/aecrnet) |✅| ✅ |   |
| 图像 | 重新识别 | [AlignedReID](https://gitee.com/mindspore/models/tree/master/research/cv/AlignedReID) | | ✅ |   |
| 图像 | 重新识别 | [AlignedReID++](https://gitee.com/mindspore/models/tree/master/research/cv/AlignedReID++) |✅| ✅ |   |
| 图像 | 姿态估计 | [AlphaPose](https://gitee.com/mindspore/models/tree/master/research/cv/AlphaPose) |✅|   |   |
| 图像 | 风格转移 | [APDrawingGAN](https://gitee.com/mindspore/models/tree/master/research/cv/APDrawingGAN) |✅| ✅ |   |
| 图像 | 风格转移 | [ArbitraryStyleTransfer](https://gitee.com/mindspore/models/tree/master/research/cv/ArbitraryStyleTransfer) |✅| ✅ |   |
| 图像 | 目标检测 | [arcface](https://gitee.com/mindspore/models/tree/master/official/cv/Arcface) |✅| ✅ |   |
| 图像 | 关键点检测 | [ArtTrack](https://gitee.com/mindspore/models/tree/master/research/cv/ArtTrack) | | ✅ |   |
| 图像 | 风格转移 | [AttGAN](https://gitee.com/mindspore/models/tree/master/research/cv/AttGAN) |✅| ✅ |   |
| 图像 | 图像分类 | [augvit](https://gitee.com/mindspore/models/tree/master/research/cv/augvit) | | ✅ |   |
| 图像 | 图像分类 | [autoaugment](https://gitee.com/mindspore/models/tree/master/research/cv/autoaugment) |✅| ✅ |   |
| 图像 | 语义分割 | [Auto-DeepLab](https://gitee.com/mindspore/models/tree/master/research/cv/Auto-DeepLab) |✅|   |   |
| 图像 | 神经架构搜索 | [AutoSlim](https://gitee.com/mindspore/models/tree/master/research/cv/AutoSlim) |✅| ✅ |   |
| 图像 | 图像分类 | [AVA_cifar](https://gitee.com/mindspore/models/tree/master/research/cv/AVA_cifar) |✅| ✅ |   |
| 图像 | 图像分类 | [AVA_hpa](https://gitee.com/mindspore/models/tree/master/research/cv/AVA_hpa) |✅| ✅ |   |
| 图像 | 图像分类 | [cait](https://gitee.com/mindspore/models/tree/master/research/cv/cait) |✅| ✅ |   |
| 图像 | 目标检测 | [CascadeRCNN](https://gitee.com/mindspore/models/tree/master/research/cv/CascadeRCNN) |✅| ✅ |   |
| 图像 | 图像分类 | [CBAM](https://gitee.com/mindspore/models/tree/master/research/cv/CBAM) |✅|   |   |
| 图像 | 图像分类 | [cct](https://gitee.com/mindspore/models/tree/master/research/cv/cct) |✅| ✅ |   |
| 图像 | 关键点检测 | [centernet](https://gitee.com/mindspore/models/tree/master/research/cv/centernet) |✅|   | ✅ |
| 图像 | 关键点检测 | [centernet_det](https://gitee.com/mindspore/models/tree/master/research/cv/centernet_det) |✅|   |   |
| 图像 | 关键点检测 | [centernet_resnet101](https://gitee.com/mindspore/models/tree/master/research/cv/centernet_resnet101) |✅| ✅ |   |
| 图像 | 关键点检测 | [centernet_resnet50_v1](https://gitee.com/mindspore/models/tree/master/research/cv/centernet_resnet50_v1) |✅|   |   |
| 图像 | 图像生成 | [CGAN](https://gitee.com/mindspore/models/tree/master/research/cv/CGAN) |✅| ✅ |   |
| 图像 | 图像分类 | [convnext](https://gitee.com/mindspore/models/tree/master/research/cv/convnext) |✅| ✅ |   |
| 图像 | 图像超分 | [csd](https://gitee.com/mindspore/models/tree/master/research/cv/csd) |✅| ✅ |   |
| 图像 | 图像生成 | [CTSDG](https://gitee.com/mindspore/models/tree/master/research/cv/CTSDG) |   | ✅ |   |
| 图像 | 风格转移 | [CycleGAN](https://gitee.com/mindspore/models/tree/master/official/cv/CycleGAN) |✅| ✅ |   |
| 图像 | 图像超分 | [DBPN](https://gitee.com/mindspore/models/tree/master/research/cv/DBPN) |✅|   |   |
| 图像 | 图像超分 | [DBPN_GAN](https://gitee.com/mindspore/models/tree/master/research/cv/DBPN) |✅|   |   |
| 图像 | 图像生成 | [dcgan](https://gitee.com/mindspore/models/tree/master/research/cv/dcgan) |✅| ✅ |   |
| 图像 | 重新识别 | [DDAG](https://gitee.com/mindspore/models/tree/master/research/cv/DDAG) |✅| ✅ |   |
| 图像 | 语义分割 | [DDM](https://gitee.com/mindspore/models/tree/master/research/cv/DDM) |✅|   |   |
| 图像 | 语义分割 | [DDRNet](https://gitee.com/mindspore/models/tree/master/research/cv/DDRNet) |✅| ✅ |   |
| 图像 | 目标检测 | [DeepID](https://gitee.com/mindspore/models/tree/master/research/cv/DeepID) |✅| ✅ |   |
| 图像 | 语义分割 | [deeplabv3plus](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P) |✅| ✅ |   |
| 图像 | 图像检索 | [delf](https://gitee.com/mindspore/models/tree/master/research/cv/delf) |✅|   |   |
| 图像 | 零样本学习 | [dem](https://gitee.com/mindspore/models/tree/master/research/cv/dem) |✅| ✅ |   |
| 图像 | 目标检测 | [detr](https://gitee.com/mindspore/models/tree/master/research/cv/detr) |✅| ✅ |   |
| 图像 | 语义分割 | [dgcnet_res101](https://gitee.com/mindspore/models/tree/master/research/cv/dgcnet_res101) | | ✅ |   |
| 图像 | 实例分割 | [dlinknet](https://gitee.com/mindspore/models/tree/master/research/cv/dlinknet) |✅|   |   |
| 图像 | 图像去噪 | [DnCNN](https://gitee.com/mindspore/models/tree/master/research/cv/DnCNN) |✅|   |   |
| 图像 | 图像分类 | [dnet_nas](https://gitee.com/mindspore/models/tree/master/research/cv/dnet_nas) |✅|   |   |
| 图像 | 图像分类 | [DRNet](https://gitee.com/mindspore/models/tree/master/research/cv/DRNet) |✅| ✅ |   |
| 图像 | 图像超分 | [EDSR](https://gitee.com/mindspore/models/tree/master/official/cv/EDSR) |✅|   |   |
| 图像 | 目标检测 | [EfficientDet_d0](https://gitee.com/mindspore/models/tree/master/research/cv/EfficientDet_d0) |✅|   |   |
| 图像 | 图像分类 | [efficientnet-b0](https://gitee.com/mindspore/models/tree/master/research/cv/efficientnet-b0) |✅|   |   |
| 图像 | 图像分类 | [efficientnet-b1](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1) |✅|   |   |
| 图像 | 图像分类 | [efficientnet-b2](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b2) |✅| ✅ |   |
| 图像 | 图像分类 | [efficientnet-b3](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b3) |✅| ✅ |   |
| 图像 | 图像分类 | [efficientnetv2](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnetv2) |✅|   |   |
| 图像 | 显著性检测 | [EGnet](https://gitee.com/mindspore/models/tree/master/research/cv/EGnet) |✅| ✅ |   |
| 图像 | 语义分割 | [E-NET](https://gitee.com/mindspore/models/tree/master/research/cv/E-NET) |✅| ✅ |   |
| 图像 | 图像超分 | [esr_ea](https://gitee.com/mindspore/models/tree/master/research/cv/esr_ea) |✅| ✅ |   |
| 图像 | 图像超分 | [ESRGAN](https://gitee.com/mindspore/models/tree/master/research/cv/ESRGAN) |✅| ✅ |   |
| 图像 | 图像分类 | [FaceAttribute](https://gitee.com/mindspore/models/tree/master/research/cv/FaceAttribute) |✅| ✅ |   |
| 图像 | 目标检测 | [faceboxes](https://gitee.com/mindspore/models/tree/master/research/cv/faceboxes) |✅|   |   |
| 图像 | 目标检测 | [FaceDetection](https://gitee.com/mindspore/models/tree/master/research/cv/FaceDetection) |✅| ✅ |   |
| 图像 | 人脸识别 | [FaceNet](https://gitee.com/mindspore/models/tree/master/research/cv/FaceNet) |✅| ✅ |   |
| 图像 | 图像分类 | [FaceQualityAssessment](https://gitee.com/mindspore/models/tree/master/research/cv/FaceQualityAssessment) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [FaceRecognition](https://gitee.com/mindspore/models/tree/master/official/cv/FaceRecognition) |✅| ✅ |   |
| 图像 | 目标检测 | [FaceRecognitionForTracking](https://gitee.com/mindspore/models/tree/master/research/cv/FaceRecognitionForTracking) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [faster_rcnn_dcn](https://gitee.com/mindspore/models/tree/master/research/cv/faster_rcnn_dcn) |✅| ✅ |   |
| 图像 | 图像抠图 | [FCANet](https://gitee.com/mindspore/models/tree/master/research/cv/FCANet) |✅|   |   |
| 图像 | 图像分类 | [FDA-BNN](https://gitee.com/mindspore/models/tree/master/research/cv/FDA-BNN) |✅| ✅ |   |
| 图像 | 图像分类 | [fishnet99](https://gitee.com/mindspore/models/tree/master/research/cv/fishnet99) |✅| ✅ |   |
| 图像 | 光流估计 | [flownet2](https://gitee.com/mindspore/models/tree/master/research/cv/flownet2) |✅|   |   |
| 图像 | 图像生成 | [gan](https://gitee.com/mindspore/models/tree/master/research/cv/gan) |✅| ✅ |   |
| 图像 | 图像分类 | [GENet_Res50](https://gitee.com/mindspore/models/tree/master/research/cv/GENet_Res50) |✅|   |   |
| 图像 | 图像分类 | [ghostnet](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet) |✅|   |   |
| 图像 | 图像分类 | [ghostnet_d](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet_d) |✅| ✅ |   |
| 图像 | 图像分类 | [glore_res200](https://gitee.com/mindspore/models/tree/master/research/cv/glore_res) |✅| ✅ |   |
| 图像 | 图像分类 | [glore_res50](https://gitee.com/mindspore/models/tree/master/research/cv/glore_res) |✅| ✅ |   |
| 图像 | 图像分类 | [hardnet](https://gitee.com/mindspore/models/tree/master/research/cv/hardnet) |✅| ✅ |   |
| 图像 | 边缘检测 | [hed](https://gitee.com/mindspore/models/tree/master/research/cv/hed) |✅| ✅ |   |
| 图像 | 图像生成 | [HiFaceGAN](https://gitee.com/mindspore/models/tree/master/research/cv/HiFaceGAN) | | ✅ |   |
| 图像 | 图像分类 | [HourNAS](https://gitee.com/mindspore/models/tree/master/research/cv/HourNAS) | | ✅ |   |
| 图像 | 图像分类 | [HRNetW48_cls](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_cls) |✅| ✅ |   |
| 图像 | 语义分割 | [HRNetW48_seg](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_seg) |✅|   |   |
| 图像 | 图像分类 | [ibnnet](https://gitee.com/mindspore/models/tree/master/research/cv/ibnnet) |✅| ✅ |   |
| 图像 | 语义分割 | [ICNet](https://gitee.com/mindspore/models/tree/master/research/cv/ICNet) |✅|   |   |
| 图像 | 图像分类 | [inception_resnet_v2](https://gitee.com/mindspore/models/tree/master/research/cv/inception_resnet_v2) |✅| ✅ |   |
| 图像 | 图像分类 | [Inceptionv2](https://gitee.com/mindspore/models/tree/master/research/cv/Inception-v2) |✅| ✅ |   |
| 图像 | 图像抠图 | [IndexNet](https://gitee.com/mindspore/models/tree/master/research/cv/IndexNet) | | ✅ |   |
| 图像 | 图像生成 | [IPT](https://gitee.com/mindspore/models/tree/master/research/cv/IPT) |✅|   |   |
| 图像 | 图像超分 | [IRN](https://gitee.com/mindspore/models/tree/master/research/cv/IRN) |✅| ✅ |   |
| 图像 | 图像分类 | [ISyNet](https://gitee.com/mindspore/models/tree/master/research/cv/ISyNet) |✅| ✅ |   |
| 图像 | 图像分类 | [ivpf](https://gitee.com/mindspore/models/tree/master/research/cv/ivpf) | | ✅ |   |
| 图像 | 图像去噪 | [LearningToSeeInTheDark](https://gitee.com/mindspore/models/tree/master/research/cv/LearningToSeeInTheDark) |✅|   |   |
| 图像 | 元学习 | [LEO](https://gitee.com/mindspore/models/tree/master/research/cv/LEO) |✅| ✅ |   |
| 图像 | 目标检测 | [LightCNN](https://gitee.com/mindspore/models/tree/master/research/cv/LightCNN) |✅| ✅ | ✅ |
| 图像 | 图像超分 | [lite-hrnet](https://gitee.com/mindspore/models/tree/master/research/cv/lite-hrnet) | | ✅ |   |
| 图像 | 图像分类 | [lresnet100e_ir](https://gitee.com/mindspore/models/tree/master/research/cv/lresnet100e_ir) | | ✅ |   |
| 图像 | 目标检测 | [m2det](https://gitee.com/mindspore/models/tree/master/research/cv/m2det) | | ✅ |   |
| 图像 | 自编码 | [mae](https://gitee.com/mindspore/models/tree/master/official/cv/MAE) |✅| ✅ |   |
| 图像 | 元学习 | [MAML](https://gitee.com/mindspore/models/tree/master/research/cv/MAML) |✅| ✅ |   |
| 图像 | 文本识别 | [ManiDP](https://gitee.com/mindspore/models/tree/master/research/cv/ManiDP) | | ✅ |   |
| 图像 | 人脸识别 | [MaskedFaceRecognition](https://gitee.com/mindspore/models/tree/master/research/cv/MaskedFaceRecognition) |✅|   |   |
| 图像 | 元学习 | [meta-baseline](https://gitee.com/mindspore/models/tree/master/research/cv/meta-baseline) |✅| ✅ |   |
| 图像 | 重新识别 | [MGN](https://gitee.com/mindspore/models/tree/master/research/cv/MGN) |✅| ✅ |   |
| 图像 | 深度估计 | [midas](https://gitee.com/mindspore/models/tree/master/research/cv/midas) |✅| ✅ |   |
| 图像 | 图像去噪 | [MIMO-UNet](https://gitee.com/mindspore/models/tree/master/research/cv/MIMO-UNet) | | ✅ |   |
| 图像 | 图像分类 | [mnasnet](https://gitee.com/mindspore/models/tree/master/research/cv/mnasnet) |✅| ✅ |   |
| 图像 | 图像分类 | [mobilenetv3_large](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetv3_large) |✅|   | ✅ |
| 图像 | 图像分类 | [mobilenetV3_small_x1_0](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetV3_small_x1_0) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [MultiTaskNet](https://gitee.com/mindspore/models/tree/master/research/cv/PAMTRI/MultiTaskNet) |✅| ✅ |   |
| 图像 | 重新识别 | [MVD](https://gitee.com/mindspore/models/tree/master/research/cv/MVD) |✅| ✅ |   |
| 图像 | 目标检测 | [nas-fpn](https://gitee.com/mindspore/models/tree/master/research/cv/nas-fpn) |✅|   |   |
| 图像 | 图像去噪 | [Neighbor2Neighbor](https://gitee.com/mindspore/models/tree/master/research/cv/Neighbor2Neighbor) |✅| ✅ |   |
| 图像 | 图像分类 | [NFNet](https://gitee.com/mindspore/models/tree/master/research/cv/NFNet) |✅| ✅ |   |
| 图像 | 图像质量评估 | [nima_vgg16](https://gitee.com/mindspore/models/tree/master/research/cv/nima_vgg16) | | ✅ |   |
| 图像 | 语义分割 | [nnUNet](https://gitee.com/mindspore/models/tree/master/research/cv/nnUNet) |✅| ✅ |   |
| 图像 | 图像分类 | [ntsnet](https://gitee.com/mindspore/models/tree/master/research/cv/ntsnet) |✅| ✅ |   |
| 图像 | 语义分割 | [OCRNet](https://gitee.com/mindspore/models/tree/master/official/cv/OCRNet) |✅| ✅ |   |
| 图像 | 重新识别 | [osnet](https://gitee.com/mindspore/models/tree/master/research/cv/osnet) |✅| ✅ |   |
| 图像 | 显著性检测 | [PAGENet](https://gitee.com/mindspore/models/tree/master/research/cv/PAGENet) |✅| ✅ |   |
| 图像 | 图像检索 | [pcb](https://gitee.com/mindspore/models/tree/master/research/cv/pcb_rpp) | | ✅ |   |
| 图像 | 图像检索 | [pcb](https://gitee.com/mindspore/models/tree/master/research/cv/pcb_rpp) | | ✅ |   |
| 图像 | 图像检索 | [pcb_rpp](https://gitee.com/mindspore/models/tree/master/research/cv/pcb_rpp) | | ✅ |   |
| 图像 | 图像分类 | [PDarts](https://gitee.com/mindspore/models/tree/master/research/cv/PDarts) |✅| ✅ |   |
| 图像 | 图像生成 | [PGAN](https://gitee.com/mindspore/models/tree/master/research/cv/PGAN) |✅| ✅ |   |
| 图像 | 图像生成 | [Pix2Pix](https://gitee.com/mindspore/models/tree/master/research/cv/Pix2Pix) |✅| ✅ |   |
| 图像 | 图像超分 | [Pix2PixHD](https://gitee.com/mindspore/models/tree/master/official/cv/Pix2PixHD) |✅|   |   |
| 图像 | 图像分类 | [pnasnet](https://gitee.com/mindspore/models/tree/master/research/cv/pnasnet) |✅| ✅ |   |
| 图像 | 点云模型 | [pointnet](https://gitee.com/mindspore/models/tree/master/official/cv/PointNet) |✅| ✅ |   |
| 图像 | 点云模型 | [pointnet2](https://gitee.com/mindspore/models/tree/master/official/cv/PointNet2) |✅| ✅ |   |
| 图像 | 图像分类 | [PoseEstNet](https://gitee.com/mindspore/models/tree/master/research/cv/PAMTRI/PoseEstNet) |✅| ✅ |   |
| 图像 | 图像分类 | [ProtoNet](https://gitee.com/mindspore/models/tree/master/research/cv/ProtoNet) |✅| ✅ |   |
| 图像 | 图像分类 | [proxylessnas](https://gitee.com/mindspore/models/tree/master/research/cv/proxylessnas) |✅| ✅ |   |
| 图像 | 语义分割 | [PSPNet](https://gitee.com/mindspore/models/tree/master/research/cv/PSPNet) |✅|   |   |
| 图像 | 显著性检测 | [ras](https://gitee.com/mindspore/models/tree/master/research/cv/ras) |✅| ✅ |   |
| 图像 | 图像超分 | [RCAN](https://gitee.com/mindspore/models/tree/master/research/cv/RCAN) |✅|   |   |
| 图像 | 目标检测 | [rcnn](https://gitee.com/mindspore/models/tree/master/research/cv/rcnn) |✅| ✅ |   |
| 图像 | 图像超分 | [REDNet30](https://gitee.com/mindspore/models/tree/master/research/cv/REDNet30) |✅| ✅ |   |
| 图像 | 目标检测 | [RefineDet](https://gitee.com/mindspore/models/tree/master/research/cv/RefineDet) |✅| ✅ |   |
| 图像 | 语义分割 | [RefineNet](https://gitee.com/mindspore/models/tree/master/research/cv/RefineNet) |✅| ✅ |   |
| 图像 | 重新识别 | [ReIDStrongBaseline](https://gitee.com/mindspore/models/tree/master/research/cv/ReIDStrongBaseline) |✅| ✅ |   |
| 图像 | 图像分类 | [relationnet](https://gitee.com/mindspore/models/tree/master/research/cv/relationnet) |✅| ✅ |   |
| 图像 | 图像分类 | [renas](https://gitee.com/mindspore/models/tree/master/research/cv/renas) |✅| ✅ | ✅ |
| 图像 | 语义分割 | [repvgg](https://gitee.com/mindspore/models/tree/master/research/cv/repvgg) |✅| ✅ |   |
| 图像 | 语义分割 | [res2net_deeplabv3](https://gitee.com/mindspore/models/tree/master/research/cv/res2net_deeplabv3) |✅|   | ✅ |
| 图像 | 目标检测 | [res2net_faster_rcnn](https://gitee.com/mindspore/models/tree/master/research/cv/res2net_faster_rcnn) |✅| ✅ |   |
| 图像 | 目标检测 | [res2net_yolov3](https://gitee.com/mindspore/models/tree/master/research/cv/res2net_yolov3) |✅| ✅ |   |
| 图像 | 图像分类 | [res2net101](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| 图像 | 图像分类 | [res2net152](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| 图像 | 图像分类 | [res2net50](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| 图像 | 图像分类 | [ResNeSt50](https://gitee.com/mindspore/models/tree/master/research/cv/ResNeSt50) |✅| ✅ |   |
| 图像 | 图像分类 | [resnet50_adv_pruning](https://gitee.com/mindspore/models/tree/master/research/cv/resnet50_adv_pruning) |✅| ✅ |   |
| 图像 | 图像分类 | [resnet50_bam](https://gitee.com/mindspore/models/tree/master/research/cv/resnet50_bam) |✅| ✅ |   |
| 图像 | 图像分类 | [ResNet50-Quadruplet](https://gitee.com/mindspore/models/tree/master/research/cv/metric_learn) |✅| ✅ |   |
| 图像 | 图像分类 | [ResNet50-Triplet](https://gitee.com/mindspore/models/tree/master/research/cv/metric_learn) |✅| ✅ |   |
| 图像 | 图像分类 | [ResnetV2_101](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2) |✅| ✅ |   |
| 图像 | 图像分类 | [ResnetV2_152](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2) |✅| ✅ |   |
| 图像 | 图像分类 | [ResnetV2_50](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2) |✅| ✅ |   |
| 图像 | 图像分类 | [resnetv2_50_frn](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2_50_frn) |✅| ✅ |   |
| 图像 | 图像分类 | [resnext152_64x4d](https://gitee.com/mindspore/models/tree/master/research/cv/resnext152_64x4d) |✅| ✅ |   |
| 图像 | 目标检测 | [retinaface_mobilenet0.25](https://gitee.com/mindspore/models/tree/master/research/cv/retinaface) |✅| ✅ |   |
| 图像 | 目标检测 | [retinanet_resnet101](https://gitee.com/mindspore/models/tree/master/research/cv/retinanet_resnet101) |✅| ✅ |   |
| 图像 | 目标检测 | [retinanet_resnet152](https://gitee.com/mindspore/models/tree/master/research/cv/retinanet_resnet152) |✅| ✅ |   |
| 图像 | 目标检测 | [rfcn](https://gitee.com/mindspore/models/tree/master/research/cv/rfcn) | | ✅ |   |
| 图像 | 图像分类 | [SE_ResNeXt50](https://gitee.com/mindspore/models/tree/master/research/cv/SE_ResNeXt50) |✅|   |   |
| 图像 | 图像分类 | [senet_resnet101](https://gitee.com/mindspore/models/tree/master/research/cv/SE-Net) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [senet_resnet50](https://gitee.com/mindspore/models/tree/master/research/cv/SE-Net) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [se-res2net50](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| 图像 | 图像分类 | [S-GhostNet](https://gitee.com/mindspore/models/tree/master/research/cv/S-GhostNet) |✅|   |   |
| 图像 | 姿态估计 | [simple_baselines](https://gitee.com/mindspore/models/tree/master/research/cv/simple_baselines) |✅| ✅ |   |
| 图像 | 图像生成 | [SinGAN](https://gitee.com/mindspore/models/tree/master/research/cv/SinGAN) |✅|   |   |
| 图像 | 图像分类 | [single_path_nas](https://gitee.com/mindspore/models/tree/master/research/cv/single_path_nas) |✅| ✅ |   |
| 图像 | 图像分类 | [sknet](https://gitee.com/mindspore/models/tree/master/research/cv/sknet) |✅| ✅ | ✅ |
| 图像 | 图像分类 | [snn_mlp](https://gitee.com/mindspore/models/tree/master/research/cv/snn_mlp) | | ✅ |   |
| 图像 | 目标检测 | [Spnas](https://gitee.com/mindspore/models/tree/master/research/cv/Spnas) |✅|   |   |
| 图像 | 图像分类 | [SPPNet](https://gitee.com/mindspore/models/tree/master/research/cv/SPPNet) |✅| ✅ |   |
| 图像 | 图像分类 | [squeezenet](https://gitee.com/mindspore/models/tree/master/research/cv/squeezenet) |✅| ✅ |   |
| 图像 | 图像超分 | [sr_ea](https://gitee.com/mindspore/models/tree/master/research/cv/sr_ea) |✅|   |   |
| 图像 | 图像超分 | [SRGAN](https://gitee.com/mindspore/models/tree/master/research/cv/SRGAN) |✅| ✅ |   |
| 图像 | 图像分类 | [ssc_resnet50](https://gitee.com/mindspore/models/tree/master/research/cv/ssc_resnet50) |✅| ✅ |   |
| 图像 | 目标检测 | [ssd_ghostnet](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_ghostnet) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [ssd_inception_v2](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_inception_v2) | | ✅ | ✅ |
| 图像 | 目标检测 | [ssd_inceptionv2](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_inceptionv2) |✅|   |   |
| 图像 | 目标检测 | [ssd_mobilenetV2](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_mobilenetV2) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [ssd_mobilenetV2_FPNlite](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_mobilenetV2_FPNlite) |✅| ✅ | ✅ |
| 图像 | 目标检测 | [ssd_resnet_34](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_resnet_34) | | ✅ |   |
| 图像 | 目标检测 | [ssd_resnet34](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_resnet34) |✅|   | ✅ |
| 图像 | 目标检测 | [ssd_resnet50](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_resnet50) |✅|   |   |
| 图像 | 姿态估计 | [StackedHourglass](https://gitee.com/mindspore/models/tree/master/research/cv/StackedHourglass) |✅|   |   |
| 图像 | 图像生成 | [StarGAN](https://gitee.com/mindspore/models/tree/master/research/cv/StarGAN) |✅| ✅ |   |
| 图像 | 图像生成 | [STGAN](https://gitee.com/mindspore/models/tree/master/research/cv/STGAN) |✅| ✅ |   |
| 图像 | 交通预测 | [stgcn](https://gitee.com/mindspore/models/tree/master/research/cv/stgcn) |✅| ✅ |   |
| 图像 | 图像分类 | [stpm](https://gitee.com/mindspore/models/tree/master/official/cv/STPM) |✅| ✅ |   |
| 图像 | 图像分类 | [swin_transformer](https://gitee.com/mindspore/models/tree/master/official/cv/SwinTransformer) |✅| ✅ |   |
| 图像 | 时间定位 | [tall](https://gitee.com/mindspore/models/tree/master/research/cv/tall) |✅|   |   |
| 图像 | 图像分类 | [TCN](https://gitee.com/mindspore/models/tree/master/research/cv/TCN) |✅| ✅ |   |
| 图像 | 文本检测 | [textfusenet](https://gitee.com/mindspore/models/tree/master/research/cv/textfusenet) |✅|   |   |
| 图像 | 交通预测 | [tgcn](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn) |✅| ✅ |   |
| 图像 | 图像分类 | [tinynet](https://gitee.com/mindspore/models/tree/master/research/cv/tinynet) | | ✅ |   |
| 图像 | 图像分类 | [TNT](https://gitee.com/mindspore/models/tree/master/research/cv/TNT) |✅| ✅ |   |
| 图像 | 目标检测 | [u2net](https://gitee.com/mindspore/models/tree/master/research/cv/u2net) |✅| ✅ |   |
| 图像 | 图像生成 | [U-GAT-IT](https://gitee.com/mindspore/models/tree/master/research/cv/U-GAT-IT) |✅| ✅ |   |
| 图像 | 语义分割 | [UNet3+](https://gitee.com/mindspore/models/tree/master/research/cv/UNet3+) |✅| ✅ |   |
| 图像 | 重新识别 | [VehicleNet](https://gitee.com/mindspore/models/tree/master/research/cv/VehicleNet) |✅|   |   |
| 图像 | 图像分类 | [vgg19](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg19) |✅| ✅ |   |
| 图像 | 图像分类 | [ViG](https://gitee.com/mindspore/models/tree/master/research/cv/ViG) |✅| ✅ |   |
| 图像 | 图像分类 | [vit_cifar](https://gitee.com/mindspore/models/tree/master/research/cv/vit_base) |✅| ✅ |   |
| 图像 | 语义分割 | [vnet](https://gitee.com/mindspore/models/tree/master/research/cv/vnet) |✅| ✅ |   |
| 图像 | 图像分类 | [wave_mlp](https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp) |✅| ✅ |   |
| 图像 | 图像超分 | [wdsr](https://gitee.com/mindspore/models/tree/master/research/cv/wdsr) |✅| ✅ |   |
| 图像 | 图像生成 | [wgan](https://gitee.com/mindspore/models/tree/master/official/cv/WGAN) |✅|   |   |
| 图像 | 图像分类 | [wideresnet](https://gitee.com/mindspore/models/tree/master/research/cv/wideresnet) |✅| ✅ |   |
| 图像 | 实例分割 | [Yolact++](https://gitee.com/mindspore/models/tree/master/research/cv/Yolact++) |✅|   |   |
| 图像 | 目标检测 | [yolov3_tiny](https://gitee.com/mindspore/models/tree/master/research/cv/yolov3_tiny) |✅| ✅ |   |
| 图像 | 目标检测 | [yolox](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOX) |✅|   |   |
| 多模态 | 多模态 | [opt](https://gitee.com/mindspore/models/tree/master/research/mm/opt) |✅| ✅ |   |
| 多模态 | 多模态 | [TokenFusion](https://gitee.com/mindspore/models/tree/master/research/cv/TokenFusion) |✅| ✅ |   |
| 多模态 | 多模态 | [wukong](https://gitee.com/mindspore/models/tree/master/research/mm/wukong) |✅|   |   |
| 推荐 | 点击率预测 | [autodis](https://gitee.com/mindspore/models/tree/master/research/recommend/autodis) |✅| ✅ |   |
| 推荐 | 点击率预测 | [DIEN](https://gitee.com/mindspore/models/tree/master/research/recommend/DIEN) |✅| ✅ |   |
| 推荐 | 点击率预测 | [dlrm](https://gitee.com/mindspore/models/tree/master/research/recommend/dlrm) |✅| ✅ |   |
| 推荐 | 点击率预测 | [EDCN](https://gitee.com/mindspore/models/tree/master/research/recommend/EDCN) |✅| ✅ |   |
| 推荐 | 点击率预测 | [Fat-DeepFFM](https://gitee.com/mindspore/models/tree/master/research/recommend/Fat-DeepFFM) |✅| ✅ |   |
| 推荐 | 点击率预测 | [mmoe](https://gitee.com/mindspore/models/tree/master/research/recommend/mmoe) |✅| ✅ |   |
| 文本 | 自然语言理解 | [albert](https://gitee.com/mindspore/models/tree/master/research/nlp/albert) |✅| ✅ |   |
| 文本 | 情绪分类 | [atae_lstm](https://gitee.com/mindspore/models/tree/master/research/nlp/atae_lstm) |✅| ✅ |   |
| 文本 | 对话 | [dam](https://gitee.com/mindspore/models/tree/master/research/nlp/dam) |✅|   |   |
| 文本 | 语言模型 | [gpt2](https://gitee.com/mindspore/models/tree/master/research/nlp/gpt2) |✅|   |   |
| 文本 | 知识图嵌入 | [hake](https://gitee.com/mindspore/models/tree/master/research/nlp/hake) | | ✅ |   |
| 文本 | 自然语言理解 | [ktnet](https://gitee.com/mindspore/models/tree/master/research/nlp/ktnet) |✅| ✅ |   |
| 文本 | 命名实体识别 | [lstm_crf](https://gitee.com/mindspore/models/tree/master/research/nlp/lstm_crf) |✅|   |   |
| 文本 | 自然语言理解 | [luke](https://gitee.com/mindspore/models/tree/master/research/nlp/luke) |✅| ✅ |   |
| 文本 | 知识图嵌入 | [rotate](https://gitee.com/mindspore/models/tree/master/research/nlp/rotate) |✅| ✅ |   |
| 文本 | 情绪分类 | [senta](https://gitee.com/mindspore/models/tree/master/research/nlp/senta) |✅| ✅ |   |
| 文本 | 机器翻译 | [seq2seq](https://gitee.com/mindspore/models/tree/master/research/nlp/seq2seq) |✅|   |   |
| 文本 | 词嵌入 | [skipgram](https://gitee.com/mindspore/models/tree/master/research/nlp/skipgram) |✅| ✅ |   |
| 文本 | 机器翻译 | [speech_transformer](https://gitee.com/mindspore/models/tree/master/research/nlp/speech_transformer) |✅|   |   |
| 文本 | 预训练 | [ternarybert](https://gitee.com/mindspore/models/tree/master/research/nlp/ternarybert) |✅| ✅ |   |
| 文本 | 自然语言理解 | [tprr](https://gitee.com/mindspore/models/tree/master/research/nlp/tprr) |✅|   |   |
| 文本 | 自然语言理解 | [transformer_xl](https://gitee.com/mindspore/models/tree/master/research/nlp/transformer_xl) |✅| ✅ |   |
| 文本 | 知识图嵌入 | [transX](https://gitee.com/mindspore/models/tree/master/research/nlp/transX) | | ✅ |   |
| 视频 | 视频分类 | [AttentionCluster](https://gitee.com/mindspore/models/tree/master/research/cv/AttentionCluster) |✅| ✅ |   |
| 视频 | 其他 | [DYR](https://gitee.com/mindspore/models/tree/master/research/nlp/DYR) |✅|   |   |
| 视频 | 视频分类 | [ecolite](https://gitee.com/mindspore/models/tree/master/research/cv/ecolite) |✅|   |   |
| 视频 | 目标追踪 | [fairmot](https://gitee.com/mindspore/models/tree/master/research/cv/fairmot) |✅| ✅ |   |
| 视频 | 视频分类 | [I3D](https://gitee.com/mindspore/models/tree/master/research/cv/I3D) |✅|   |   |
| 视频 | 目标追踪 | [JDE](https://gitee.com/mindspore/models/tree/master/research/cv/JDE) | | ✅ |   |
| 视频 | 视频分割 | [OSVOS](https://gitee.com/mindspore/models/tree/master/research/cv/OSVOS) | | ✅ |   |
| 视频 | 视频分类 | [r2plus1d](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d) |✅| ✅ |   |
| 视频 | 视频超分 | [rbpn](https://gitee.com/mindspore/models/tree/master/research/cv/rbpn) |✅|   |   |
| 视频 | 视频分类 | [resnet3d](https://gitee.com/mindspore/models/tree/master/research/cv/resnet3d) |✅|   |   |
| 视频 | 目标追踪 | [SiamFC](https://gitee.com/mindspore/models/tree/master/research/cv/SiamFC) |✅|   |   |
| 视频 | 目标追踪 | [siamRPN](https://gitee.com/mindspore/models/tree/master/research/cv/siamRPN) |✅| ✅ |   |
| 视频 | 视频分类 | [slowfast](https://gitee.com/mindspore/models/tree/master/research/cv/slowfast) |✅| ✅ |   |
| 视频 | 视频分类 | [stnet](https://gitee.com/mindspore/models/tree/master/research/cv/stnet) |✅|   |   |
| 视频 | 目标追踪 | [tracktor](https://gitee.com/mindspore/models/tree/master/research/cv/tracktor) | | ✅ |   |
| 视频 | 目标追踪 | [tracktor++](https://gitee.com/mindspore/models/tree/master/research/cv/tracktor++) |✅| ✅ |   |
| 视频 | 视频分类 | [trn](https://gitee.com/mindspore/models/tree/master/research/cv/trn) | | ✅ |   |
| 视频 | 视频分类 | [tsm](https://gitee.com/mindspore/models/tree/master/research/cv/tsm) |✅| ✅ |   |
| 视频 | 视频分类 | [tsn](https://gitee.com/mindspore/models/tree/master/research/cv/tsn) |✅| ✅ |   |

Process finished with exit code 0

- [社区](https://gitee.com/mindspore/models/tree/master/community)

## 公告

### 2021.9.15 `models`独立建仓

`models`仓库由原[mindspore仓库](https://gitee.com/mindspore/mindspore)的model_zoo目录独立分离而来，新仓库不继承历史commit记录，如果需要查找历史提2021.9.15之前的提交，请到mindspore仓库进行查询。

## 关联站点

这里是MindSpore框架提供的可以运行于包括Ascend/GPU/CPU/移动设备等多种设备的模型库。

相应的专属于Ascend平台的多框架模型可以参考[昇腾ModelZoo](https://hiascend.com/software/modelzoo)以及对应的[代码仓](https://gitee.com/ascend/modelzoo)。

MindSpore相关的预训练模型可以在[MindSpore hub](https://www.mindspore.cn/resources/hub)或[下载中心](https://download.mindspore.cn/model_zoo/).

## 免责声明

MindSpore仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在MindSpore中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过GitHub或Gitee与我们联系。非常感谢您对这个社区的理解和贡献。

MindSpore已获得Apache 2.0许可，请参见LICENSE文件。

## 许可证

[Apache 2.0许可证](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)

## FAQ

想要获取更多关于`MindSpore`框架使用本身的FAQ问题的，可以参考[官网FAQ](https://www.mindspore.cn/docs/zh-CN/master/faq/installation.html)

- **Q: 直接使用models下的模型出现内存不足错误，例如*Failed to alloc memory pool memory*, 该怎么处理?**

  **A**: 直接使用models下的模型出现内存不足的典型原因是由于运行模式（`PYNATIVE_MODE`)、运行环境配置、License控制（AI-TOKEN）的不同造成的：
    - `PYNATIVE_MODE`通常比`GRAPH_MODE`使用更多内存，尤其是在需要进行反向传播计算的训练图中，当前有2种方法可以尝试解决该问题。
        方法1：你可以尝试使用一些更小的batch size；
        方法2：添加context.set_context(mempool_block_size="XXGB")，其中，“XX”当前最大有效值可设置为“31”。
        如果将方法1与方法2结合使用，效果会更好。
    - 运行环境由于NPU的核数、内存等配置不同也会产生类似问题。
    - License控制（AI-TOKEN）的不同档位会造成执行过程中内存开销不同，也可以尝试使用一些更小的batch size。

- **Q: 一些网络运行中报错接口不存在，例如cannot import，该怎么处理?**

  **A**: 优先检查一下获取网络脚本的分支，与所使用的MindSpore版本是否一致，部分新分支中的模型脚本会使用一些新版本MindSpore才支持的接口，从而在使用老版本MindSpore时会发生报错.

- **Q: 一些模型描述中提到的*RANK_TABLE_FILE*文件，是什么？**

  **A**: *RANK_TABLE_FILE*是一个Ascend环境上用于指定分布式集群信息的文件，更多信息可以参考生成工具[hccl_toos](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)和[分布式并行训练教程](https://mindspore.cn/docs/programming_guide/zh-CN/r1.5/distributed_training_ascend.html#id4)

- **Q: 在windows环境上要怎么运行网络脚本？**

  **A**: 多数模型都是使用bash作为启动脚本，在Windows环境上无法直接使用bash命令，你可以考虑直接运行python命令而不是bash启动脚本 ，如果你确实想需要使用bash脚本，你可以考虑使用以下几种方法来运行模型：
    1. 使用虚拟环境，可以构造一个linux的虚拟机或docker容器，然后在虚拟环境中运行脚本
    2. 使用WSL，可以开启Windows的linux子系统来在Windows系统中运行linux，然后再WSL中运行脚本。
    3. 使用Windows Bash，需要获取一个可以直接在Windows上运行bash的环境，常见的选择是[cygwin](http://www.cygwin.com)或[git bash](https://gitforwindows.org)
    4. 跳过bash脚本，直接调用python程序。

- **Q: 网络在310推理时出现编译失败，报错信息指向gflags，例如*undefined reference to 'google::FlagRegisterer::FlagRegisterer'*，该怎么处理?**

  **A**: 优先检查一下环境GCC版本和gflags版本是否匹配，可以参考[官方链接](https://www.mindspore.cn/install)安装对应的GCC版本，[gflags](https://github.com/gflags/gflags/archive/v2.2.2.tar.gz)安装gflags。你需要保证所使用的组件之间是ABI兼容的，更多信息可以参考[_GLIBCXX_USE_CXX11_ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)

- **Q: 在Mac系统上加载mindrecord格式的数据集出错,例如*Invalid file, failed to open files for reading mindrecord files.*，该怎么处理?**

  **A**: 优先使用*ulimit -a*检查系统限制，如果*file descriptors*数量为256（默认值），需要使用*ulimit -n 1024*将其设置为1024（或者更大的值）。之后再检查文件是否损坏或者被修改。

- **Q: 我在多台服务器构成的大集群上进行训练，但是得到的精度比预期要低，该怎么办？**

  **A**: 当前模型库中的大部分模型只在单机内进行过验证，最大使用8卡进行训练。由于MindSpore训练时指定的`batch_size`是单卡的，所以当单机8卡升级到多机时，会导致全局的`global_batch_size`变大，这就导致需要针对当前多机场景的`global_batch_size`进行重新调参优化。
