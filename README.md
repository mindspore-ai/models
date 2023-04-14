# ![MindSpore Logo](https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png)

## Welcome to the Model Zoo for MindSpore

The MindSpore models repository provides different task domains, classic SOTA model implementations and end-to-end solutions. The purpose is to make it easier for MindSpore users to use MindSpore for research and product development.

In order to facilitate developers to enjoy the benefits of MindSpore framework, we will continue to add typical networks and some of the related pre-trained models. If you have needs for the model zoo, you can file an issue on [gitee](https://gitee.com/mindspore/mindspore/issues) or [MindSpore](https://bbs.huaweicloud.com/forum/forum-1076-1.html), We will consider it in time.

| Directory               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [official](official)    | • Official maintenance, iteratively updated with the MindSpore version, ensure that no problem in accuracy and performance for released version<br/>• Recommended writing style, use the latest MindSpore interface and recommended features, ensure faster performance while maintaining code readability<br/>• Detailed network information and documentation, including but not limited to model description, dataset usage, specification support, accuracy and performance data, network checkpoint files, MindIR files, etc                  |
| [research](research)    | • Passed the acceptance test in the older MindSpore version, indicate supported MindSpore versions in the README<br/>• Maintained and upgraded on demand, it will not be updated iteratively with the MindSpore version, but only adapt to the corresponding interface changes, Maintenance support is provided by MindSpore developers<br/>• Relatively detailed network information and documentation, including but not limited to model description, dataset usage, specification support, accuracy and performance data, network checkpoint files, MindIR files, etc |
| [community](community)  | • Contributed by ecological developer, maintained and upgraded on demand, indicate supported MindSpore versions in the README<br/>• Model file is not necessarily provided                                                                                                                                                                                                                                                                                                                                                                         |

- SOTA models using the latest MindSpore APIs

- The  best benefits from MindSpore

- Officially maintained and supported

## Table of Contents

### Official

|  Domain | Sub Domain    | Network  | Ascend  | GPU | CPU |
|:------   |:------| :-----------  |:------:   |:------:  |:-----: |
| Audio | Speaker Recognition | [ecapa_tdnn](https://gitee.com/mindspore/models/tree/master/official/audio/EcapaTDNN) |✅|   |   |
| Audio | Speech Synthesis | [lpcnet](https://gitee.com/mindspore/models/tree/master/official/audio/LPCNet) |✅| ✅ |   |
| Audio | Speech Synthesis | [melgan](https://gitee.com/mindspore/models/tree/master/official/audio/MELGAN) |✅| ✅ |   |
| Audio | Speech Synthesis | [tacotron2](https://gitee.com/mindspore/models/tree/master/official/audio/Tacotron2) |✅|   |   |
| Graph Neural Network | Text Classification | [bgcf](https://gitee.com/mindspore/models/tree/master/research/gnn/bgcf) |✅| ✅ |   |
| Graph Neural Network | Text Classification | [gat](https://gitee.com/mindspore/models/tree/master/research/gnn/gat) |✅| ✅ |   |
| Graph Neural Network | Text Classification | [gcn](https://gitee.com/mindspore/models/tree/master/official/gnn/GCN) |✅| ✅ |   |
| Recommendation | Recommender System | [naml](https://gitee.com/mindspore/models/tree/master/research/recommend/naml) |✅| ✅ |   |
| Recommendation | Recommender System | [ncf](https://gitee.com/mindspore/models/tree/master/research/recommend/ncf) |✅| ✅ |   |
| Recommendation | Recommender System | [tbnet](https://gitee.com/mindspore/models/tree/master/official/recommend/Tbnet) |✅| ✅ |   |
| Image | Image Classification | [alexnet](https://gitee.com/mindspore/models/tree/master/research/cv/Alexnet) |✅| ✅ |   |
| Image | Image Denoise | [brdnet](https://gitee.com/mindspore/models/tree/master/research/cv/brdnet) |✅|   |   |
| Image | Object Detection | [centerface](https://gitee.com/mindspore/models/tree/master/research/cv/centerface) |✅| ✅ | ✅ |
| Image | Image Classification | [cnn_direction_model](https://gitee.com/mindspore/models/tree/master/research/cv/cnn_direction_model) |✅| ✅ |   |
| Image | Scene Text Recognition | [cnnctc](https://gitee.com/mindspore/models/tree/master/research/cv/cnnctc) |✅| ✅ | ✅ |
| Image | Scene Text Recognition | [crnn](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN) |✅| ✅ | ✅ |
| Image | Scene Text Recognition | [crnn_seq2seq_ocr](https://gitee.com/mindspore/models/tree/master/research/cv/crnn_seq2seq_ocr) |✅|   |   |
| Image | Image Classification | [cspdarknet53](https://gitee.com/mindspore/models/tree/master/research/cv/cspdarknet53) |✅|   |   |
| Image | Object Detection | [ctpn](https://gitee.com/mindspore/models/tree/master/official/cv/CTPN) |✅| ✅ |   |
| Image | Object Detection | [darknet53](https://gitee.com/mindspore/models/tree/master/research/cv/darknet53) | | ✅ |   |
| Image | Semantic Segmentation | [deeplabv3](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P) |✅| ✅ | ✅ |
| Image | Text Detection | [deeptext](https://gitee.com/mindspore/models/tree/master/official/cv/DeepText) |✅| ✅ |   |
| Image | Image Classification | [densenet100](https://gitee.com/mindspore/models/tree/master/research/cv/densenet) |✅| ✅ |   |
| Image | Image Classification | [densenet121](https://gitee.com/mindspore/models/tree/master/research/cv/densenet) |✅| ✅ |   |
| Image | Depth Estimation | [depthnet](https://gitee.com/mindspore/models/tree/master/official/cv/DepthNet) |✅|   |   |
| Image | Image Denoise | [dncnn](https://gitee.com/mindspore/models/tree/master/research/cv/dncnn) | | ✅ |   |
| Image | Image Classification | [dpn](https://gitee.com/mindspore/models/tree/master/research/cv/dpn) |✅| ✅ |   |
| Image | Scene Text Detection | [east](https://gitee.com/mindspore/models/tree/master/research/cv/east) |✅| ✅ |   |
| Image | Image Classification | [efficientnet](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet) | | ✅ | ✅ |
| Image | Image Classification | [erfnet](https://gitee.com/mindspore/models/tree/master/research/cv/erfnet) |✅| ✅ |   |
| Image | Scene Text Recognition | [essay-recogination](https://gitee.com/mindspore/models/tree/master/research/cv/essay-recogination) | | ✅ |   |
| Image | Object Detection | [FasterRCNN_Inception_Resnetv2](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| Image | Object Detection | [FasterRCNN_ResNetV1.5_50](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| Image | Object Detection | [FasterRCNN_ResNetV1_101](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| Image | Object Detection | [FasterRCNN_ResNetV1_152](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| Image | Object Detection | [FasterRCNN_ResNetV1_50](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |✅| ✅ |   |
| Image | Semantic Segmentation | [fastscnn](https://gitee.com/mindspore/models/tree/master/research/cv/fastscnn) |✅|   |   |
| Image | Semantic Segmentation | [FCN8s](https://gitee.com/mindspore/models/tree/master/research/cv/FCN8s) |✅| ✅ |   |
| Image | Image Classification | [googlenet](https://gitee.com/mindspore/models/tree/master/research/cv/googlenet) |✅| ✅ |   |
| Image | Image Classification | [inceptionv3](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv3) |✅| ✅ | ✅ |
| Image | Image Classification | [inceptionv4](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4) |✅| ✅ | ✅ |
| Image | Image Denoise | [LearningToSeeInTheDark](https://gitee.com/mindspore/models/tree/master/research/cv/LearningToSeeInTheDark) |✅|   |   |
| Image | Image Classification | [lenet](https://gitee.com/mindspore/models/tree/master/research/cv/lenet) |✅| ✅ | ✅ |
| Image | Object Detection | [maskrcnn_resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_resnet50) |✅| ✅ |   |
| Image | Object Detection | [maskrcnn_mobilenetv1](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |✅| ✅ | ✅ |
| Image | Crowd Counting | [MCNN](https://gitee.com/mindspore/models/tree/master/research/cv/MCNN) |✅| ✅ |   |
| Image | Image Classification | [mobilenetv1](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv1) |✅| ✅ |   |
| Image | Image Classification | [mobilenetv2](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv2) |✅| ✅ | ✅ |
| Image | Image Classification | [mobilenetv3](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv3) |✅| ✅ | ✅ |
| Image | Image Classification | [nasnet](https://gitee.com/mindspore/models/tree/master/research/cv/nasnet) |✅| ✅ |   |
| Image | Image Quality Assessment | [nima](https://gitee.com/mindspore/models/tree/master/research/cv/nima) |✅| ✅ |   |
| Image | Point Cloud Model | [octsqueeze](https://gitee.com/mindspore/models/tree/master/official/cv/OctSqueeze) |✅| ✅ |   |
| Image | Keypoint Detection | [openpose](https://gitee.com/mindspore/models/tree/master/official/cv/OpenPose) |✅|   |   |
| Image | Defect Detection | [patchcore](https://gitee.com/mindspore/models/tree/master/official/cv/PatchCore) |✅| ✅ |   |
| Image | Camera Relocalization | [posenet](https://gitee.com/mindspore/models/tree/master/research/cv/PoseNet) |✅| ✅ |   |
| Image | Video Predictive Learning | [predrnn++](https://gitee.com/mindspore/models/tree/master/research/cv/predrnn++) |✅|   |   |
| Image | Scene Text Detection | [psenet](https://gitee.com/mindspore/models/tree/master/research/cv/psenet) |✅| ✅ |   |
| Image | Pose Estimation | [pvnet](https://gitee.com/mindspore/models/tree/master/official/cv/PVNet) |✅|   |   |
| Image | Optical Flow Estimation | [pwcnet](https://gitee.com/mindspore/models/tree/master/official/cv/PWCNet) |✅| ✅ |   |
| Image | Image Super Resolution | [RDN](https://gitee.com/mindspore/models/tree/master/research/cv/RDN) |✅| ✅ |   |
| Image | Image Classification | [resnet101](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| Image | Image Classification | [resnet152](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| Image | Image Classification | [resnet18](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| Image | Image Classification | [resnet34](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| Image | Image Classification | [resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| Image | Image Classification | [resnet50_thor](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ |   |
| Image | Image Classification | [resnext101](https://gitee.com/mindspore/models/tree/master/official/cv/ResNeXt) |✅| ✅ |   |
| Image | Image Classification | [resnext50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNeXt) |✅| ✅ |   |
| Image | Object Detection | [retinaface_resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaFace_ResNet50) | | ✅ |   |
| Image | Object Detection | [retinanet](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaNet) |✅| ✅ |   |
| Image | Image Classification | [se_resnext50](https://gitee.com/mindspore/models/tree/master/research/cv/SE_ResNeXt50) |✅|   |   |
| Image | Image Matting | [semantic_human_matting](https://gitee.com/mindspore/models/tree/master/official/cv/SemanticHumanMatting) |✅|   |   |
| Image | Image Classification | [se-resnet50](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet) |✅| ✅ | ✅ |
| Image | Image Classification | [shufflenetv1](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1) |✅| ✅ | ✅ |
| Image | Image Classification | [shufflenetv2](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv2) |✅| ✅ | ✅ |
| Image | Image Classification | [simclr](https://gitee.com/mindspore/models/tree/master/research/cv/simclr) |✅| ✅ |   |
| Image | Keypoint Detection | [simple_pose](https://gitee.com/mindspore/models/tree/master/research/cv/simple_pose) |✅| ✅ |   |
| Image | Object Detection | [sphereface](https://gitee.com/mindspore/models/tree/master/research/cv/sphereface) |✅| ✅ |   |
| Image | Image Classification | [squeezenet](https://gitee.com/mindspore/models/tree/master/research/cv/squeezenet) |✅| ✅ |   |
| Image | Image Classification | [SqueezeNet_Residual](https://gitee.com/mindspore/models/tree/master/research/cv/squeezenet) |✅| ✅ |   |
| Image | Image Super Resolution | [srcnn](https://gitee.com/mindspore/models/tree/master/research/cv/srcnn) |✅| ✅ |   |
| Image | Object Detection | [ssd_mobilenet-v1-fpn](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| Image | Object Detection | [ssd-mobilenet-v2](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| Image | Object Detection | [ssd-resnet50-fpn](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| Image | Object Detection | [ssd-vgg16](https://gitee.com/mindspore/models/tree/master/official/cv/SSD) |✅| ✅ | ✅ |
| Image | Defect Detection | [ssim-ae](https://gitee.com/mindspore/models/tree/master/official/cv/SSIM-AE) |✅|   |   |
| Image | Image Classification | [tinydarknet](https://gitee.com/mindspore/models/tree/master/research/cv/tinydarknet) |✅| ✅ | ✅ |
| Image | Semantic Segmentation | [UNet_nested](https://gitee.com/mindspore/models/tree/master/official/cv/Unet) |✅| ✅ |   |
| Image | Semantic Segmentation | [unet2d](https://gitee.com/mindspore/models/tree/master/official/cv/Unet) |✅| ✅ |   |
| Image | Semantic Segmentation | [unet3d](https://gitee.com/mindspore/models/tree/master/official/cv/Unet3d) |✅| ✅ |   |
| Image | Image Classification | [vgg16](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg16) |✅| ✅ | ✅ |
| Image | Image Classification | [vit](https://gitee.com/mindspore/models/tree/master/official/cv/VIT) |✅| ✅ |   |
| Image | Scene Text Recognition | [warpctc](https://gitee.com/mindspore/models/tree/master/research/cv/warpctc) |✅| ✅ |   |
| Image | Image Classification | [xception](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception) |✅| ✅ |   |
| Image | Object Detection | [yolov3_darknet53](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3) |✅| ✅ |   |
| Image | Object Detection | [yolov3_resnet18](https://gitee.com/mindspore/models/tree/master/research/cv/yolov3_resnet18) |✅|   |   |
| Image | Object Detection | [yolov4](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv4) |✅|   |   |
| Image | Object Detection | [yolov5s](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [deep_and_cross](https://gitee.com/mindspore/models/tree/master/research/recommend/deep_and_cross) | | ✅ |   |
| Recommendation | Click-Through Rate Prediction | [deepfm](https://gitee.com/mindspore/models/tree/master/official/recommend/DeepFM) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [fibinet](https://gitee.com/mindspore/models/tree/master/research/recommend/fibinet) | | ✅ |   |
| Recommendation | Click-Through Rate Prediction | [wide_and_deep](https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [wide_and_deep_multitable](https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep_Multitable) |✅| ✅ |   |
| Text | Natural Language Understanding | [bert_base](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| Text | Natural Language Understanding | [bert_bilstm_crf](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| Text | Natural Language Understanding | [bert_finetuning](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| Text | Natural Language Understanding | [bert_large](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅|   |   |
| Text | Natural Language Understanding | [bert_nezha](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |✅| ✅ |   |
| Text | Natural Language Understanding | [cpm](https://gitee.com/mindspore/models/tree/master/research/nlp/cpm) |✅| ✅ |   |
| Text | Dialogue | [dgu](https://gitee.com/mindspore/models/tree/master/research/nlp/dgu) |✅| ✅ |   |
| Text | Dialogue | [duconv](https://gitee.com/mindspore/models/tree/master/research/nlp/duconv) |✅| ✅ |   |
| Text | Emotion Classification | [emotect](https://gitee.com/mindspore/models/tree/master/research/nlp/emotect) |✅| ✅ |   |
| Text | Natural Language Understanding | [ernie](https://gitee.com/mindspore/models/tree/master/research/nlp/ernie) |✅| ✅ |   |
| Text | Natural Language Understanding | [fasttext](https://gitee.com/mindspore/models/tree/master/research/nlp/fasttext) |✅| ✅ |   |
| Text | Natural Language Understanding | [gnmt_v2](https://gitee.com/mindspore/models/tree/master/research/nlp/gnmt_v2) |✅| ✅ |   |
| Text | Natural Language Understanding | [gpt3](https://gitee.com/mindspore/models/tree/master/official/nlp/GPT) |✅|   |   |
| Text | Natural Language Understanding | [gru](https://gitee.com/mindspore/models/tree/master/official/nlp/GRU) |✅| ✅ |   |
| Text | Emotion Classification | [lstm](https://gitee.com/mindspore/models/tree/master/official/nlp/LSTM) |✅| ✅ |   |
| Text | Natural Language Understanding | [mass](https://gitee.com/mindspore/models/tree/master/research/nlp/mass) |✅| ✅ |   |
| Text | Pre Training | [pangu_alpha](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha) |✅| ✅ |   |
| Text | Natural Language Understanding | [textcnn](https://gitee.com/mindspore/models/tree/master/research/nlp/textcnn) |✅| ✅ |   |
| Text | Natural Language Understanding | [tinybert](https://gitee.com/mindspore/models/tree/master/research/nlp/tinybert) |✅| ✅ |   |
| Text | Natural Language Understanding | [transformer](https://gitee.com/mindspore/models/tree/master/official/nlp/Transformer) |✅| ✅ |   |
| Video | Object Tracking | [ADNet](https://gitee.com/mindspore/models/tree/master/research/cv/ADNet) |✅|   |   |
| Video | Video Classification | [c3d](https://gitee.com/mindspore/models/tree/master/official/cv/C3D) |✅| ✅ |   |
| Video | Object Tracking | [Deepsort](https://gitee.com/mindspore/models/tree/master/research/cv/Deepsort) |✅| ✅ |   |

### Research

|  Domain | Sub Domain    | Network  | Ascend  | GPU | CPU |
|:------   |:------| :-----------  |:------:   |:------:  |:-----: |
| 3D | 3D Reconstruction | [cmr](https://gitee.com/mindspore/models/tree/master/research/cv/cmr) | | ✅ |   |
| 3D | 3D Reconstruction | [DecoMR](https://gitee.com/mindspore/models/tree/master/research/cv/DecoMR) | | ✅ |   |
| 3D | 3D Reconstruction | [DeepLM](https://gitee.com/mindspore/models/tree/master/research/3d/DeepLM) | | ✅ |   |
| 3D | 3D Reconstruction | [eppmvsnet](https://gitee.com/mindspore/models/tree/master/research/cv/eppmvsnet) | | ✅ |   |
| 3D | 3D Object Detection | [pointpillars](https://gitee.com/mindspore/models/tree/master/research/cv/pointpillars) |✅| ✅ |   |
| Audio | Speech Recognition | [ctcmodel](https://gitee.com/mindspore/models/tree/master/research/audio/ctcmodel) |✅|   |   |
| Audio | Speech Recognition | [deepspeech2](https://gitee.com/mindspore/models/tree/master/official/audio/DeepSpeech2) | | ✅ |   |
| Audio | Keyword Spotting | [dscnn](https://gitee.com/mindspore/models/tree/master/research/audio/dscnn) |✅| ✅ |   |
| Audio | Speech Synthesis | [FastSpeech](https://gitee.com/mindspore/models/tree/master/research/audio/FastSpeech) | | ✅ |   |
| Audio | Audio Tagging | [fcn-4](https://gitee.com/mindspore/models/tree/master/research/audio/fcn-4) |✅| ✅ |   |
| Audio | Speech Recognition | [jasper](https://gitee.com/mindspore/models/tree/master/research/audio/jasper) |✅| ✅ |   |
| Audio | Speech Synthesis | [wavenet](https://gitee.com/mindspore/models/tree/master/research/audio/wavenet) |✅| ✅ |   |
| Graph Neural Network | Graph Classification | [dgcn](https://gitee.com/mindspore/models/tree/master/research/gnn/dgcn) |✅|   |   |
| Graph Neural Network | Text Classification | [hypertext](https://gitee.com/mindspore/models/tree/master/research/nlp/hypertext) |✅| ✅ |   |
| Graph Neural Network | Graph Classification | [sdne](https://gitee.com/mindspore/models/tree/master/research/gnn/sdne) |✅|   |   |
| Graph Neural Network | Social and Information Networks | [sgcn](https://gitee.com/mindspore/models/tree/master/research/gnn/sgcn) |✅| ✅ |   |
| Graph Neural Network | Text Classification | [textrcnn](https://gitee.com/mindspore/models/tree/master/research/nlp/textrcnn) |✅| ✅ |   |
| High Performance Computing | High Performance Computing | [deepbsde](https://gitee.com/mindspore/models/tree/master/research/hpc/deepbsde) | | ✅ |   |
| High Performance Computing | High Performance Computing | [molecular_dynamics](https://gitee.com/mindspore/models/tree/master/research/hpc/molecular_dynamics) |✅|   |   |
| High Performance Computing | High Performance Computing | [ocean_model](https://gitee.com/mindspore/models/tree/master/research/hpc/ocean_model) | | ✅ |   |
| High Performance Computing | High Performance Computing | [pafnucy](https://gitee.com/mindspore/models/tree/master/research/hpc/pafnucy) |✅| ✅ |   |
| High Performance Computing | High Performance Computing | [pfnn](https://gitee.com/mindspore/models/tree/master/research/hpc/pfnn) | | ✅ |   |
| High Performance Computing | High Performance Computing | [pinns](https://gitee.com/mindspore/models/tree/master/research/hpc/pinns) | | ✅ |   |
| Image | Image Classification | [3D_DenseNet](https://gitee.com/mindspore/models/tree/master/research/cv/3D_DenseNet) |✅| ✅ |   |
| Image | Semantic Segmentation | [3dcnn](https://gitee.com/mindspore/models/tree/master/research/cv/3dcnn) |✅| ✅ |   |
| Image | Semantic Segmentation | [adelaide_ea](https://gitee.com/mindspore/models/tree/master/research/cv/adelaide_ea) |✅|   |   |
| Image | Scene Text Detection | [advanced_east](https://gitee.com/mindspore/models/tree/master/research/cv/advanced_east) |✅| ✅ |   |
| Image | Style Transfer | [aecrnet](https://gitee.com/mindspore/models/tree/master/research/cv/aecrnet) |✅| ✅ |   |
| Image | Re-Identification | [AlignedReID](https://gitee.com/mindspore/models/tree/master/research/cv/AlignedReID) | | ✅ |   |
| Image | Re-Identification | [AlignedReID++](https://gitee.com/mindspore/models/tree/master/research/cv/AlignedReID++) |✅| ✅ |   |
| Image | Pose Estimation | [AlphaPose](https://gitee.com/mindspore/models/tree/master/research/cv/AlphaPose) |✅|   |   |
| Image | Style Transfer | [APDrawingGAN](https://gitee.com/mindspore/models/tree/master/research/cv/APDrawingGAN) |✅| ✅ |   |
| Image | Style Transfer | [ArbitraryStyleTransfer](https://gitee.com/mindspore/models/tree/master/research/cv/ArbitraryStyleTransfer) |✅| ✅ |   |
| Image | Object Detection | [arcface](https://gitee.com/mindspore/models/tree/master/official/cv/Arcface) |✅| ✅ |   |
| Image | Keypoint Detection | [ArtTrack](https://gitee.com/mindspore/models/tree/master/research/cv/ArtTrack) | | ✅ |   |
| Image | Style Transfer | [AttGAN](https://gitee.com/mindspore/models/tree/master/research/cv/AttGAN) |✅| ✅ |   |
| Image | Image Classification | [augvit](https://gitee.com/mindspore/models/tree/master/research/cv/augvit) | | ✅ |   |
| Image | Image Classification | [autoaugment](https://gitee.com/mindspore/models/tree/master/research/cv/autoaugment) |✅| ✅ |   |
| Image | Semantic Segmentation | [Auto-DeepLab](https://gitee.com/mindspore/models/tree/master/research/cv/Auto-DeepLab) |✅|   |   |
| Image | Neural Architecture Search | [AutoSlim](https://gitee.com/mindspore/models/tree/master/research/cv/AutoSlim) |✅| ✅ |   |
| Image | Image Classification | [AVA_cifar](https://gitee.com/mindspore/models/tree/master/research/cv/AVA_cifar) |✅| ✅ |   |
| Image | Image Classification | [AVA_hpa](https://gitee.com/mindspore/models/tree/master/research/cv/AVA_hpa) |✅| ✅ |   |
| Image | Image Classification | [cait](https://gitee.com/mindspore/models/tree/master/research/cv/cait) |✅| ✅ |   |
| Image | Object Detection | [CascadeRCNN](https://gitee.com/mindspore/models/tree/master/research/cv/CascadeRCNN) |✅| ✅ |   |
| Image | Image Classification | [CBAM](https://gitee.com/mindspore/models/tree/master/research/cv/CBAM) |✅|   |   |
| Image | Image Classification | [cct](https://gitee.com/mindspore/models/tree/master/research/cv/cct) |✅| ✅ |   |
| Image | Keypoint Detection | [centernet](https://gitee.com/mindspore/models/tree/master/research/cv/centernet) |✅|   | ✅ |
| Image | Keypoint Detection | [centernet_det](https://gitee.com/mindspore/models/tree/master/research/cv/centernet_det) |✅|   |   |
| Image | Keypoint Detection | [centernet_resnet101](https://gitee.com/mindspore/models/tree/master/research/cv/centernet_resnet101) |✅| ✅ |   |
| Image | Keypoint Detection | [centernet_resnet50_v1](https://gitee.com/mindspore/models/tree/master/research/cv/centernet_resnet50_v1) |✅|   |   |
| Image | Image Generation | [CGAN](https://gitee.com/mindspore/models/tree/master/research/cv/CGAN) |✅| ✅ |   |
| Image | Image Classification | [convnext](https://gitee.com/mindspore/models/tree/master/research/cv/convnext) |✅| ✅ |   |
| Image | Image Super Resolution | [csd](https://gitee.com/mindspore/models/tree/master/research/cv/csd) |✅| ✅ |   |
| Image | Image Generation | [CTSDG](https://gitee.com/mindspore/models/tree/master/research/cv/CTSDG) |   | ✅ |   |
| Image | Style Transfer | [CycleGAN](https://gitee.com/mindspore/models/tree/master/official/cv/CycleGAN) |✅| ✅ |   |
| Image | Image Super Resolution | [DBPN](https://gitee.com/mindspore/models/tree/master/research/cv/DBPN) |✅|   |   |
| Image | Image Super Resolution | [DBPN_GAN](https://gitee.com/mindspore/models/tree/master/research/cv/DBPN) |✅|   |   |
| Image | Image Generation | [dcgan](https://gitee.com/mindspore/models/tree/master/research/cv/dcgan) |✅| ✅ |   |
| Image | Re-Identification | [DDAG](https://gitee.com/mindspore/models/tree/master/research/cv/DDAG) |✅| ✅ |   |
| Image | Semantic Segmentation | [DDM](https://gitee.com/mindspore/models/tree/master/research/cv/DDM) |✅|   |   |
| Image | Semantic Segmentation | [DDRNet](https://gitee.com/mindspore/models/tree/master/research/cv/DDRNet) |✅| ✅ |   |
| Image | Object Detection | [DeepID](https://gitee.com/mindspore/models/tree/master/research/cv/DeepID) |✅| ✅ |   |
| Image | Semantic Segmentation | [deeplabv3plus](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P) |✅| ✅ |   |
| Image | Image Retrieval | [delf](https://gitee.com/mindspore/models/tree/master/research/cv/delf) |✅|   |   |
| Image | Zero-Shot Learning | [dem](https://gitee.com/mindspore/models/tree/master/research/cv/dem) |✅| ✅ |   |
| Image | Object Detection | [detr](https://gitee.com/mindspore/models/tree/master/research/cv/detr) |✅| ✅ |   |
| Image | Semantic Segmentation | [dgcnet_res101](https://gitee.com/mindspore/models/tree/master/research/cv/dgcnet_res101) | | ✅ |   |
| Image | Instance Segmentation | [dlinknet](https://gitee.com/mindspore/models/tree/master/research/cv/dlinknet) |✅|   |   |
| Image | Image Denoise | [DnCNN](https://gitee.com/mindspore/models/tree/master/research/cv/DnCNN) |✅|   |   |
| Image | Image Classification | [dnet_nas](https://gitee.com/mindspore/models/tree/master/research/cv/dnet_nas) |✅|   |   |
| Image | Image Classification | [DRNet](https://gitee.com/mindspore/models/tree/master/research/cv/DRNet) |✅| ✅ |   |
| Image | Image Super Resolution | [EDSR](https://gitee.com/mindspore/models/tree/master/official/cv/EDSR) |✅|   |   |
| Image | Object Detection | [EfficientDet_d0](https://gitee.com/mindspore/models/tree/master/research/cv/EfficientDet_d0) |✅|   |   |
| Image | Image Classification | [efficientnet-b0](https://gitee.com/mindspore/models/tree/master/research/cv/efficientnet-b0) |✅|   |   |
| Image | Image Classification | [efficientnet-b1](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1) |✅|   |   |
| Image | Image Classification | [efficientnet-b2](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b2) |✅| ✅ |   |
| Image | Image Classification | [efficientnet-b3](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b3) |✅| ✅ |   |
| Image | Image Classification | [efficientnetv2](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnetv2) |✅|   |   |
| Image | Salient Object Detection | [EGnet](https://gitee.com/mindspore/models/tree/master/research/cv/EGnet) |✅| ✅ |   |
| Image | Semantic Segmentation | [E-NET](https://gitee.com/mindspore/models/tree/master/research/cv/E-NET) |✅| ✅ |   |
| Image | Image Super Resolution | [esr_ea](https://gitee.com/mindspore/models/tree/master/research/cv/esr_ea) |✅| ✅ |   |
| Image | Image Super Resolution | [ESRGAN](https://gitee.com/mindspore/models/tree/master/research/cv/ESRGAN) |✅| ✅ |   |
| Image | Image Classification | [FaceAttribute](https://gitee.com/mindspore/models/tree/master/research/cv/FaceAttribute) |✅| ✅ |   |
| Image | Object Detection | [faceboxes](https://gitee.com/mindspore/models/tree/master/research/cv/faceboxes) |✅|   |   |
| Image | Object Detection | [FaceDetection](https://gitee.com/mindspore/models/tree/master/research/cv/FaceDetection) |✅| ✅ |   |
| Image | Face Recognition | [FaceNet](https://gitee.com/mindspore/models/tree/master/research/cv/FaceNet) |✅| ✅ |   |
| Image | Image Classification | [FaceQualityAssessment](https://gitee.com/mindspore/models/tree/master/research/cv/FaceQualityAssessment) |✅| ✅ | ✅ |
| Image | Object Detection | [FaceRecognition](https://gitee.com/mindspore/models/tree/master/official/cv/FaceRecognition) |✅| ✅ |   |
| Image | Object Detection | [FaceRecognitionForTracking](https://gitee.com/mindspore/models/tree/master/research/cv/FaceRecognitionForTracking) |✅| ✅ | ✅ |
| Image | Object Detection | [faster_rcnn_dcn](https://gitee.com/mindspore/models/tree/master/research/cv/faster_rcnn_dcn) |✅| ✅ |   |
| Image | Image Matting | [FCANet](https://gitee.com/mindspore/models/tree/master/research/cv/FCANet) |✅|   |   |
| Image | Image Classification | [FDA-BNN](https://gitee.com/mindspore/models/tree/master/research/cv/FDA-BNN) |✅| ✅ |   |
| Image | Image Classification | [fishnet99](https://gitee.com/mindspore/models/tree/master/research/cv/fishnet99) |✅| ✅ |   |
| Image | Optical Flow Estimation | [flownet2](https://gitee.com/mindspore/models/tree/master/research/cv/flownet2) |✅|   |   |
| Image | Image Generation | [gan](https://gitee.com/mindspore/models/tree/master/research/cv/gan) |✅| ✅ |   |
| Image | Image Classification | [GENet_Res50](https://gitee.com/mindspore/models/tree/master/research/cv/GENet_Res50) |✅|   |   |
| Image | Image Classification | [ghostnet](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet) |✅|   |   |
| Image | Image Classification | [ghostnet_d](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet_d) |✅| ✅ |   |
| Image | Image Classification | [glore_res200](https://gitee.com/mindspore/models/tree/master/research/cv/glore_res) |✅| ✅ |   |
| Image | Image Classification | [glore_res50](https://gitee.com/mindspore/models/tree/master/research/cv/glore_res) |✅| ✅ |   |
| Image | Image Classification | [hardnet](https://gitee.com/mindspore/models/tree/master/research/cv/hardnet) |✅| ✅ |   |
| Image | Edge Detection | [hed](https://gitee.com/mindspore/models/tree/master/research/cv/hed) |✅| ✅ |   |
| Image | Image Generation | [HiFaceGAN](https://gitee.com/mindspore/models/tree/master/research/cv/HiFaceGAN) | | ✅ |   |
| Image | Image Classification | [HourNAS](https://gitee.com/mindspore/models/tree/master/research/cv/HourNAS) | | ✅ |   |
| Image | Image Classification | [HRNetW48_cls](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_cls) |✅| ✅ |   |
| Image | Semantic Segmentation | [HRNetW48_seg](https://gitee.com/mindspore/models/tree/master/research/cv/HRNetW48_seg) |✅|   |   |
| Image | Image Classification | [ibnnet](https://gitee.com/mindspore/models/tree/master/research/cv/ibnnet) |✅| ✅ |   |
| Image | Semantic Segmentation | [ICNet](https://gitee.com/mindspore/models/tree/master/research/cv/ICNet) |✅|   |   |
| Image | Image Classification | [inception_resnet_v2](https://gitee.com/mindspore/models/tree/master/research/cv/inception_resnet_v2) |✅| ✅ |   |
| Image | Image Classification | [Inceptionv2](https://gitee.com/mindspore/models/tree/master/research/cv/Inception-v2) |✅| ✅ |   |
| Image | Image Matting | [IndexNet](https://gitee.com/mindspore/models/tree/master/research/cv/IndexNet) | | ✅ |   |
| Image | Image Generation | [IPT](https://gitee.com/mindspore/models/tree/master/research/cv/IPT) |✅|   |   |
| Image | Image Super Resolution | [IRN](https://gitee.com/mindspore/models/tree/master/research/cv/IRN) |✅| ✅ |   |
| Image | Image Classification | [ISyNet](https://gitee.com/mindspore/models/tree/master/research/cv/ISyNet) |✅| ✅ |   |
| Image | Image Classification | [ivpf](https://gitee.com/mindspore/models/tree/master/research/cv/ivpf) | | ✅ |   |
| Image | Image Denoise | [LearningToSeeInTheDark](https://gitee.com/mindspore/models/tree/master/research/cv/LearningToSeeInTheDark) |✅|   |   |
| Image | Meta Learning | [LEO](https://gitee.com/mindspore/models/tree/master/research/cv/LEO) |✅| ✅ |   |
| Image | Object Detection | [LightCNN](https://gitee.com/mindspore/models/tree/master/research/cv/LightCNN) |✅| ✅ | ✅ |
| Image | Image Super Resolution | [lite-hrnet](https://gitee.com/mindspore/models/tree/master/research/cv/lite-hrnet) | | ✅ |   |
| Image | Image Classification | [lresnet100e_ir](https://gitee.com/mindspore/models/tree/master/research/cv/lresnet100e_ir) | | ✅ |   |
| Image | Object Detection | [m2det](https://gitee.com/mindspore/models/tree/master/research/cv/m2det) | | ✅ |   |
| Image | Autoencoder | [mae](https://gitee.com/mindspore/models/tree/master/official/cv/MAE) |✅| ✅ |   |
| Image | Meta Learning | [MAML](https://gitee.com/mindspore/models/tree/master/research/cv/MAML) |✅| ✅ |   |
| Image | Scene Text Recognition | [ManiDP](https://gitee.com/mindspore/models/tree/master/research/cv/ManiDP) | | ✅ |   |
| Image | Face Recognition | [MaskedFaceRecognition](https://gitee.com/mindspore/models/tree/master/research/cv/MaskedFaceRecognition) |✅|   |   |
| Image | Meta Learning | [meta-baseline](https://gitee.com/mindspore/models/tree/master/research/cv/meta-baseline) |✅| ✅ |   |
| Image | Re-Identification | [MGN](https://gitee.com/mindspore/models/tree/master/research/cv/MGN) |✅| ✅ |   |
| Image | Depth Estimation | [midas](https://gitee.com/mindspore/models/tree/master/research/cv/midas) |✅| ✅ |   |
| Image | Image Denoise | [MIMO-UNet](https://gitee.com/mindspore/models/tree/master/research/cv/MIMO-UNet) | | ✅ |   |
| Image | Image Classification | [mnasnet](https://gitee.com/mindspore/models/tree/master/research/cv/mnasnet) |✅| ✅ |   |
| Image | Image Classification | [mobilenetv3_large](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetv3_large) |✅|   | ✅ |
| Image | Image Classification | [mobilenetV3_small_x1_0](https://gitee.com/mindspore/models/tree/master/research/cv/mobilenetV3_small_x1_0) |✅| ✅ | ✅ |
| Image | Image Classification | [MultiTaskNet](https://gitee.com/mindspore/models/tree/master/research/cv/PAMTRI/MultiTaskNet) |✅| ✅ |   |
| Image | Re-Identification | [MVD](https://gitee.com/mindspore/models/tree/master/research/cv/MVD) |✅| ✅ |   |
| Image | Object Detection | [nas-fpn](https://gitee.com/mindspore/models/tree/master/research/cv/nas-fpn) |✅|   |   |
| Image | Image Denoise | [Neighbor2Neighbor](https://gitee.com/mindspore/models/tree/master/research/cv/Neighbor2Neighbor) |✅| ✅ |   |
| Image | Image Classification | [NFNet](https://gitee.com/mindspore/models/tree/master/research/cv/NFNet) |✅| ✅ |   |
| Image | Image Quality Assessment | [nima_vgg16](https://gitee.com/mindspore/models/tree/master/research/cv/nima_vgg16) | | ✅ |   |
| Image | Semantic Segmentation | [nnUNet](https://gitee.com/mindspore/models/tree/master/research/cv/nnUNet) |✅| ✅ |   |
| Image | Image Classification | [ntsnet](https://gitee.com/mindspore/models/tree/master/research/cv/ntsnet) |✅| ✅ |   |
| Image | Semantic Segmentation | [OCRNet](https://gitee.com/mindspore/models/tree/master/official/cv/OCRNet) |✅| ✅ |   |
| Image | Re-Identification | [osnet](https://gitee.com/mindspore/models/tree/master/research/cv/osnet) |✅| ✅ |   |
| Image | Salient Object Detection | [PAGENet](https://gitee.com/mindspore/models/tree/master/research/cv/PAGENet) |✅| ✅ |   |
| Image | Image Retrieval | [pcb](https://gitee.com/mindspore/models/tree/master/research/cv/pcb_rpp) | | ✅ |   |
| Image | Image Retrieval | [pcb](https://gitee.com/mindspore/models/tree/master/research/cv/pcb_rpp) | | ✅ |   |
| Image | Image Retrieval | [pcb_rpp](https://gitee.com/mindspore/models/tree/master/research/cv/pcb_rpp) | | ✅ |   |
| Image | Image Classification | [PDarts](https://gitee.com/mindspore/models/tree/master/research/cv/PDarts) |✅| ✅ |   |
| Image | Image Generation | [PGAN](https://gitee.com/mindspore/models/tree/master/research/cv/PGAN) |✅| ✅ |   |
| Image | Image Generation | [Pix2Pix](https://gitee.com/mindspore/models/tree/master/research/cv/Pix2Pix) |✅| ✅ |   |
| Image | Image Super Resolution | [Pix2PixHD](https://gitee.com/mindspore/models/tree/master/official/cv/Pix2PixHD) |✅|   |   |
| Image | Image Classification | [pnasnet](https://gitee.com/mindspore/models/tree/master/research/cv/pnasnet) |✅| ✅ |   |
| Image | Point Cloud Model | [pointnet](https://gitee.com/mindspore/models/tree/master/official/cv/PointNet) |✅| ✅ |   |
| Image | Point Cloud Model | [pointnet2](https://gitee.com/mindspore/models/tree/master/official/cv/PointNet2) |✅| ✅ |   |
| Image | Image Classification | [PoseEstNet](https://gitee.com/mindspore/models/tree/master/research/cv/PAMTRI/PoseEstNet) |✅| ✅ |   |
| Image | Image Classification | [ProtoNet](https://gitee.com/mindspore/models/tree/master/research/cv/ProtoNet) |✅| ✅ |   |
| Image | Image Classification | [proxylessnas](https://gitee.com/mindspore/models/tree/master/research/cv/proxylessnas) |✅| ✅ |   |
| Image | Semantic Segmentation | [PSPNet](https://gitee.com/mindspore/models/tree/master/research/cv/PSPNet) |✅|   |   |
| Image | Salient Object Detection | [ras](https://gitee.com/mindspore/models/tree/master/research/cv/ras) |✅| ✅ |   |
| Image | Image Super Resolution | [RCAN](https://gitee.com/mindspore/models/tree/master/research/cv/RCAN) |✅|   |   |
| Image | Object Detection | [rcnn](https://gitee.com/mindspore/models/tree/master/research/cv/rcnn) |✅| ✅ |   |
| Image | Image Super Resolution | [REDNet30](https://gitee.com/mindspore/models/tree/master/research/cv/REDNet30) |✅| ✅ |   |
| Image | Object Detection | [RefineDet](https://gitee.com/mindspore/models/tree/master/research/cv/RefineDet) |✅| ✅ |   |
| Image | Semantic Segmentation | [RefineNet](https://gitee.com/mindspore/models/tree/master/research/cv/RefineNet) |✅| ✅ |   |
| Image | Re-Identification | [ReIDStrongBaseline](https://gitee.com/mindspore/models/tree/master/research/cv/ReIDStrongBaseline) |✅| ✅ |   |
| Image | Image Classification | [relationnet](https://gitee.com/mindspore/models/tree/master/research/cv/relationnet) |✅| ✅ |   |
| Image | Image Classification | [renas](https://gitee.com/mindspore/models/tree/master/research/cv/renas) |✅| ✅ | ✅ |
| Image | Semantic Segmentation | [repvgg](https://gitee.com/mindspore/models/tree/master/research/cv/repvgg) |✅| ✅ |   |
| Image | Semantic Segmentation | [res2net_deeplabv3](https://gitee.com/mindspore/models/tree/master/research/cv/res2net_deeplabv3) |✅|   | ✅ |
| Image | Object Detection | [res2net_faster_rcnn](https://gitee.com/mindspore/models/tree/master/research/cv/res2net_faster_rcnn) |✅| ✅ |   |
| Image | Object Detection | [res2net_yolov3](https://gitee.com/mindspore/models/tree/master/research/cv/res2net_yolov3) |✅| ✅ |   |
| Image | Image Classification | [res2net101](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| Image | Image Classification | [res2net152](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| Image | Image Classification | [res2net50](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| Image | Image Classification | [ResNeSt50](https://gitee.com/mindspore/models/tree/master/research/cv/ResNeSt50) |✅| ✅ |   |
| Image | Image Classification | [resnet50_adv_pruning](https://gitee.com/mindspore/models/tree/master/research/cv/resnet50_adv_pruning) |✅| ✅ |   |
| Image | Image Classification | [resnet50_bam](https://gitee.com/mindspore/models/tree/master/research/cv/resnet50_bam) |✅| ✅ |   |
| Image | Image Classification | [ResNet50-Quadruplet](https://gitee.com/mindspore/models/tree/master/research/cv/metric_learn) |✅| ✅ |   |
| Image | Image Classification | [ResNet50-Triplet](https://gitee.com/mindspore/models/tree/master/research/cv/metric_learn) |✅| ✅ |   |
| Image | Image Classification | [ResnetV2_101](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2) |✅| ✅ |   |
| Image | Image Classification | [ResnetV2_152](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2) |✅| ✅ |   |
| Image | Image Classification | [ResnetV2_50](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2) |✅| ✅ |   |
| Image | Image Classification | [resnetv2_50_frn](https://gitee.com/mindspore/models/tree/master/research/cv/resnetv2_50_frn) |✅| ✅ |   |
| Image | Image Classification | [resnext152_64x4d](https://gitee.com/mindspore/models/tree/master/research/cv/resnext152_64x4d) |✅| ✅ |   |
| Image | Object Detection | [retinaface_mobilenet0.25](https://gitee.com/mindspore/models/tree/master/research/cv/retinaface) |✅| ✅ |   |
| Image | Object Detection | [retinanet_resnet101](https://gitee.com/mindspore/models/tree/master/research/cv/retinanet_resnet101) |✅| ✅ |   |
| Image | Object Detection | [retinanet_resnet152](https://gitee.com/mindspore/models/tree/master/research/cv/retinanet_resnet152) |✅| ✅ |   |
| Image | Object Detection | [rfcn](https://gitee.com/mindspore/models/tree/master/research/cv/rfcn) | | ✅ |   |
| Image | Image Classification | [SE_ResNeXt50](https://gitee.com/mindspore/models/tree/master/research/cv/SE_ResNeXt50) |✅|   |   |
| Image | Image Classification | [senet_resnet101](https://gitee.com/mindspore/models/tree/master/research/cv/SE-Net) |✅| ✅ | ✅ |
| Image | Image Classification | [senet_resnet50](https://gitee.com/mindspore/models/tree/master/research/cv/SE-Net) |✅| ✅ | ✅ |
| Image | Image Classification | [se-res2net50](https://gitee.com/mindspore/models/tree/master/research/cv/res2net) |✅| ✅ |   |
| Image | Image Classification | [S-GhostNet](https://gitee.com/mindspore/models/tree/master/research/cv/S-GhostNet) |✅|   |   |
| Image | Pose Estimation | [simple_baselines](https://gitee.com/mindspore/models/tree/master/research/cv/simple_baselines) |✅| ✅ |   |
| Image | Image Generation | [SinGAN](https://gitee.com/mindspore/models/tree/master/research/cv/SinGAN) |✅|   |   |
| Image | Image Classification | [single_path_nas](https://gitee.com/mindspore/models/tree/master/research/cv/single_path_nas) |✅| ✅ |   |
| Image | Image Classification | [sknet](https://gitee.com/mindspore/models/tree/master/research/cv/sknet) |✅| ✅ | ✅ |
| Image | Image Classification | [snn_mlp](https://gitee.com/mindspore/models/tree/master/research/cv/snn_mlp) | | ✅ |   |
| Image | Object Detection | [Spnas](https://gitee.com/mindspore/models/tree/master/research/cv/Spnas) |✅|   |   |
| Image | Image Classification | [SPPNet](https://gitee.com/mindspore/models/tree/master/research/cv/SPPNet) |✅| ✅ |   |
| Image | Image Classification | [squeezenet](https://gitee.com/mindspore/models/tree/master/research/cv/squeezenet) |✅| ✅ |   |
| Image | Image Super Resolution | [sr_ea](https://gitee.com/mindspore/models/tree/master/research/cv/sr_ea) |✅|   |   |
| Image | Image Super Resolution | [SRGAN](https://gitee.com/mindspore/models/tree/master/research/cv/SRGAN) |✅| ✅ |   |
| Image | Image Classification | [ssc_resnet50](https://gitee.com/mindspore/models/tree/master/research/cv/ssc_resnet50) |✅| ✅ |   |
| Image | Object Detection | [ssd_ghostnet](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_ghostnet) |✅| ✅ | ✅ |
| Image | Object Detection | [ssd_inception_v2](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_inception_v2) | | ✅ | ✅ |
| Image | Object Detection | [ssd_inceptionv2](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_inceptionv2) |✅|   |   |
| Image | Object Detection | [ssd_mobilenetV2](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_mobilenetV2) |✅| ✅ | ✅ |
| Image | Object Detection | [ssd_mobilenetV2_FPNlite](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_mobilenetV2_FPNlite) |✅| ✅ | ✅ |
| Image | Object Detection | [ssd_resnet_34](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_resnet_34) | | ✅ |   |
| Image | Object Detection | [ssd_resnet34](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_resnet34) |✅|   | ✅ |
| Image | Object Detection | [ssd_resnet50](https://gitee.com/mindspore/models/tree/master/research/cv/ssd_resnet50) |✅|   |   |
| Image | Pose Estimation | [StackedHourglass](https://gitee.com/mindspore/models/tree/master/research/cv/StackedHourglass) |✅|   |   |
| Image | Image Generation | [StarGAN](https://gitee.com/mindspore/models/tree/master/research/cv/StarGAN) |✅| ✅ |   |
| Image | Image Generation | [STGAN](https://gitee.com/mindspore/models/tree/master/research/cv/STGAN) |✅| ✅ |   |
| Image | Traffic Prediction | [stgcn](https://gitee.com/mindspore/models/tree/master/research/cv/stgcn) |✅| ✅ |   |
| Image | Image Classification | [stpm](https://gitee.com/mindspore/models/tree/master/official/cv/STPM) |✅| ✅ |   |
| Image | Image Classification | [swin_transformer](https://gitee.com/mindspore/models/tree/master/official/cv/SwinTransformer) |✅| ✅ |   |
| Image | Temporal Localization | [tall](https://gitee.com/mindspore/models/tree/master/research/cv/tall) |✅|   |   |
| Image | Image Classification | [TCN](https://gitee.com/mindspore/models/tree/master/research/cv/TCN) |✅| ✅ |   |
| Image | Scene Text Detection | [textfusenet](https://gitee.com/mindspore/models/tree/master/research/cv/textfusenet) |✅|   |   |
| Image | Traffic Prediction | [tgcn](https://gitee.com/mindspore/models/tree/master/research/cv/tgcn) |✅| ✅ |   |
| Image | Image Classification | [tinynet](https://gitee.com/mindspore/models/tree/master/research/cv/tinynet) | | ✅ |   |
| Image | Image Classification | [TNT](https://gitee.com/mindspore/models/tree/master/research/cv/TNT) |✅| ✅ |   |
| Image | Object Detection | [u2net](https://gitee.com/mindspore/models/tree/master/research/cv/u2net) |✅| ✅ |   |
| Image | Image Generation | [U-GAT-IT](https://gitee.com/mindspore/models/tree/master/research/cv/U-GAT-IT) |✅| ✅ |   |
| Image | Semantic Segmentation | [UNet3+](https://gitee.com/mindspore/models/tree/master/research/cv/UNet3+) |✅| ✅ |   |
| Image | Re-Identification | [VehicleNet](https://gitee.com/mindspore/models/tree/master/research/cv/VehicleNet) |✅|   |   |
| Image | Image Classification | [vgg19](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg19) |✅| ✅ |   |
| Image | Image Classification | [ViG](https://gitee.com/mindspore/models/tree/master/research/cv/ViG) |✅| ✅ |   |
| Image | Image Classification | [vit_cifar](https://gitee.com/mindspore/models/tree/master/research/cv/vit_base) |✅| ✅ |   |
| Image | Semantic Segmentation | [vnet](https://gitee.com/mindspore/models/tree/master/research/cv/vnet) |✅| ✅ |   |
| Image | Image Classification | [wave_mlp](https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp) |✅| ✅ |   |
| Image | Image Super Resolution | [wdsr](https://gitee.com/mindspore/models/tree/master/research/cv/wdsr) |✅| ✅ |   |
| Image | Image Generation | [wgan](https://gitee.com/mindspore/models/tree/master/official/cv/WGAN) |✅|   |   |
| Image | Image Classification | [wideresnet](https://gitee.com/mindspore/models/tree/master/research/cv/wideresnet) |✅| ✅ |   |
| Image | Instance Segmentation | [Yolact++](https://gitee.com/mindspore/models/tree/master/research/cv/Yolact++) |✅|   |   |
| Image | Object Detection | [yolov3_tiny](https://gitee.com/mindspore/models/tree/master/research/cv/yolov3_tiny) |✅| ✅ |   |
| Image | Object Detection | [yolox](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOX) |✅|   |   |
| Multi Modal | Multi Modal | [opt](https://gitee.com/mindspore/models/tree/master/research/mm/opt) |✅| ✅ |   |
| Multi Modal | Multi Modal | [TokenFusion](https://gitee.com/mindspore/models/tree/master/research/cv/TokenFusion) |✅| ✅ |   |
| Multi Modal | Multi Modal | [wukong](https://gitee.com/mindspore/models/tree/master/research/mm/wukong) |✅|   |   |
| Recommendation | Click-Through Rate Prediction | [autodis](https://gitee.com/mindspore/models/tree/master/research/recommend/autodis) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [DIEN](https://gitee.com/mindspore/models/tree/master/research/recommend/DIEN) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [dlrm](https://gitee.com/mindspore/models/tree/master/research/recommend/dlrm) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [EDCN](https://gitee.com/mindspore/models/tree/master/research/recommend/EDCN) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [Fat-DeepFFM](https://gitee.com/mindspore/models/tree/master/research/recommend/Fat-DeepFFM) |✅| ✅ |   |
| Recommendation | Click-Through Rate Prediction | [mmoe](https://gitee.com/mindspore/models/tree/master/research/recommend/mmoe) |✅| ✅ |   |
| Text | Natural Language Understanding | [albert](https://gitee.com/mindspore/models/tree/master/research/nlp/albert) |✅| ✅ |   |
| Text | Emotion Classification | [atae_lstm](https://gitee.com/mindspore/models/tree/master/research/nlp/atae_lstm) |✅| ✅ |   |
| Text | Dialogue | [dam](https://gitee.com/mindspore/models/tree/master/research/nlp/dam) |✅|   |   |
| Text | Language Model | [gpt2](https://gitee.com/mindspore/models/tree/master/research/nlp/gpt2) |✅|   |   |
| Text | Knowledge Graph Embedding | [hake](https://gitee.com/mindspore/models/tree/master/research/nlp/hake) | | ✅ |   |
| Text | Natural Language Understanding | [ktnet](https://gitee.com/mindspore/models/tree/master/research/nlp/ktnet) |✅| ✅ |   |
| Text | Named Entity Recognition | [lstm_crf](https://gitee.com/mindspore/models/tree/master/research/nlp/lstm_crf) |✅|   |   |
| Text | Natural Language Understanding | [luke](https://gitee.com/mindspore/models/tree/master/research/nlp/luke) |✅| ✅ |   |
| Text | Knowledge Graph Embedding | [rotate](https://gitee.com/mindspore/models/tree/master/research/nlp/rotate) |✅| ✅ |   |
| Text | Emotion Classification | [senta](https://gitee.com/mindspore/models/tree/master/research/nlp/senta) |✅| ✅ |   |
| Text | Machine Translation | [seq2seq](https://gitee.com/mindspore/models/tree/master/research/nlp/seq2seq) |✅|   |   |
| Text | Word Embedding | [skipgram](https://gitee.com/mindspore/models/tree/master/research/nlp/skipgram) |✅| ✅ |   |
| Text | Machine Translation | [speech_transformer](https://gitee.com/mindspore/models/tree/master/research/nlp/speech_transformer) |✅|   |   |
| Text | Pre Training | [ternarybert](https://gitee.com/mindspore/models/tree/master/research/nlp/ternarybert) |✅| ✅ |   |
| Text | Natural Language Understanding | [tprr](https://gitee.com/mindspore/models/tree/master/research/nlp/tprr) |✅|   |   |
| Text | Natural Language Understanding | [transformer_xl](https://gitee.com/mindspore/models/tree/master/research/nlp/transformer_xl) |✅| ✅ |   |
| Text | Knowledge Graph Embedding | [transX](https://gitee.com/mindspore/models/tree/master/research/nlp/transX) | | ✅ |   |
| Video | Video Classification | [AttentionCluster](https://gitee.com/mindspore/models/tree/master/research/cv/AttentionCluster) |✅| ✅ |   |
| Video | Others | [DYR](https://gitee.com/mindspore/models/tree/master/research/nlp/DYR) |✅|   |   |
| Video | Video Classification | [ecolite](https://gitee.com/mindspore/models/tree/master/research/cv/ecolite) |✅|   |   |
| Video | Object Tracking | [fairmot](https://gitee.com/mindspore/models/tree/master/research/cv/fairmot) |✅| ✅ |   |
| Video | Video Classification | [I3D](https://gitee.com/mindspore/models/tree/master/research/cv/I3D) |✅|   |   |
| Video | Object Tracking | [JDE](https://gitee.com/mindspore/models/tree/master/research/cv/JDE) | | ✅ |   |
| Video | video Segment | [OSVOS](https://gitee.com/mindspore/models/tree/master/research/cv/OSVOS) | | ✅ |   |
| Video | Video Classification | [r2plus1d](https://gitee.com/mindspore/models/tree/master/research/cv/r2plus1d) |✅| ✅ |   |
| Video | video Super Resolution | [rbpn](https://gitee.com/mindspore/models/tree/master/research/cv/rbpn) |✅|   |   |
| Video | Video Classification | [resnet3d](https://gitee.com/mindspore/models/tree/master/research/cv/resnet3d) |✅|   |   |
| Video | Object Tracking | [SiamFC](https://gitee.com/mindspore/models/tree/master/research/cv/SiamFC) |✅|   |   |
| Video | Object Tracking | [siamRPN](https://gitee.com/mindspore/models/tree/master/research/cv/siamRPN) |✅| ✅ |   |
| Video | Video Classification | [slowfast](https://gitee.com/mindspore/models/tree/master/research/cv/slowfast) |✅| ✅ |   |
| Video | Video Classification | [stnet](https://gitee.com/mindspore/models/tree/master/research/cv/stnet) |✅|   |   |
| Video | Object Tracking | [tracktor](https://gitee.com/mindspore/models/tree/master/research/cv/tracktor) | | ✅ |   |
| Video | Object Tracking | [tracktor++](https://gitee.com/mindspore/models/tree/master/research/cv/tracktor++) |✅| ✅ |   |
| Video | Video Classification | [trn](https://gitee.com/mindspore/models/tree/master/research/cv/trn) | | ✅ |   |
| Video | Video Classification | [tsm](https://gitee.com/mindspore/models/tree/master/research/cv/tsm) |✅| ✅ |   |
| Video | Video Classification | [tsn](https://gitee.com/mindspore/models/tree/master/research/cv/tsn) |✅| ✅ |   |

- [Community](https://gitee.com/mindspore/models/tree/master/community)

## Announcements

### 2021.9.15 Set up repository `models`

`models` comes from the directory `model_zoo` of repository [mindspore](https://gitee.com/mindspore/mindspore). This new repository doesn't contain any history of commits about the directory `model_zoo` in `mindspore`, you could refer to the repository `mindspore` for the past commits.

## Related Website

Here is the ModelZoo for MindSpore which support different devices including Ascend, GPU, CPU and mobile.

If you are looking for exclusive models only for Ascend using different ML platform, you could refer to [Ascend ModelZoo](https://hiascend.com/software/modelzoo) and corresponding [gitee repository](https://gitee.com/ascend/modelzoo)

If you are looking for some pretrained checkpoint of mindspore, you could refer to [MindSpore Hub](https://www.mindspore.cn/resources/hub/en) or [Download Website](https://download.mindspore.cn/model_zoo/).

## Disclaimers

Mindspore only provides scripts that downloads and preprocesses public datasets. We do not own these datasets and are not responsible for their quality or maintenance. Please make sure you have permission to use the dataset under the dataset’s license. The models trained on these dataset are for non-commercial research and educational purpose only.

To dataset owners: we will remove or update all public content upon request if you don’t want your dataset included on Mindspore, or wish to update it in any way. Please contact us through a Github/Gitee issue. Your understanding and contribution to this community is greatly appreciated.

MindSpore is Apache 2.0 licensed. Please see the LICENSE file.

## License

[Apache License 2.0](https://gitee.com/mindspore/mindspore/blob/master/LICENSE)

## FAQ

For more information about `MindSpore` framework, please refer to [FAQ](https://www.mindspore.cn/docs/en/master/faq/installation.html)

- **Q: How to resolve the lack of memory while using the model directly under "models" with errors such as *Failed to alloc memory pool memory*?**

  **A**: The typical reason for insufficient memory when directly using models under "models" is due to differences in operating mode (`PYNATIVE_MODE`), operating environment configuration, and license control (AI-TOKEN).
    - `PYNATIVE_MODE` usually uses more memory than `GRAPH_MODE` , especially in the training graph that needs back propagation calculation, there are two ways to try to solve this problem.
        Method 1: You can try to use some smaller batch size;
        Method 2: Add context.set_context(mempool_block_size="XXGB"), where the current maximum effective value of "XX" can be set to "31".
        If method 1 and method 2 are used in combination, the effect will be better.
    - The operating environment will also cause similar problems due to the different configurations of NPU cores, memory, etc.;
    - Different gears of License control (AI-TOKEN ) will cause different memory overhead during execution. You can also try to use some smaller batch sizes.

- **Q: How to resolve the error about the interface are not supported in some network operations, such as `cann not import`?**

  **A**: Please check the version of MindSpore and the branch you fetch the modelzoo scripts. Some model scripits in latest branch will use new interface in the latest version of MindSpore.

- **Q: What is Some *RANK_TBAL_FILE* which mentioned in many models?**

  **A**: *RANK_TABLE_FILE* is the config file of cluster on Ascend while running distributed training. For more information, you could refer to the generator [hccl_tools](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools) and [Parallel Distributed Training Example](https://mindspore.cn/docs/programming_guide/en/r1.5/distributed_training_ascend.html#configuring-distributed-environment-variables)

- **Q: How to run the scripts on Windows system?**

  **A**: Most the start-up scripts are written in `bash`, but we usually can't run bash directly on Windows. You can try start python directly without bash scripts. If you really need the start-up bash scripts, we suggest you the following method to get a bash environment on Windows:
    1. Use a virtual system or docker container with linux system. Then run the scripts in the virtual system or container.
    2. Use WSL, you could turn on the `Windows Subsystem for Linux` on Windows to obtain an linux system which could run the bash scripts.
    3. Use some bash tools on Windows, such as [cygwin](http://www.cygwin.com) and [git bash](https://gitforwindows.org).

- **Q: How to resolve the compile error point to gflags when infer on ascend310 with errors such as *undefined reference to 'google::FlagRegisterer::FlagRegisterer'*?**

  **A**: Please check the version of GCC and gflags. You can refer to [GCC](https://www.mindspore.cn/install) and [gflags](https://github.com/gflags/gflags/archive/v2.2.2.tar.gz) to install GCC and gflags. You need to ensure that the components used are ABI compatible, for more information, please refer to [_GLIBCXX_USE_CXX11_ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html).

- **Q: How to solve the error when loading dataset in mindrecord format on Mac system, such as *Invalid file, failed to open files for reading mindrecord files.*?**

  **A**: Please check the system limit with *ulimit -a*, if the number of *file descriptors* is 256 (default), you need to use *ulimit -n 1024* to set it to 1024 (or larger). Then check whether the file is damaged or modified.

- **Q: What should I do if I can't reach the accuracy while training with several servers instead of a single server?**

  **A**: Most of the models has only been trained on single server with at most 8 pcs. Because the `batch_size` used in MindSpore only represent the batch size of single GPU/NPU, the `global_batch_size` will increase while training with multi-server. Different `gloabl_batch_size` requires different hyper parameter including learning_rate, etc. So you have to optimize these hyperparameters will training with multi-servers.
