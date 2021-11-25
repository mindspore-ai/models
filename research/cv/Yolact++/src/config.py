# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

""" Related parameter configuration """
from src.yolact.layers.backbone_dcnV2 import ResNetBackbone

resnet_transform = {
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
}

resnet50_dcnv2 = {
    'name': 'ResNet50_DCNv2',
    # 'path': 'resnet50-19c8e357.pth',
    'path': 'resnet50.ckpt',
    'type': ResNetBackbone,
    'transform': resnet_transform,
    'args': ([3, 4, 6, 3], [0, 4, 6, 3]),
    # 'arg1': ([3, 4, 6, 3]),
    # 'arg2': ([0, 4, 6, 3]),
    'selected_layers': list(range(1, 4)),
    # 'selected_layers': range(1,3),
    'pred_aspect_ratios': [[[1, 1 / 2, 2]]] * 5,
    # 'pred_scales': [[24], [48], [96], [192], [384]],
    'pred_scales': [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]],
    'use_pixel_scales': True,
    'preapply_sqrt': False,
    'use_square_anchors': False,
}

fpn = {
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 2,  # Add two new convolutional downsampling, that is, add p6, p7

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': True,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,

    # Whether to add relu to the downsampled layers.
    'relu_downsample_layers': False,

    # Whether to add relu to the regular layers
    'relu_pred_layers': True,
}

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                  '9': 9, '10': 10, '11': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16,
                  '18': 17, '19': 18, '20': 19, '21': 20, '22': 21, '23': 22, '24': 23, '25': 24,
                  '27': 25, '28': 26, '31': 27, '32': 28, '33': 29, '34': 30, '35': 31, '36': 32,
                  '37': 33, '38': 34, '39': 35, '40': 36, '41': 37, '42': 38, '43': 39, '44': 40,
                  '46': 41, '47': 42, '48': 43, '49': 44, '50': 45, '51': 46, '52': 47, '53': 48,
                  '54': 49, '55': 50, '56': 51, '57': 52, '58': 53, '59': 54, '60': 55, '61': 56,
                  '62': 57, '63': 58, '64': 59, '65': 60, '67': 61, '70': 62, '72': 63, '73': 64,
                  '74': 65, '75': 66, '76': 67, '77': 68, '78': 69, '79': 70, '80': 71, '81': 72,
                  '82': 73, '84': 74, '85': 75, '86': 76, '87': 77, '88': 78, '89': 79, '90': 80}

COCO_LABEL_MAP_EVAL = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                       9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                       18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                       27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                       37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                       46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                       54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                       62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                       74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                       82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


coco2017_dataset = {
    'name': 'COCO 2017',

    'train_images': './data/coco2017/images/',
    'train_info': './data/coco2017/annotations/instances_val2017.json',
    'valid_images': './data/coco2017/images/',
    'valid_info': './data/coco2017/annotations/instances_val2017.json',
    'has_gt': True,
    'class_names': COCO_CLASSES,
    'label_map': COCO_LABEL_MAP,
    'label_map_eval': COCO_LABEL_MAP_EVAL,
}

mask_type = {
    'direct': 0,
    'lincomb': 1,
}

yolact_plus_resnet50_config = {

    'mindrecord_dir': '/data/yolactms/MindRecord_COCO2017_val/',
    'coco_root': '/data/coco2017',
    'IMAGE_DIR': '/data/coco2017/val2017',
    'ANNO_PATH': '/data/coco2017/annotations',

    'max_instance_count': 128,
    'img_width': 550,
    'img_height': 550,
    'ckpt_file': '/data/yolact_yin/checkpoint/ckpt_0/yolact-90_619.ckpt',
    'mask_count': 15,
    'file_name': 'Yolact',
    'file_format': 'MINDIR',
    'random_crowd': 0,

    'train_data_type': "val2017",
    'val_data_type': "val2017",
    'instance_set': "annotations/instances_{}.json",

    'name': 'yolact_plus_resnet50',

    'epoch_size': 300,  # 54,
    'pretrain_epoch_size': 0,
    'save_checkpoint': True,
    'save_checkpoint_epochs': 10,  # 1,
    'keep_checkpoint_max': 100, # 54,
    "save_checkpoint_path": './checkpoint/with_torch_pth',

    'batch_size': 8,  # 8,
    'num_priors': 57744, # 19248,
    # Backbone Settings
    'backbone': resnet50_dcnv2,
    #'selected_layers': range(1, 3),
    'dataset': coco2017_dataset,
    # 'num_classes': len(coco2017_dataset['class_names']) + 1,  # This should include the background class
    'num_classes': len(coco2017_dataset['class_names']) + 1,  # This should include the background class

    'max_iter': 800000,

    # Mask Settings
    # 'mask_type': mask_type.lincomb,
    'mask_type': mask_type['lincomb'],
    'mask_alpha': 6.125,
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_normalize_emulate_roi_pooling': True,

    # Other stuff
    'share_prediction_module': True,
    'extra_head_net': [(256, 3, {'padding': 1})],

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,
    'score_threshold': 0,

    'crowd_iou_threshold': 0.7,

    'use_semantic_segmentation_loss': True,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 5e-4,# 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,
    'loss_scale': 8,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 600000, 700000, 750000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 500,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'bbox_alpha': 1.5,

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,

    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,

    # See mask_type for details.
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_bias': False,
    # 'mask_proto_prototype_activation': activation_func.relu,
    # 'mask_proto_mask_activation': activation_func.sigmoid,
    # 'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': True,
    'mask_proto_crop_expand': 0,
    'mask_proto_loss': None,
    'mask_proto_binarize_downsampled_gt': True,
    'mask_proto_normalize_mask_loss_by_sqrt_area': False,
    'mask_proto_reweight_mask_loss': False,
    'mask_proto_grid_file': 'data/grid.npy',
    'mask_proto_use_grid': False,
    'mask_proto_coeff_gate': False,
    'mask_proto_prototypes_as_features': False,
    'mask_proto_prototypes_as_features_no_grad': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_coeff_diversity_alpha': 1,
    'mask_proto_split_prototypes_by_head': False,
    # 'mask_proto_crop_with_pred_box': False,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': True,
    # Potentially sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True,
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    # With uniform probability, rotate the image [0,90,180,270] degrees
    'augment_random_rot90': False,

    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4 / 550,
    'discard_box_height': 4 / 550,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': fpn,


    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,

    # The initial bias toward foreground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Keeps track of the average number of examples for each class, and weights the loss for that class accordingly.
    'use_class_balanced_conf': False,

    # Whether to use sigmoid focal loss instead of softmax, all else being the same.
    'use_sigmoid_focal_loss': False,

    # Use class[0] to be the objectness score and class[1:] to be the softmax predicted class.
    # Note: at the moment this is only implemented if use_focal_loss is on.
    'use_objectness_score': False,

    # Adds a global pool + fc layer to the smallest selected layer that predicts the existence of each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_class_existence_loss': False,
    'class_existence_alpha': 1,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'semantic_segmentation_alpha': 1,

    # Adds another branch to the netwok to predict Mask IoU.
    'use_mask_scoring': False,
    'mask_scoring_alpha': 1,

    # Match gt boxes using the Box2Pix change metric instead of the standard IoU metric.
    # Note that the threshold you set for iou_threshold should be negative with this setting on.
    'use_change_matching': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    # 因为conv2d，所以加了pad_mode参数
    'head_layer_params': {'kernel_size': 3, 'padding': 1, 'pad_mode': 'pad'},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),


    # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
    'ohem_negpos_ratio': 3,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size.
    'max_size': 550,

    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for coeff_diversity_loss
    # Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
    # To see their effect, also remember to turn on use_coeff_nms.
    'use_instance_coeff': False,
    'num_instance_coeffs': 64,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,

    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    # Use command-line arguments to set this.
    'no_jit': False,




    # Fast Mask Re-scoring Network
    # Inspried by Mask Scoring R-CNN (https://arxiv.org/abs/1903.00241)
    # Do not crop out the mask with bbox but slide a convnet on the image-size mask,
    # then use global pooling to get the final mask score
    # 'use_maskiou': False,
    'use_maskiou': True,


    # Archecture for the mask iou network. A (num_classes-1, 1, {}) layer is appended to the end.
    'maskiou_net': [(8, 3, {'stride': 2}), (16, 3, {'stride': 2}), (32, 3, {'stride': 2}), (64, 3, {'stride': 2}),
                    (128, 3, {'stride': 2})],

    # Discard predicted masks whose area is less than this
    'discard_mask_area': 5 * 5,

    'maskiou_alpha': 25,
    'rescore_mask': True,
    'rescore_bbox': False,
    'maskious_to_train': -1,
}


COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
