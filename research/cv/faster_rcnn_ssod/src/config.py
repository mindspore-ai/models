# Copyright 2022 Huawei Technologies Co., Ltd
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
"""FasterRcnn config"""


class FasterRcnnConfig:
    # Builtin Configurations(DO NOT CHANGE THESE CONFIGURATIONS unless you know exactly what you are doing)
    device_target = "Ascend"

    # ==============================================================================
    # config
    img_width = 1280    # 1280
    img_height = 768   # 768
    keep_ratio = True
    flip_ratio = 0.5

    # anchor
    feature_shapes = [
        [img_height // 4, img_width // 4],
        [img_height // 8, img_width // 8],
        [img_height // 16, img_width // 16],
        [img_height // 32, img_width // 32],
        [img_height // 64, img_width // 64],
    ]
    anchor_scales = [8]
    anchor_ratios = [0.5, 1.0, 2.0]
    anchor_strides = [4, 8, 16, 32, 64]
    num_anchors = 3

    # resnet
    resnet_block = [3, 4, 6, 3]
    resnet_in_channels = [64, 256, 512, 1024]
    resnet_out_channels = [256, 512, 1024, 2048]

    # fpn
    fpn_in_channels = [256, 512, 1024, 2048]
    fpn_out_channels = 256
    fpn_num_outs = 5

    # rpn
    rpn_in_channels = 256
    rpn_feat_channels = 256
    rpn_loss_cls_weight = 1.0
    rpn_loss_reg_weight = 1.0
    rpn_cls_out_channels = 1
    rpn_target_means = [0., 0., 0., 0.]
    rpn_target_stds = [1.0, 1.0, 1.0, 1.0]

    # bbox_assign_sampler
    neg_iou_thr = 0.3
    pos_iou_thr = 0.7
    min_pos_iou = 0.3
    num_bboxes = num_anchors * sum([lst[0] * lst[1] for lst in feature_shapes])
    num_gts = 128
    num_expected_neg = 256
    num_expected_pos = 128

    # proposal
    activate_num_classes = 2
    use_sigmoid_cls = True

    # roi_align
    class RoiLayer:
        type = 'RoIAlign'
        out_size = 7
        sample_num = 2

    roi_layer = RoiLayer()
    roi_align_out_channels = 256
    roi_align_featmap_strides = [4, 8, 16, 32]
    roi_align_finest_scale = 56
    roi_sample_num = 640

    # bbox_assign_sampler_stage2
    neg_iou_thr_stage2 = 0.5
    pos_iou_thr_stage2 = 0.5
    min_pos_iou_stage2 = 0.5
    num_bboxes_stage2 = 2000
    num_expected_pos_stage2 = 128
    num_expected_neg_stage2 = 512
    num_expected_total_stage2 = 512

    # rcnn
    rcnn_num_layers = 2
    rcnn_in_channels = 256
    rcnn_fc_out_channels = 1024
    rcnn_loss_cls_weight = 1
    rcnn_loss_reg_weight = 1
    rcnn_target_means = [0., 0., 0., 0.]
    rcnn_target_stds = [0.1, 0.1, 0.2, 0.2]

    # train proposal
    rpn_proposal_nms_across_levels = False
    rpn_proposal_nms_pre = 2000
    rpn_proposal_nms_post = 2000
    rpn_proposal_max_num = 2000
    rpn_proposal_nms_thr = 0.7
    rpn_proposal_min_bbox_size = 0

    # test proposal
    rpn_nms_across_levels = False
    rpn_nms_pre = 1000
    rpn_nms_post = 1000
    rpn_max_num = 1000
    rpn_nms_thr = 0.7
    rpn_min_bbox_min_size = 0
    test_score_thr = 0.05
    test_iou_thr = 0.5
    test_max_per_img = 100
    test_batch_size = 2

    rpn_head_use_sigmoid = True
    rpn_head_weight = 1.0

    # LR
    lr_schedule = "step"
    milestones = [19990, 19995]
    base_lr = 0.005
    gamma = 0.1
    warmup_ratio = 0.0625
    warmup_step = 500

    # train
    global_seed = 10
    resume = False
    run_distribute = False
    batch_size = test_batch_size
    loss_scale = 256
    momentum = 0.9
    weight_decay = 0.0001
    save_checkpoint = True
    save_checkpoint_path = "./outputs/"
    save_checkpoint_interval = 1000

    # semi-train
    ema_keep_rate = 0.9996
    unsup_loss_weight = 4.0
    bbox_threshold = 0.7
    max_iter = 20000
    start_iter = 0
    burn_up_iter = 3000
    teacher_update_iter = 1
    print_interval_iter = 100

    # dataset
    num_parallel_workers = 2
    num_classes = 4     # need add one
    train_img_dir = "/home/datasets/FaceMaskDetectionDataset/train/images"
    train_ann_file = "/home/datasets/FaceMaskDetectionDataset/train/annotations/instances_train2017_25.json"
    eval_img_dir = "/home/datasets/FaceMaskDetectionDataset/val/images"
    eval_ann_file = "/home/datasets/FaceMaskDetectionDataset/val/annotations/instances_val2017.json"

    # train.py FasterRcnn training
    pre_trained = None
    filter_prefix = ["fpn_ncek", "rpn_with_loss", "rcnn"]
    pre_trained_teacher = None
    device_id = 0

    # eval.py FasterRcnn evaluation
    checkpoint_path = None
    eval_output_dir = "./outputs/"
    eval_device_id = 0

    # infer.py FasterRcnn Infer
    draw_pics = False
