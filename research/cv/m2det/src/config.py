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

device = 'GPU'
random_seed = 1
experiment_tag = 'm2det512_vgg16'
checkpoint_name = None
start_epoch = 0

if checkpoint_name:
    checkpoint_path = '/workdir/m2det-mindspore/checkpoints/' + experiment_tag + '/' + checkpoint_name
else:
    checkpoint_path = None


model = dict(
    type='m2det',
    input_size=512,
    init_net=True,
    m2det_config=dict(
        backbone='vgg16',
        net_family='vgg',
        base_out=[22, 34],  # [22,34] for vgg, [2,4] or [3,4] for res families
        planes=256,
        num_levels=8,
        num_scales=6,
        sfam=False,
        smooth=True,
        num_classes=81,
        checkpoint_path='/workdir/m2det-mindspore/checkpoints/vgg16_reducedfc.ckpt'
    ),
    rgb_means=(104, 117, 123),
    p=0.6,
    anchor_config=dict(
        step_pattern=[8, 16, 32, 64, 128, 256],
        size_pattern=[0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
    ),
    checkpoint_interval=10,
    weights_save='weights/'
)

train_cfg = dict(
    lr=1e-3,
    warmup=5,
    per_batch_size=7,
    gamma=[0.5, 0.2, 0.1, 0.1],
    lr_epochs=[90, 110, 130, 150, 160],
    total_epochs=160,
    print_epochs=10,
    num_workers=3,
    )

test_cfg = {
    'cuda': True,
    'topk': 0,
    'iou': 0.45,
    'soft_nms': True,
    'score_threshold': 0.1,
    'keep_per_class': 50,
    'save_folder': 'eval',
}

loss = {
    'overlap_thresh': 0.5,
    'prior_for_matching': True,
    'bkg_label': 0,
    'neg_mining': True,
    'neg_pos': 3,
    'neg_overlap': 0.5,
    'encode_target': False,
}

optimizer = {
    'type': 'SGD',
    'momentum': 0.9,
    'weight_decay': 0.00005,
    'dampening': 0.0,
    'clip_grad_norm': 4.,
}

dataset = {
    'COCO': {
        'train_sets': [('2014', 'train'), ('2014', 'valminusminival')],
        'eval_sets': [('2014', 'minival')],
        'test_sets': [('2015', 'test-dev')],
    }
}

# Dataset root folder. Used then no dataset_path argument specified for training script
COCOroot = '/workdir/datasets/coco2014/'
