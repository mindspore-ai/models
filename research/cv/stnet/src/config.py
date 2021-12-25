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
"""config"""
cfg_res50 = {
    # normol use
    'target_size': 224,
    'short_size': 256,
    'class_num': 400,
    'T': 7,  # segnum
    'N': 5,  # seglen
    'image_mean': [0.485, 0.456, 0.406],
    'image_std': [0.229, 0.224, 0.225],
    # train
    'batch_size': 16,
    'num_epochs': 60,
    'mode': 'GRAPH',
    'target': 'Ascend',
    'device_num': 8,
    'save_checkpoint': False,
    'eval_per_epoch': 10,
    'parameter_server': True,
    'dataset_sink_mod': True,
    'keep_checkpoint_max': 6,
    'save_checkpoint_epochs': 10,
    'save_best_ckpt': True,
    'eval_start_epoch': 10,
    'run_eval': True,
    'eval_interval': 10,
    'pre_res50': "/disk0/dataset/kin400/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76.ckpt",

    # opt
    'momentum': 0.9,
    'lr': 0.01,
    'gamma': 0.1,
    'weight_decay': 1e-4,

    # val
    'checkpoint_path': '/cache/train_out/',
    'val_data_dir': 'kin_400/val',
    'summary_dir': '/cache/train_out/summary_dir',

    # model_Art
    'run_online': False,
    'data_url': '',
    'local_data_url': '',
    'pre_res50_art_load_path': '',
    'best_acc_art_load_path': '',
    'pre_url': '',
    'load_path': '',
    'local_train_list': '',
    'local_val_list': '',
    'train_url': '',
    'output_path': '',
}
