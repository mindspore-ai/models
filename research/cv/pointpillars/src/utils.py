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
"""utils"""
import yaml

from src.builder import box_coder_builder
from src.builder import dataset_builder
from src.builder import model_builder
from src.builder import target_assigner_builder
from src.builder import voxel_builder


def get_model_dataset(cfg, is_training=True):
    """get model dataset"""
    model_cfg = cfg['model']

    voxel_cfg = model_cfg['voxel_generator']
    voxel_generator = voxel_builder.build(voxel_cfg)

    box_coder_cfg = model_cfg['box_coder']
    box_coder = box_coder_builder.build(box_coder_cfg)

    target_assigner_cfg = model_cfg['target_assigner']
    target_assigner = target_assigner_builder.build(target_assigner_cfg, box_coder)

    pointpillarsnet = model_builder.build(model_cfg, voxel_generator, target_assigner)

    if is_training:
        input_cfg = cfg['train_input_reader']
    else:
        input_cfg = cfg['eval_input_reader']

    dataset = dataset_builder.build(
        input_reader_config=input_cfg,
        model_config=model_cfg,
        training=is_training,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner
    )
    if is_training:
        return pointpillarsnet, dataset
    return pointpillarsnet, dataset, box_coder


def get_params_for_net(params):
    """get params for net"""
    new_params = {}
    for key, value in params.items():
        if key.startswith('optimizer.'):
            new_params[key[10:]] = value
        elif key.startswith('network.network.'):
            new_params[key[16:]] = value
    return new_params


def get_config(cfg_path):
    """get config"""
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, yaml.Loader)
    return cfg
