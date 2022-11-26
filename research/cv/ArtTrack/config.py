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

import os

import yaml
from easydict import EasyDict as edict


def get_default_config():
    cfg1 = edict()
    cfg1.context = edict({"mode": 1, "device_target": "GPU"})
    cfg1.parallel_context = None
    cfg1.stride = 8.0
    cfg1.weigh_only_present_joints = False
    cfg1.mean_pixel = [123.68, 116.779, 103.939]
    cfg1.global_scale = 1.0
    cfg1.location_refinement = False
    cfg1.locref_stdev = 7.2801
    cfg1.locref_loss_weight = 1.0
    cfg1.locref_huber_loss = True
    cfg1.intermediate_supervision = False
    cfg1.intermediate_supervision_layer = 12
    cfg1.intermediate_supervision_input = 1024
    cfg1.mirror = False
    cfg1.crop = False
    cfg1.crop_pad = 0
    cfg1.scoremap_dir = "out/test"
    cfg1.dataset = edict({"path": "", "type": "", "parallel": 1, "batch_size": 1,
                          "shuffle": False, "mirror": False, "padding": False})
    cfg1.use_gt_segm = False
    cfg1.sparse_graph = []
    cfg1.pairwise_stats_collect = False
    cfg1.pairwise_stats_fn = "pairwise_stats.mat"
    cfg1.pairwise_predict = False
    cfg1.pairwise_huber_loss = True
    cfg1.pairwise_loss_weight = 1.0
    cfg1.tensorflow_pairwise_order = True
    return cfg1


cfg = get_default_config()


def merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    for k, v in a.items():
        b[k] = v


def cfg_from_file(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    merge_a_into_b(yaml_cfg, cfg)

    return cfg


def load_config(filename="/cache/data/config/pose_cfg.yaml"):
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.path.join(os.environ['POSE_PARAM_PATH'], filename)
    return cfg_from_file(filename)


def check_config(c, args):
    if c is None:
        if not args.is_model_arts:
            return load_config("./config/pose_cfg.yaml")
        return load_config("/cache/data/config/pose_cfg.yaml")
    return load_config(c)


if __name__ == "__main__":

    print(load_config())
