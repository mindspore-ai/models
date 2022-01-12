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

import yaml
from easydict import EasyDict as edict


def get_default_config():
    _cfg = edict()
    _cfg.context = edict({"mode": 1, "device_target": "GPU"})
    _cfg.parallel_context = None
    _cfg.stride = 8.0
    _cfg.weigh_only_present_joints = False
    _cfg.mean_pixel = [123.68, 116.779, 103.939]
    _cfg.global_scale = 1.0
    _cfg.location_refinement = False
    _cfg.locref_stdev = 7.2801
    _cfg.locref_loss_weight = 1.0
    _cfg.locref_huber_loss = True
    _cfg.intermediate_supervision = False
    _cfg.intermediate_supervision_layer = 12
    _cfg.intermediate_supervision_input = 1024
    _cfg.mirror = False
    _cfg.crop = False
    _cfg.crop_pad = 0
    _cfg.scoremap_dir = "out/eval"
    _cfg.dataset = edict({"path": "", "type": "", "parallel": 1, "batch_size": 1,
                          "shuffle": False, "mirror": False, "padding": False})
    _cfg.use_gt_segm = False
    _cfg.sparse_graph = []
    _cfg.pairwise_stats_collect = False
    _cfg.pairwise_stats_fn = "pairwise_stats.mat"
    _cfg.pairwise_predict = False
    _cfg.pairwise_huber_loss = True
    _cfg.pairwise_loss_weight = 1.0
    _cfg.tensorflow_pairwise_order = True
    return _cfg


cfg = get_default_config()


def merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if not isinstance(a, edict) and not isinstance(a, dict):
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if isinstance(v, (edict, dict)):
            try:
                item = b.get(k, None)
                if item is None:
                    item = edict()
                    b[k] = item
                merge_a_into_b(a[k], b[k])
            except Exception:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def load_config(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    merge_a_into_b(yaml_cfg, cfg)

    return cfg


def check_config(c=None):
    if c is None:
        return load_config()
    return c


if __name__ == "__main__":
    print(load_config())
