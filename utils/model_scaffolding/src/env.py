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

"""Initialize the runtime environment"""
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size


def init_env(cfg):
    """Initialize the runtime environment."""
    ms.set_seed(cfg.seed)
    # If device_target setting is None, the framework automatically obtain the device_target,
    # otherwise use the setting.
    if hasattr(cfg, "device_target") and cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)

    # Configure runtime mode, support GRAPH mode and PYNATIVE mode.
    if not hasattr(cfg, "context_mode"):
        cfg.context_mode = "pynative"
        print("Not set context_mode in yaml, now set PYNATIVE mode."
              "If you want to run in GRAPH mode, please add context_mode in yaml file.")
    if cfg.context_mode.lower() not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode.lower() == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")
    # If it is running on the CPU, the multi card environment is not configured
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    if cfg.device_num > 1:
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        # Set the device_id
        if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
            ms.set_context(device_id=cfg.device_id)
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)
