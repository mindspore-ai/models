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
"""Environ setting."""
import os
import cv2
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size


def init_env(cfg):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(2)
    ms.set_seed(cfg.seed)

    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU")
        ms.set_context(device_target=cfg.device_target)

    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")
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
        cpu_affinity(cfg.rank_id, cfg.device_num)
    else:
        if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
            ms.set_context(device_id=cfg.device_id)
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)


def cpu_affinity(rank_id, device_num):
    """Bind CPU cores according to rank_id and device_num."""
    import psutil
    cores = psutil.cpu_count()
    if cores < device_num:
        return
    process = psutil.Process()
    used_cpu_num = cores // device_num
    rank_id = rank_id % device_num
    used_cpu_list = [i for i in range(rank_id * used_cpu_num, (rank_id + 1) * used_cpu_num)]
    process.cpu_affinity(used_cpu_list)
    print(f"==== {rank_id}/{device_num} ==== bind cpu: {used_cpu_list}")
