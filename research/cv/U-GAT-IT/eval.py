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
"""
launch file for traning and testing
"""
import os
import time

import mindspore
from mindspore import context
from mindspore.communication import init, get_group_size
from mindspore.context import ParallelMode

from src.modelarts_utils.config import config
from src.modelarts_utils.moxing_adapter import moxing_wrapper
from src.models.UGATIT import UGATIT
from src.utils.tools import check_folder

mindspore.set_seed(1)

@moxing_wrapper()
def main():
    # parse arguments
    config.phase = 'test'
    if config.distributed:
        if config.device_target == 'Ascend':
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(mode=context.GRAPH_MODE,
                                device_target=config.device_target,
                                device_id=device_id,
                                save_graphs=config.save_graphs,
                                save_graphs_path=config.graph_path)
            init()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              device_num=device_num)
        elif config.device_target == 'GPU':
            init("nccl")
            context.reset_auto_parallel_context()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            raise ValueError("Wrong device", config.device_target)

    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=config.device_target,
                            device_id=int(config.device_id),
                            save_graphs=config.save_graphs,
                            save_graphs_path=config.graph_path)

    check_folder(os.path.join(config.output_path, config.dataset, 'model'))
    check_folder(os.path.join(config.output_path, config.dataset, 'img'))
    check_folder(os.path.join(config.output_path, config.dataset, 'test'))
    check_folder(config.graph_path)
    # open session
    gan = UGATIT(config)

    # build graph
    start_time = time.time()
    gan.build_model()
    print("build_model cost time: %.4f" % (time.time() - start_time))

    if config.compute_metrics:
        gan.test(config.inception_ckpt_path)
    else:
        gan.test()
    print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
