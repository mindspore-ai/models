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
"""python eval.py"""

import random
import numpy as np
from src.model.Cnet import Cnet
from src.dataset import create_evalloaders
from src.EvalMetrics import inference
from src.EvalMetrics import ErrorRateAt95Recall
from mindspore import context
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from model_utils.device_adapter import get_device_id
from model_utils.config import config

ms.common.set_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

if __name__ == '__main__':
    triplet_flag = (config.batch_reduce == 'random_global') or config.gor
    if config.modelArts_mode:
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=config.data_url, dst_url=local_data_url)

        # download dataset from obs to cache
        if "obs://" in config.checkpoint_path:
            local_checkpoint_url = "/cache/" + config.checkpoint_path.split("/")[-1]
            mox.file.copy_parallel(config.checkpoint_path, local_checkpoint_url)
            config.checkpoint_path = local_checkpoint_url
        config.dataroot = local_data_url
    device_id = get_device_id()

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    elif config.device_target == "Ascend":
        context.set_context(device_id=device_id)

    eval_data = create_evalloaders(load_random_triplets=triplet_flag, config=config)
    network = Cnet()
    param_dict = load_checkpoint(ckpt_file_name=config.checkpoint_path)
    print("load checkpoint from [{}].".format(config.checkpoint_path))
    load_param_into_net(network, param_dict)
    network.set_train(False)
    acc_fn = ErrorRateAt95Recall()
    acc = inference(network, eval_data, acc_fn)

    print("============= 910 Inference =============", flush=True)
    print("eval dataset:{}, Accuracy(FPR95): {:.5f}".format(config.eval_set, acc))
    print("=========================================", flush=True)
