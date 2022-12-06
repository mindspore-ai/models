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

"""export pruned mindir and air.
pruning masks (.json format) are obtained during training in the experiment directory."""
import os
import numpy as np
import mindspore as ms
from mindspore_gs.pruner.uni_pruning import UniPruner
#pylint: disable=ungrouped-imports
from src.resnet import resnet18, resnet50
from src.model_utils.config import config

def export():
    """export pruned mindir and air.
    pruning masks (.json format) are obtained during training in the experiment directory."""

    target = config.device_target

    # init context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=target, save_graphs=False)
    if target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
    else:
        device_id = config.device_id
    ms.set_context(device_id=device_id)

    # define net
    if config.net_name == 'resnet18':
        net = resnet18(class_num=config.class_num)
    elif config.net_name == 'resnet50':
        net = resnet50(class_num=config.class_num)
    net.set_train(False)
    input_size = [config.export_batch_size, 3, config.height, config.width]
    algo = UniPruner({"exp_name": config.exp_name,
                      "frequency": config.retrain_epochs,
                      "target_sparsity": 1 - config.prune_rate,
                      "pruning_step": config.pruning_step,
                      "filter_lower_threshold": config.filter_lower_threshold,
                      "input_size": input_size,
                      "output_path": config.output_path,
                      "prune_flag": config.prune_flag,
                      "rank": config.device_id,
                      "device_target": config.device_target})
    print('start converting')
    save_path = algo.callbacks()[0].output_path
    net_deploy = algo.convert(net_opt=net,
                              ckpt_path=config.checkpoint_file_path,
                              mask_path=config.mask_path)
    inputs = np.random.uniform(0.0, 1.0, size=input_size).astype(np.float32)
    inputs = ms.Tensor(inputs)
    ms.export(net_deploy, inputs, file_name=f"{save_path}_pruned.mindir", file_format="MINDIR")


if __name__ == '__main__':
    export()
