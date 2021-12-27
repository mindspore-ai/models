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
"""Export ckpt to model
"""
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint

from src.dlrm import ModelBuilder
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == 'Ascend':
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_dlrm():
    """ export_dlrm"""
    model_builder = ModelBuilder(config, config)
    _, network = model_builder.get_train_eval_net()
    network.set_train(False)

    load_checkpoint(config.checkpoint_path, net=network)

    #NOTE: change the batch size to 1
    batch_cats = Tensor(np.zeros([1, config.slot_dim]).astype(np.int32))
    batch_nums = Tensor(np.zeros([1, config.dense_dim]).astype(np.float32))
    labels = Tensor(np.zeros([1, 1]).astype(np.float32))

    input_data = [batch_cats, batch_nums, labels]
    export(network, *input_data, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_dlrm()
