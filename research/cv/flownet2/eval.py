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

import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net


import src.dataset as datasets
import src.models as models
from src.metric import FlowNetEPE
import src.model_utils.tools as tools
from src.model_utils.config import config

def run_eval():
    set_seed(config.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.STAND_ALONE, gradients_mean=True, device_num=1)
    ds.config.set_enable_shared_mem(False)
    # load dataset by config param
    config.eval_dataset_class = tools.module_to_dict(datasets)[config.eval_data]
    flownet_eval_gen = config.eval_dataset_class("Center", config.crop_size, config.eval_size,
                                                 config.eval_data_path)
    eval_dataset = ds.GeneratorDataset(flownet_eval_gen, ["images", "flow"]
                                       , num_parallel_workers=config.num_parallel_workers,
                                       max_rowsize=config.max_rowsize)
    eval_dataset = eval_dataset.batch(config.batch_size)

    # load model by config param
    config.model_class = tools.module_to_dict(models)[config.model]
    net = config.model_class(config.rgb_max, config.batchNorm)

    loss = nn.L1Loss()

    param_dict = load_checkpoint(config.eval_checkpoint_path)
    print("load checkpoint from [{}].".format(config.eval_checkpoint_path))
    load_param_into_net(net, param_dict)
    net.set_train(False)

    model = Model(net, loss_fn=loss, metrics={'flownetEPE': FlowNetEPE()})

    mean_error = model.eval(eval_dataset, dataset_sink_mode=False)

    print("flownet2 mean error: ", mean_error)


if __name__ == '__main__':
    run_eval()
