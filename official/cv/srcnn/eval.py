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
"""srcnn evaluation"""
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_eval_dataset
from src.srcnn import SRCNN
from src.metric import SRCNNpsnr

from src.model_utils.config import config
from src.model_utils.moxing_adapter import sync_data

def run_eval():
    cfg = config
    if cfg.enable_modelarts == "True":
        sync_data(cfg.data_url, cfg.data_path)
        sync_data(cfg.checkpoint_url, cfg.checkpoint_path)
    local_dataset_path = cfg.data_path
    local_checkpoint_path = cfg.checkpoint_path
    if cfg.device_target == "GPU" or cfg.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=cfg.device_target,
                            save_graphs=False)
    else:
        raise ValueError("Unsupported device target.")
    eval_ds = create_eval_dataset(local_dataset_path, cfg.scale)
    net = SRCNN()
    lr = Tensor(config.lr, ms.float32)
    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)
    loss = nn.MSELoss(reduction='mean')
    param_dict = load_checkpoint(local_checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'PSNR': SRCNNpsnr()})
    res = model.eval(eval_ds, dataset_sink_mode=False)
    print("result ", res)

if __name__ == '__main__':
    run_eval()
