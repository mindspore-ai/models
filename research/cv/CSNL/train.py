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
from mindspore import context
from mindspore.communication.management import init, get_group_size
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from model_utils.local_adapter import get_device_id
from model_utils.config import config
from src.CSNLN import CSNLN
from src.dataset.dataset import create_dataset_DIV2K
from src.utils import Trainer

set_seed(2)


def train():
    cfg = config
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    if cfg.distribute:
        init()
        device_num = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
    else:
        context.set_context(device_id=get_device_id())
    if cfg.dataset_name == 'DIV2K':
        dataset = create_dataset_DIV2K(cfg)
    else:
        raise ValueError("Unsupported dataset")

    net = CSNLN(args=cfg)
    print("init net weights succeed")
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.save + cfg.ckpt_file)
        load_param_into_net(net, param_dict)
        print("load net weights succeed")

    train_fun = Trainer(config, dataset, net)
    for epoch in range(cfg.epoches):
        train_fun.update_learning_rate(epoch)
        train_fun.train()


if __name__ == '__main__':
    train()
