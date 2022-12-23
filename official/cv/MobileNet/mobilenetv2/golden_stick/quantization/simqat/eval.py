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
eval.
"""
import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from simqat import create_simqat

from src.dataset import create_dataset_cifar10
from src.models import define_net
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id

config.is_training = config.is_training_eval


@moxing_wrapper(pre_process=modelarts_process)
def eval_mobilenetv2():
    """eval_mobilenetv2 """
    if config.mode_name == "GRAPH":
        ms.set_context(mode=ms.GRAPH_MODE, device_target=config.platform, save_graphs=False)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=config.platform, save_graphs=False)
    config.dataset_path = config.dataset_path
    print('\nconfig: \n', config)
    if not config.device_id:
        config.device_id = get_device_id()
    _, _, net = define_net(config, config.is_training)
    algo = create_simqat()
    net = algo.apply(net)
    param_dict = load_checkpoint(config.pretrain_ckpt)
    load_param_into_net(net, param_dict)

    net.set_train(False)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = ms.Model(net, loss_fn=loss, metrics={'acc'})
    dataset = create_dataset_cifar10(dataset_path=config.dataset_path, do_train=False, config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of eval dataset is more \
            than batch_size in config.py")

    res = model.eval(dataset)
    print(f"result:{res}\npretrain_ckpt={config.pretrain_ckpt}")


if __name__ == '__main__':
    eval_mobilenetv2()
