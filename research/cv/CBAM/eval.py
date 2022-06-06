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
##############test CBAM example on dataset#################
python eval.py
"""

from mindspore import context
from mindspore.train import Model
from mindspore.communication.management import init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.metrics import Accuracy
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.data import create_dataset
from src.model import resnet50_cbam


def modelarts_process():
    config.ckpt_path = config.ckpt_file


@moxing_wrapper(pre_process=modelarts_process)
def eval_():
    """ model eval """
    device_num = get_device_num()
    if device_num > 1:
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
        if config.device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif config.device_target == "GPU":
            init()
    print("================init finished=====================")
    ds_eval = create_dataset(data_path=config.data_path, batch_size=config.batch_size,
                             training=False, snr=2, target=config.device_target)
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size.")
    print("ds_eval_size", ds_eval.get_dataset_size())
    print("===============create dataset finished==============")

    net = resnet50_cbam(phase="test")
    param_dict = load_checkpoint(config.ckpt_path)
    print("load checkpoint from [{}].".format(config.ckpt_path))
    load_param_into_net(net, param_dict)
    net.set_train(False)
    loss = SoftmaxCrossEntropyWithLogits()
    model = Model(net, loss_fn=loss, metrics={"Accuracy": Accuracy()})
    result = model.eval(ds_eval, dataset_sink_mode=False)
    print("===================result: {}==========================".format(result))


if __name__ == '__main__':
    eval_()
