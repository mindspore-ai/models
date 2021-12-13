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
"""
##############test MMoE example on census-income.data#################
python eval.py
"""

from sklearn.metrics import roc_auc_score

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.load_dataset import create_dataset
from src.mmoe import MMoE
from src.model_utils.moxing_adapter import moxing_wrapper

from mindspore import context
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init


def modelarts_process():
    config.ckpt_path = config.ckpt_file


@moxing_wrapper(pre_process=modelarts_process)
def eval_mmoe():
    """MMoE eval"""
    device_num = get_device_num()
    if device_num > 1:
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
        if config.device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
        elif config.device_target == "GPU":
            init()

    ds_eval = create_dataset(data_path=config.data_path, batch_size=config.batch_size,
                             training=False, target=config.device_target)
    eval_dataloader = ds_eval.create_tuple_iterator()
    if ds_eval.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    print("ds_eval_size", ds_eval.get_dataset_size())

    net = MMoE(num_features=config.num_features, num_experts=config.num_experts, units=config.units)

    param_dict = load_checkpoint(config.ckpt_path)
    print("load checkpoint from [{}].".format(config.ckpt_path))
    load_param_into_net(net, param_dict)
    net.set_train(False)

    income_auc = 0
    marital_auc = 0
    for data, income_label, marital_label in eval_dataloader:
        output = net(Tensor(data, mstype.float16))
        income_output = output[0].asnumpy()
        income_output = income_output.flatten().tolist()

        marital_output = output[1].asnumpy()
        marital_output = marital_output.flatten().tolist()

        income_label = income_label.asnumpy()
        income_label = income_label.flatten().tolist()

        marital_label = marital_label.asnumpy()
        marital_label = marital_label.flatten().tolist()

        if len(income_output) != len(income_label):
            raise RuntimeError('income_output.size() is not equal income_label.size().')
        if len(marital_output) != len(marital_label):
            raise RuntimeError('marital_output.size is not equal marital_label.size().')

        income_auc = roc_auc_score(income_label, income_output)
        marital_auc = roc_auc_score(marital_label, marital_output)

    results = [[income_auc], [marital_auc]]
    print("result : {}".format(results))


if __name__ == "__main__":
    eval_mmoe()
