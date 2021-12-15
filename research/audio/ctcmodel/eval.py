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

"""ctc evaluation"""

from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import CTCModel
from src.dataset import create_dataset
from src.model_for_eval import CTCEvalModel
from src.metric import LER
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

if config.enable_modelarts:
    device_id = get_device_id()
else:
    device_id = config.device_id
context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target=config.device_target,
    device_id=device_id
)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.test_path = config.local_test_path
    config.checkpoint_path = config.local_checkpoint_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''eval_function'''
    ds_test = create_dataset(config.test_path, False, config.test_batch_size)
    ckpt_file = config.checkpoint_path
    param_dict = load_checkpoint(ckpt_file)
    net = CTCModel(input_size=config.feature_dim, batch_size=config.test_batch_size, hidden_size=config.hidden_size,
                   num_class=config.n_class, num_layers=config.n_layer)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    eval_net = CTCEvalModel(net)
    model = Model(net, eval_network=eval_net, metrics={'ler': LER(beam=config.beam)})
    res = model.eval(ds_test, dataset_sink_mode=False)
    print(res)


if __name__ == "__main__":
    run_eval()
