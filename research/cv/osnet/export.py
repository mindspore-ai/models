# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Export model"""

import os
import numpy as np


import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context


from src.osnet import create_osnet
from src.datasets_define import Market1501, DukeMTMCreID, MSMT17, CUHK03
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id

def init_dataset(name, **kwargs):
    """Initializes an image dataset."""
    __image_datasets = {
        'market1501': Market1501,
        'cuhk03': CUHK03,
        'dukemtmcreid': DukeMTMCreID,
        'msmt17': MSMT17,
    }
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export model for ascend310.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=get_device_id())
    dataset = init_dataset(name=config.target, root=config.data_path, mode='train',
                           cuhk03_labeled=config.cuhk03_labeled, cuhk03_classic_split=config.cuhk03_classic_split)
    num_classes = dataset.num_train_pids
    net = create_osnet(num_classes=num_classes)
    assert config.ckpt_file is not None, "config.ckpt_file is None."
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    input_arr = Tensor(np.ones([config.batch_size_test, 3, config.height, config.width]), ms.float32)
    export(net, input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
