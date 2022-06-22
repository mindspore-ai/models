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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""

import mindspore as ms
from mindspore import context, Tensor, export
import numpy as np


from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from src.nnunet.training.model_restore import load_model_and_checkpoint_files

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)


def modelarts_process():
    """model arts process"""
    return

@moxing_wrapper(pre_process=modelarts_process)
def export_nnUNet():
    """ export_nnUNet """

    # model is a str
    # nnUnet need read pkl from disk
    model = config.model
    checkpoint_name = config.checkpoint_file
    trainer, params = load_model_and_checkpoint_files(model, folds=0, mixed_precision=False,
                                                      checkpoint_name=checkpoint_name)
    export_file(params, trainer)



def export_file(params, trainer):
    """
    export param logic
    """
    trainer.load_checkpoint_ram(params[0], False)
    input_arr = Tensor(np.zeros([config.batch_size, config.channel, config.depth, config.height, config.width]),
                       ms.float32)
    export(trainer.network, input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_nnUNet()
