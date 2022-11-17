# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""export checkpoint file into air, onnx, mindir models"""
import numpy as np

import mindspore as ms
from mindspore import Tensor
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.FasterRcnn.faster_rcnn import FasterRcnn_Infer

ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, max_call_depth=2000)
if config.device_target == "Ascend":
    ms.set_context(device_id=get_device_id())


def modelarts_pre_process():
    pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def export_fasterrcnn():
    """ export_fasterrcnn """
    config.restore_bbox = True
    config.ori_h = None
    config.ori_w = None
    net = FasterRcnn_Infer(config=config)

    try:
        param_dict = ms.load_checkpoint(config.ckpt_file)
    except RuntimeError as ex:
        ex = str(ex)
        print("Traceback:\n", ex, flush=True)
        if "reg_scores.weight" in ex:
            exit("[ERROR] The loss calculation of faster_rcnn has been updated. "
                 "If the training is on an old version, please set `without_bg_loss` to False.")

    param_dict_new = {}
    for key, value in param_dict.items():
        key = key.replace("ncek", "neck")
        param_dict_new["network." + key] = value

    ms.load_param_into_net(net, param_dict_new)

    device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(ms.float16)

    img = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), ms.float32)
    img_metas = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, 4]), ms.float32)

    if not config.restore_bbox:
        print("[WARNING] When parameter 'restore_bbox' set to False, "
              "ascend310_infer of this project provided will not be available "
              "and need to complete 310 infer function by yourself.")
        ms.export(net, img, file_name=config.file_name, file_format=config.file_format)
    else:
        ms.export(net, img, img_metas, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_fasterrcnn()
