# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import mindspore as ms

from src.yolo import YOLOV3DarkNet53
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper, modelarts_export_preprocess


@moxing_wrapper(pre_process=modelarts_export_preprocess)
def run_export():
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        ms.set_context(device_id=config.device_id)
    network = YOLOV3DarkNet53(is_training=False)

    param_dict = ms.load_checkpoint(config.ckpt_file)
    ms.load_param_into_net(network, param_dict)

    network.set_train(False)

    shape = [config.batch_size, 3] + config.test_img_shape
    input_data = ms.numpy.zeros(shape, ms.float32)

    ms.export(network, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
    run_export()
