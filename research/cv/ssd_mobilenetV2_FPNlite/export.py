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

"""export"""
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
import mindspore.common.dtype as mstype
from src.ssd import SsdInferWithDecoder, ssd_mobilenet_v2_fpn
from src.box_utils import default_boxes
from src.model_utils.config import config as cfg

context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
if cfg.device_target != "GPU":
    context.set_context(device_id=cfg.device_id)

if __name__ == '__main__':
    net = ssd_mobilenet_v2_fpn(config=cfg)
    net = SsdInferWithDecoder(net, Tensor(default_boxes), cfg)

    param_dict = load_checkpoint(cfg.ckpt_file)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [cfg.batch_size, 3] + cfg.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp), mstype.float32)
    export(net, input_array, file_name=cfg.file_name, file_format=cfg.file_format)
