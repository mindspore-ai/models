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
Export CenterNet mindir model.
"""

import os
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src import CenterNetMultiPoseEval
from src.dataset import COCOHP
from src.model_utils.config import config, net_config, eval_config, export_config, dataset_config
from src.model_utils.moxing_adapter import moxing_wrapper

def modelarts_pre_process():
    '''modelarts pre process function.'''
    export_config.ckpt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_config.ckpt_file)
    export_config.export_name = os.path.join(config.output_path, export_config.export_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''export function'''
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=config.device_id)
    net = CenterNetMultiPoseEval(net_config, eval_config.K)
    param_dict = load_checkpoint(export_config.ckpt_file)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    coco = COCOHP(dataset_config, run_mode=config.run_mode, net_opt=net_config,
                  enable_visual_image=(config.visual_image == "true"), save_path=config.save_result_dir)
    coco.init(config.data_dir, keep_res=eval_config.keep_res)
    dataset = coco.create_eval_dataset()

    index = 0
    for data in dataset.create_dict_iterator(num_epochs=1):
        index += 1
        image = data['image']
        image_id = data['image_id'].asnumpy().reshape((-1))[0]
        for scale in eval_config.multi_scales:
            images, _ = coco.pre_process_for_test(image.asnumpy(), image_id, scale)
            _, _, h, w = images.shape
            print(images.shape)
            export_path = str(h) + '_' + str(w)
            input_data = Tensor(np.ones([1, 3, h, w]), mindspore.float32)
            export(net, input_data, file_name=export_path, file_format="ONNX")

if __name__ == '__main__':
    run_export()
