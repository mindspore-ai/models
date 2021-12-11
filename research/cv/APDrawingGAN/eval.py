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
"""eval"""

import os
import mindspore
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context, Tensor
from PIL import Image
import numpy as np
from src.models.APDrawingGAN_G import Generator
from src.data import create_dataset
from src.data.single_dataloader import single_dataloader
from src.option.options_test import TestOptions

context.set_context(mode=context.GRAPH_MODE)

class Eval:
    """ Eval """
    @staticmethod
    def save_image(pic_tensor, pic_path="test.png"):
        """ save image """
        pic_np = pic_tensor.asnumpy()[0]
        if pic_np.shape[0] == 1:
            pic_np = (pic_np[0] + 1) / 2.0 * 255.0
        elif pic_np.shape[0] == 3:
            pic_np = (np.transpose(pic_np, (1, 2, 0)) + 1) / 2.0 * 255.0
        pic = Image.fromarray(pic_np)
        pic = pic.convert('RGB')
        pic.save(pic_path)
        print(pic_path + ' is saved.')

    @staticmethod
    def infer_one_image(net, all_data):
        """ infer one image """
        real_A = all_data['A']
        real_A_bg = all_data['bg_A']
        real_A_eyel = all_data['eyel_A']
        real_A_eyer = all_data['eyer_A']
        real_A_nose = all_data['nose_A']
        real_A_mouth = all_data['mouth_A']
        real_A_hair = all_data['hair_A']
        mask = all_data['mask']
        mask2 = all_data['mask2']
        center = all_data['center']
        net.set_pad(center[0])
        result = net.construct(real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose, real_A_mouth, real_A_hair,
                               mask, mask2)
        Eval.save_image(result[0], all_data['out_path'])

    @staticmethod
    def expand_tensor_data(data_tensor):
        """expand_tensor_data"""
        tmp = np.expand_dims(data_tensor, axis=0)
        data_out = Tensor(tmp, mindspore.float32)
        return data_out

    @staticmethod
    def process_input(all_data, result_dir):
        """process_input"""
        all_data['A'] = Eval.expand_tensor_data(all_data['A'])
        all_data['bg_A'] = Eval.expand_tensor_data(all_data['bg_A'])
        all_data['eyel_A'] = Eval.expand_tensor_data(all_data['eyel_A'])
        all_data['eyer_A'] = Eval.expand_tensor_data(all_data['eyer_A'])
        all_data['nose_A'] = Eval.expand_tensor_data(all_data['nose_A'])
        all_data['mouth_A'] = Eval.expand_tensor_data(all_data['mouth_A'])
        all_data['hair_A'] = Eval.expand_tensor_data(all_data['hair_A'])
        all_data['mask'] = Eval.expand_tensor_data(all_data['mask'])
        all_data['mask2'] = Eval.expand_tensor_data(all_data['mask2'])
        all_data['center'] = np.expand_dims(all_data['center'], axis=0)
        pic_path = all_data['A_path']
        pic_path = pic_path[pic_path.rfind('/') + 1:-1]
        all_data['out_path'] = os.path.join(result_dir, pic_path)
        return all_data

if __name__ == "__main__":
    opt = TestOptions().get_settings()
    opt.rank = 0
    opt.group_size = 1
    context.set_context(device_id=opt.device_id, device_target=opt.device_target)
    if not os.path.exists(opt.results_dir):
        os.mkdir(opt.results_dir)

    models = Generator(opt)
    param_dict = load_checkpoint(opt.model_path)
    load_param_into_net(models, param_dict)

    dataset = create_dataset(opt)
    for data in dataset.create_dict_iterator(output_numpy=True):
        input_data = {}
        item = single_dataloader(data, opt)
        for d, v in item.items():
            if d in ('A_paths', 'B_paths'):
                input_data[d] = v
            else:
                input_data[d] = v[0]
        Eval.infer_one_image(models, Eval.process_input(input_data, opt.results_dir))
    if opt.isModelarts:
        from src.utils.tools import modelarts_result2obs
        modelarts_result2obs(opt)
