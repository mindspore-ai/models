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
"""eval_onnx"""

import os
import onnxruntime as ort
from PIL import Image
import numpy as np

from src.data import create_dataset
from src.data.single_dataloader import single_dataloader
from src.option.options_test import TestOptions


def create_session(onnx_path, target_device='GPU'):
    """Create ONNX runtime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [x.name for x in session.get_inputs()]
    return session, input_names


class Eval:
    """ Eval """
    @staticmethod
    def save_image(pic_tensor, pic_path="test.png"):
        """ save image """
        pic_np = pic_tensor[0]
        if pic_np.shape[0] == 1:
            pic_np = (pic_np[0] + 1) / 2.0 * 255.0
        elif pic_np.shape[0] == 3:
            pic_np = (np.transpose(pic_np, (1, 2, 0)) + 1) / 2.0 * 255.0
        pic = Image.fromarray(pic_np)
        pic = pic.convert('RGB')
        pic.save(pic_path)
        print(pic_path + ' is saved.')

    @staticmethod
    def expand_tensor_data(data_tensor):
        """expand_tensor_data"""
        data_out = np.expand_dims(data_tensor, axis=0)
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
    if not os.path.exists(opt.results_dir):
        os.mkdir(opt.results_dir)

    dataset = create_dataset(opt)

    for data in dataset.create_dict_iterator(output_numpy=True):
        input_data = {}
        item = single_dataloader(data, opt)
        for d, v in item.items():
            if d in ('A_paths', 'B_paths'):
                input_data[d] = v
            else:
                input_data[d] = v[0]
        data_info = Eval.process_input(input_data, opt.results_dir)
        real_A = data_info['A']
        real_A_bg = data_info['bg_A']
        real_A_eyel = data_info['eyel_A']
        real_A_eyer = data_info['eyer_A']
        real_A_nose = data_info['nose_A']
        real_A_mouth = data_info['mouth_A']
        real_A_hair = data_info['hair_A']
        mask = data_info['mask']
        mask2 = data_info['mask2']
        center = data_info['center'][0]

        onnx_name = opt.onnx_path + opt.onnx_filename + '_' + str(center[0][0]) + '_' + str(center[0][1]) + '_' +\
                    str(center[1][0]) +  '_' + str(center[1][1]) + '_' + str(center[2][0]) + '_' + str(center[2][1]) +\
                    '_' + str(center[3][0]) + '_' + str(center[3][1]) + '.onnx'

        sess, [real_A_name, real_A_bg_name, real_A_eyel_name, real_A_eyer_name, real_A_nose_name, real_A_mouth_name,
               real_A_hair_name, mask_name, mask2_name] = create_session(onnx_name)

        [result] = sess.run(None, {real_A_name: real_A, real_A_bg_name: real_A_bg, real_A_eyel_name: real_A_eyel,
                                   real_A_eyer_name: real_A_eyer, real_A_nose_name: real_A_nose, real_A_mouth_name:
                                       real_A_mouth, real_A_hair_name: real_A_hair, mask_name: mask, mask2_name: mask2})

        Eval.save_image(result, data_info['out_path'])
