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
""" infer STGAN ONNX"""
import os
import onnxruntime
import numpy as np
import tqdm
import cv2

from mindspore.common import set_seed

from src.utils import get_args
from src.dataset import CelebADataLoader

set_seed(1)


def get_input_name(onnx_session):
    """
    input_name = onnx_session.get_inputs()[0].name
    :param onnx_session:
    :return:
    """
    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)
    return input_name


def run_eval():
    """ eval onnx model function """
    args = get_args("test")
    print('\n\n=============== start eval onnx model ===============\n\n')
    data_loader = CelebADataLoader(args.dataroot,
                                   mode=args.phase,
                                   selected_attrs=args.attrs,
                                   batch_size=1,
                                   image_size=args.image_size)
    iter_per_epoch = len(data_loader)
    args.dataset_size = iter_per_epoch
    ## onnx model export
    onnx_path = args.onnx_path
    session = onnxruntime.InferenceSession(onnx_path)

    for _ in tqdm.trange(iter_per_epoch, desc='Eval onnx model Loop'):

        data = next(data_loader.test_loader)
        input_image = data['image'].asnumpy()
        input_label = data['label'].asnumpy()
        filename = data_loader.test_set.get_current_filename()
        fake_image = session.run(None, {get_input_name(session)[0]: input_image,
                                        get_input_name(session)[1]: input_label})
        images = fake_image[0]
        final_imgs = np.transpose(images, (0, 2, 3, 1))
        final_imgs = np.squeeze(final_imgs)
        out_path = args.onnx_output
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        cv2.imwrite(os.path.join(out_path, str(filename)), cv2.cvtColor(final_imgs * 255, cv2.COLOR_BGR2RGB))


    print('\n\n=============== finish eval onnx model ===============\n\n')

if __name__ == '__main__':
    run_eval()
