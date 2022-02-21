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
'''eval'''
import os
import glob
import numpy as np
import PIL.Image as Image

from mindspore import context
from mindspore.common import set_seed

#from src.logger import get_logger
from src.dataset import AugmentNoise
from src.config import config as cfg

def test():
    '''test'''
    noise_generator = AugmentNoise(cfg.noisetype)

    for filename in os.listdir(cfg.test_dir):
        if filename == cfg.dataset:
            tem_path = os.path.join(cfg.test_dir, filename)
            out_dir = os.path.join(cfg.save_dir, filename)
            if not cfg.use_modelarts and not os.path.exists(out_dir):
                os.makedirs(out_dir)

            file_list = glob.glob(os.path.join(tem_path, '*'))

            for file in file_list:
                # read image
                img_clean = np.array(Image.open(file), dtype='float32') / 255.0

                img_test = noise_generator.add_noise(img_clean)
                H = img_test.shape[0]
                W = img_test.shape[1]
                val_size = (max(H, W) + 31) // 32 * 32
                img_test = np.pad(img_test,
                                  [[0, val_size - H], [0, val_size - W], [0, 0]],
                                  'reflect')

                img_clean = np.array(img_clean).astype(np.float32) #HWC
                img_test = np.array(img_test).astype(np.float32)   #HWC

                # predict
                img_clean = np.expand_dims(np.transpose(img_clean, (2, 0, 1)), 0)#NCHW
                img_test = np.expand_dims(np.transpose(img_test, (2, 0, 1)), 0)#NCHW

                # save images
                file_name = file.split('/')[-1].split('.')[0] + ".bin"   # get the name of image file
                img_test.tofile(os.path.join(out_dir, file_name))
        else:
            continue


if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.abspath(cfg.output_path)
    if not cfg.use_modelarts and not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, device_id=device_id, save_graphs=False)
    test()
