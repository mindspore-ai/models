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
""" Visualizer """

import os
import ntpath
from . import util


class Visualizer:
    """visualizer"""

    def __init__(self, opt):
        self.opt = opt

    def convert_visuals_to_numpy(self, visuals):
        """convert visuals to numpy"""
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if key == 'input_label':
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, img_dir, visuals, image_path):
        """save images"""
        visuals = self.convert_visuals_to_numpy(visuals)

        short_path = ntpath.basename(image_path)
        name = os.path.splitext(short_path)[0]

        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(img_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
