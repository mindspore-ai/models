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
"""eval dice and Hausdorff distance"""
import os
import argparse
import numpy as np
from src.data_manager import DataManager
from src.config import vnet_cfg as cfg

parser = argparse.ArgumentParser(description='Vnet eval running')
parser.add_argument("--data_path", type=str, default="./promise/TestData", help="Path of dataset, default is ./promise")
parser.add_argument("--eval_split_file_path", type=str, default="./val.csv",
                    help="Path of dataset, default is ./split/eval.csv")
parser.add_argument("--output_path", type=str, default="./infer/data/infer_data",
                    help="Path of dataset, default is ./promise")


class InferImagelist:
    """infer data list"""

    def __init__(self, parameters, data_path, split_file_path):
        self.parameters = parameters
        self.dataManagerInfer = DataManager(split_file_path,
                                            data_path,
                                            self.parameters)

        self.dataManagerInfer.loadInferData()
        self.infer_images = self.dataManagerInfer.getNumpyImages()

    def __getitem__(self, index):
        img = None
        keysImg = list(self.infer_images.keys())
        img = self.infer_images[keysImg[index]]
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img, keysImg[index]

    def __len__(self):
        return len(list(self.infer_images.keys()))

def main():
    """Main entrance for eval"""
    args = parser.parse_args()
    dataInferlist = InferImagelist(cfg, args.data_path, args.eval_split_file_path)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    os.makedirs(os.path.join(args.output_path, 'img'), exist_ok=True)
    with open(os.path.join(args.output_path, "infer_anno.txt"), "w") as f:
        for i in range(dataInferlist.__len__()):
            img, img_id = dataInferlist.__getitem__(i)
            img.tofile(os.path.join(args.output_path, 'img', img_id + '.bin'))
            f.write(os.path.join(img_id + '.bin') + '\n')

if __name__ == '__main__':
    main()
