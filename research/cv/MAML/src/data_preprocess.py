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

import os.path
import argparse
from Omniglot import Omniglot
import numpy as np
import  torchvision.transforms as transforms
from    PIL import Image

class DataProcess():
    def __init__(self, root, imgsz):
        self.imgsz = imgsz
        self.root = root
    def process(self):
        self.x = Omniglot(self.root, download=False,
                          transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                        lambda x: x.resize((self.imgsz, self.imgsz)),
                                                        lambda x: np.reshape(x, (self.imgsz, self.imgsz, 1)),
                                                        lambda x: np.transpose(x, [2, 0, 1]),
                                                        lambda x: x/255.])
                         )

        temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        for (img, label) in self.x:
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label] = [img]

        self.x = []
        for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
            self.x.append(np.array(imgs))
        # as different class may have different number of imgs
        self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
        # each character contains 20 imgs
        print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
        temp = []  # Free memory
        # save all dataset into npy file.
        np.save(os.path.join(self.root, 'omniglot.npy'), self.x)
        print('write into omniglot.npy.')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, help='path of data', default='your/path/omniglot')
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    args = argparser.parse_args()
    dataProcess = DataProcess(root=args.data_path, imgsz=args.imgsz)
    dataProcess.process()
    print("process finished!!!!!!!!!!!!!!!!!!")
