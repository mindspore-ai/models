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
import argparse
import os
import glob
import numpy as np
import PIL.Image as Image

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True,
                    help='directory to store the image with noise')
parser.add_argument('--image_path', type=str, required=True,
                    help='directory of image to add noise')
parser.add_argument("--image_height", type=int, default=500, help="resized image height.")
parser.add_argument("--image_width", type=int, default=500, help="resized image width.")
parser.add_argument('--channel', type=int, default=3
                    , help='image channel, 3 for color, 1 for gray')
parser.add_argument('--sigma', type=int, default=15, help='level of noise')
args = parser.parse_args()

def add_noise():
    image_list = glob.glob(os.path.join(args.image_path, '*'))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for image in sorted(image_list):
        print("Adding noise to: ", image)
        # read image
        if args.channel == 3:
            img_clean = np.array(Image.open(image).resize((args.image_width, \
                                 args.image_height), Image.ANTIALIAS), dtype='float32') / 255.0
        else:
            assert args.channel == 1
            img_clean = np.expand_dims(np.array(Image.open(image).resize((args.image_width, \
                        args.image_height), Image.ANTIALIAS).convert('L'), dtype='float32') / 255.0, axis=2)

        np.random.seed(0) #obtain the same random data when it is in the test phase
        img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape).astype(np.float32)#HWC
        img_test = np.expand_dims(img_test.transpose((2, 0, 1)), 0)#NCHW
        filename = image.split('/')[-1].split('.')[0]    # get the name of image file
        img_test.tofile(os.path.join(args.out_dir, filename+'_noise.bin'))

if __name__ == "__main__":
    add_noise()
