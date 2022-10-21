# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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


from __future__ import print_function
import os
import sys
import glob
import scipy.io
from PIL import Image as PILImage

def main():
    input_path, output_path = process_arguments(sys.argv)

    if os.path.isdir(input_path) and os.path.isdir(output_path):
    # glob.blob 返回所有匹配的文件路径列表
        mat_files = glob.glob(os.path.join(input_path, '*.mat'))
        convert_mat2png(mat_files, output_path)
    else:
        helps('Input or output path does not exist!\n')

def mat2png_hariharan(mat_file, key='GTcls'):
    mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
    return mat[key].Segmentation

def process_arguments(argv):
    num_args = len(argv)

    input_path = None
    output_path = None

    if num_args == 3:
        input_path = argv[1]
        output_path = argv[2]
    else:
        help()

    return input_path, output_path

def convert_mat2png(mat_files, output_path):
    if not mat_files:
        help('Input directory does not contain any Matlab files!\n')

    for mat in mat_files:
        numpy_img = mat2png_hariharan(mat)
        pil_img = PILImage.fromarray(numpy_img)
        pil_img.save(os.path.join(output_path, modify_image_name(mat, 'png')))

# Extract name of image from given path, replace its extension with specified one
# and return new name only, not path.
def modify_image_name(path, ext):
    return os.path.basename(path).split('.')[0] + '.' + ext

def helps(msg=''):
    print(msg +
          'Usage: python mat2png.py INPUT_PATH OUTPUT_PATH\n'
          'INPUT_PATH denotes path containing Matlab files for conversion.\n'
          'OUTPUT_PATH denotes path where converted Png files ar going to be saved.'
          , file=sys.stderr)

    exit()

if __name__ == '__main__':
    main()
