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
preprocess dataset
"""

import argparse
import shutil
from pathlib import Path


def copy(src, dst):
    """copy"""
    dst_blur = Path(dst, 'blur')
    dst_blur.mkdir(parents=True, exist_ok=True)
    dst_sharp = Path(dst, 'sharp')
    dst_sharp.mkdir(parents=True, exist_ok=True)

    src = Path(src)
    for num, f_path in enumerate(sorted(src.rglob('*blur/*'))):
        print(f_path, f_path.name)
        print(f_path.parts[-3])
        shutil.copy(f_path, dst_blur / f'{num + 1}.png')

    for num, f_path in enumerate(sorted(src.rglob('*sharp/*'))):
        print(f_path, f_path.name)
        print(f_path.parts[-3])
        shutil.copy(f_path, dst_sharp / f'{num + 1}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_src', type=str)
    parser.add_argument('--root_dst', type=str)

    args = parser.parse_args()

    copy(Path(args.root_src, 'train'), Path(args.root_dst, 'train'))
    copy(Path(args.root_src, 'test'), Path(args.root_dst, 'test'))
