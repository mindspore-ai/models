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
"""310 preprocess script."""

import os
import os.path as osp

from model_utils.config import config as default_config

from src.dataset import LoadImages
from src.log import logger

_MOT16_VALIDATION_FOLDERS = (
    'MOT16-02',
    'MOT16-04',
    'MOT16-05',
    'MOT16-09',
    'MOT16-10',
    'MOT16-11',
    'MOT16-13',
)

def main(
        opt,
        data_root,
        seqs,
        exp_name,
        save_videos=False,
):
    for seq in seqs:
        logger.info('start seq: %s', seq)
        dataloader = LoadImages(osp.join(data_root, seq, 'img1'), opt)
        print('img num:', len(dataloader))
        for i, (img, _) in  enumerate(dataloader):
            path = "./pre/{}/{:06}.bin".format(seq, i)
            if os.path.exists(path):
                continue
            dirs = os.path.dirname(path)
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            img.tofile(path)


if __name__ == '__main__':
    config = default_config
    config.img_size = [1088, 608]

    data_root_path = config.dataset_root

    if not os.path.isdir(data_root_path):
        raise NotADirectoryError(
            f'Cannot find "{data_root_path}" subdirectory '
            f'in the specified dataset root "{config.dataset_root}"'
        )

    main(
        config,
        data_root=data_root_path,
        seqs=_MOT16_VALIDATION_FOLDERS,
        exp_name=config.ckpt_url.split('/')[-2],
        save_videos=False,
    )
