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

import os
import datetime
import libransac_voting as ransac_vote
from model_utils.config import config as cfg
from model_utils.data_file_utils import read_rgb_np
import numpy as np
from api.infer import SdkApi
from config import config as stream_cfg

def vote(seg_pred, ver_pred):
    """
    save infer result to the file, Write format:
        Object detected num is 5
        #Obj: 1, box: 453 369 473 391, confidence: 0.3, label: person, id: 0
        ...
    :param result_dir is the dir of save result
    :param result content bbox and class_id of all object
    """
    data = np.concatenate([seg_pred, ver_pred], 1)[0]
    channel = cfg.vote_num * 2 + 2
    ransac_vote.init_voting(cfg.img_height, cfg.img_width, channel, 2, cfg.vote_num)
    print('vote init success!----------------------------------------------------------')
    corner_pred = np.zeros((cfg.vote_num, 2), dtype=np.float32)
    ransac_vote.do_voting(data, corner_pred)
    print('do voting success!----------------------------------------------------------')
    return corner_pred


if __name__ == '__main__':
    # init stream manager
    pipeline_path = "./config/pvnet.pipeline"
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    # Construct the input of the stream

    img_data_plugin_id = 0
    re_dir = os.path.join("./data/", cfg.cls_name)
    image_dir = os.path.join(re_dir, 'images')
    test_fn = os.path.join(re_dir, 'test.txt')
    res_dir_name = './result'
    stream_name = b'pvnet'
    TENSOR_DTYPE_FLOAT32 = 0


    if not os.path.exists(os.path.join(res_dir_name, 'seg_pred')):
        os.makedirs(os.path.join(res_dir_name, 'seg_pred'))
    if not os.path.exists(os.path.join(res_dir_name, 'ver_pred')):
        os.makedirs(os.path.join(res_dir_name, 'ver_pred'))

    test_fns = np.loadtxt(test_fn, dtype=str)
    corner_preds = []
    poses = []
    for _, img_fn in enumerate(test_fns):
        rgb_path = os.path.join(image_dir, img_fn)

        rgb = read_rgb_np(rgb_path).reshape(1, 480, 640, 3)
        sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0",
                                  rgb.tobytes(), rgb.shape, stream_cfg.TENSOR_DTYPE_FLOAT32)
        start_time = datetime.datetime.now()
        result = sdk_api.get_result(stream_name)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        seg_result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                                   dtype=np.float32).reshape(1, -1, 480, 640)
        ver_result = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr,
                                   dtype=np.float32).reshape(1, -1, 480, 640)
        seg_result.tofile(os.path.join(res_dir_name, 'seg_pred', img_fn.split('.')[0]+'.bin'))
        ver_result.tofile(os.path.join(res_dir_name, 'ver_pred', img_fn.split('.')[0]+'.bin'))
