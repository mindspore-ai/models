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
""" Test centerface example """
import os
import scipy.io as sio
import onnxruntime
from src.model_utils.config import config
from dependency.centernet.src.lib.detectors.base_detector import CenterFaceDetector
from dependency.evaluate.eval import evaluation


def create_session(onnx_path, target_device):
    """" Create ONNX runtime session """
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {config.device_target},'
            f'Expected one of: "CPU", "GPU", "Ascend"'
        )
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return session

def test_centerface():
    """ test onnx """
    ground_truth_mat = sio.loadmat(config.ground_truth_mat)
    event_list = ground_truth_mat['event_list']
    file_list = ground_truth_mat['file_list']
    save_path = config.save_dir + '/'

    sess = create_session(config.onnx_path, config.device_target)
    detector = CenterFaceDetector(config, sess)

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
            print('save_path + im_dir={}'.format(save_path + im_dir))
        for num, file_obj in enumerate(file_list_item):
            im_name = file_obj[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            img_path = os.path.join(config.data_dir, zip_name)
            print('img_path={}'.format(img_path))

            dets = detector.run_onnx(img_path)['results']

            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:{}, num:{}'.format(index + 1, num + 1))

    if config.eval:
        print('==========start eval===============')
        print("test output path = {}".format(save_path))
        if os.path.isdir(save_path):
            evaluation(save_path, config.ground_truth_path)
        else:
            print('no test output path')
        print('==========end eval===============')

    print('==========end testing===============')


if __name__ == "__main__":
    test_centerface()
