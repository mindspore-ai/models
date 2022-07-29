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
'''post process for 310 inference'''

import os
import time
import numpy as np
from src.config import config


class CalcAccuracy():

    def __call__(self, result_path, label_path):

        files = os.listdir(result_path)
        MPJPE = []
        for i, file in enumerate(files):
            full_file_result = os.path.join(result_path, file)
            full_file_label = os.path.join(
                label_path, files[i].replace(
                    '_0', ''))
            if os.path.isfile(full_file_result):
                data_predict = np.fromfile(full_file_result, dtype=np.float32)
                predict_j3d = data_predict.reshape(1, -1, 3)
                data_3d_label = np.fromfile(full_file_label, dtype=np.float32)
                real_3d = data_3d_label[42:42 + 42].reshape(1, -1, 3)
                loss_kp_3d = self.batch_kp_3d_l2_loss(
                    real_3d, predict_j3d[:, :14, :], 1) * 1000
                MPJPE.append(loss_kp_3d)

        return MPJPE

    def batch_kp_3d_l2_loss(self, real_3d_kp, fake_3d_kp, w_3d):
        shape = real_3d_kp.shape
        k = np.sum(w_3d, axis=0) * shape[1] * 3.0 * 2.0 + 1e-8

        real_3d_kp, fake_3d_kp = self.align_by_pelvis(
            real_3d_kp), self.align_by_pelvis(fake_3d_kp)
        kp_gt = real_3d_kp
        kp_pred = fake_3d_kp
        kp_dif = (kp_gt - kp_pred)**2

        return np.multiply(kp_dif.sum(1).sum(1), w_3d) * 1.0 / k

    def align_by_pelvis(self, joints):

        joints = np.reshape(joints, (joints.shape[0], 14, 3))
        pelvis = (joints[:, 3, :] + joints[:, 2, :]) / 2.0
        return joints - np.expand_dims(pelvis, 1)


if __name__ == "__main__":
    Calc_Acc = CalcAccuracy()
    if config.dataset.lower() == "human3.6m":
        t1 = time.time()
        Acc = Calc_Acc(config.result_path, config.label_file)
        print('PA-MPJPE is : ', np.mean(np.array(Acc)))
        t2 = time.time()
        print('time:', t2 - t1)
