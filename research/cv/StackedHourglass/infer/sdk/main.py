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
Hourglass infer using SDK run in docker
"""
import time
import numpy as np
from eval.api import SdkApi
from eval.inference import get_img, infer, MPIIEval


if __name__ == '__main__':
    # init stream manager
    sdk_pipeline_name = b"im_hourglass"
    sdk_api = SdkApi()
    if not sdk_api.init():
        exit(-1)
    gts = []
    preds = []
    normalizing = []
    total_time = 0
    total_file = 0
    for anns, img, c, s, n in get_img():
        total_file += 1
        gts.append(anns)
        inp = img / 255
        input0 = np.array([inp]).astype(np.float32)
        input1 = np.array([inp[:, ::-1]]).astype(np.float32)
        sdk_api.send_tensor_input(sdk_pipeline_name, 0, "appsrc0", input0.tobytes(), [1, 256, 256, 3], 0)
        sdk_api.send_tensor_input(sdk_pipeline_name, 1, "appsrc1", input1.tobytes(), [1, 256, 256, 3], 0)
        start = time.time()
        result = sdk_api.get_result(sdk_pipeline_name)
        end = time.time()
        total_time += end - start
        det, ret = result[0], result[1]
        ans = infer(img, c, s, det, ret)
        if ans.size > 0:
            ans = ans[:, :, :3]
        pred = []
        for i in range(ans.shape[0]):
            pred.append({"keypoints": ans[i, :, :]})
        preds.append(pred)
        normalizing.append(n)

    mpii_eval = MPIIEval()
    mpii_eval.eval(preds, gts, normalizing)
    print("Infer images sum ", total_file, ", cost total time: ", total_time, "s")
    print("The throughput: ", total_file / total_time, " images/sec.")
