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
"""FastFlow Anomaly Map Generator Implementation."""

import cv2
import numpy as np

def anomaly_map_generator(hidden_variables, output_size=256):
    flow_maps = []
    for hidden_variable in hidden_variables:
        if not isinstance(hidden_variable, np.ndarray):
            hidden_variable = hidden_variable.asnumpy()
        log_prob = - np.mean(hidden_variable ** 2, axis=1) * 0.5
        prob = np.exp(log_prob).transpose(1, 2, 0)
        flow_map = cv2.resize(-prob, dsize=(output_size, output_size), interpolation=cv2.INTER_LINEAR)
        if len(flow_map.shape) == 2:
            flow_map = np.expand_dims(flow_map, axis=-1)
        flow_map = flow_map.transpose(2, 0, 1)
        flow_maps.append(flow_map)
    flow_maps = np.stack(flow_maps, axis=-1)
    anomaly_map = np.mean(flow_maps, axis=-1)

    return anomaly_map
