# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
Inference parameter configuration
"""
MODEL_WIDTH = 512
MODEL_HEIGHT = 512
NUM_CLASSES = 80
SCORE_THRESH = 0.3
STREAM_NAME = "im_centernet"

INFER_TIMEOUT = 100000

TENSOR_DTYPE_FLOAT32 = 0
TENSOR_DTYPE_FLOAT16 = 1
TENSOR_DTYPE_INT8 = 2
