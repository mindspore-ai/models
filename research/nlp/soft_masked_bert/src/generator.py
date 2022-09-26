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

from utils import load_json
import numpy as np

class CscDatasetGenerator:
    def __init__(self, fp):
        self.data = load_json(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        original_text = self.data[index]['original_text']
        original_text_np = np.array(original_text)
        res = (original_text_np)
        return res
