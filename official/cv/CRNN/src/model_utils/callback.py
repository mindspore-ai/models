# Copyright 2023 Huawei Technologies Co., Ltd
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
"""crnn callback."""
from mindspore.train.callback import Callback

class ResumeCallback(Callback):
    def __init__(self, start_epoch=0):
        super(ResumeCallback, self).__init__()
        self.start_epoch = start_epoch

    def epoch_begin(self, run_context):
        run_context.original_args().cur_epoch_num += self.start_epoch
