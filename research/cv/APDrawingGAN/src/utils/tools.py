# Copyright 2021 Huawei Technologies Co., Ltd
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
"""tools"""

import os
import datetime
import moxing as mox

def obs_data2modelarts(args):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args.data_url, args.modelarts_data_dir))
    mox.file.copy_parallel(src_url=args.data_url, dst_url=args.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(args.modelarts_data_dir)
    print("===>>>Files:", files)

def modelarts_result2obs(args):
    """
    Copy result data from modelarts to obs.
    """
    mox.file.copy_parallel(src_url=args.modelarts_result_dir, dst_url=args.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(args.modelarts_result_dir,
                                                                                  args.train_url))
    files = os.listdir()
    print("===>>>current Files:", files)
