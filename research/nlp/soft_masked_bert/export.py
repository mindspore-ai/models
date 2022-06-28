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
"""export mindir file"""
import argparse
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from src.soft_masked_bert import SoftMaskedBertCLS

def run_csc():
    """run csc task"""
    parser = argparse.ArgumentParser(description="run csc")
    parser.add_argument("--device", type=str, default="Ascend")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bert_ckpt", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--O3", default=False, action='store_true')
    args_opt = parser.parse_args()
    # context setting
    if args_opt.device == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    else:
        raise Exception("Only support on Ascend currently.")
    ckpt_path = './weight/' + args_opt.bert_ckpt
    netwithloss = SoftMaskedBertCLS(args_opt.batch_size, is_training=False, \
    if_O3=args_opt.O3, load_checkpoint_path=ckpt_path)
    param_dict = load_checkpoint(args_opt.ckpt_dir)
    load_param_into_net(netwithloss, param_dict)
    t = ms.Tensor(np.ones([2, 512]).astype(np.int32))
    input1 = [t, t, t, t, t, t, t]
    export(netwithloss, *input1, file_name='smb', file_format='MINDIR')
if __name__ == "__main__":
    run_csc()
