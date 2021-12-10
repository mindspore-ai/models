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
"""Export"""
import os
import numpy as np
from mindspore.context import set_context, GRAPH_MODE
from mindspore.train.serialization import export
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from src.model_utils.device_adapter import get_device_id
from src.model.net import TSM
from src.model_utils.config import config

def main():

    args = config
    set_context(mode=GRAPH_MODE, device_target=args.device_target, device_id=get_device_id())
    num_class = 174

    base_model = args.arch
    tsm = TSM(num_class, args.num_segments, args.modality,
              base_model=base_model,
              consensus_type=args.consensus_type,
              dropout=args.dropout,
              img_feature_dim=args.img_feature_dim,
              partial_bn=not args.no_partialbn,
              pretrain=args.pretrain,
              is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
              fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
              temporal_pool=args.temporal_pool,
              non_local=args.non_local)
    tsm.set_train(False)
    check_point = os.path.join(args.checkpoint_path, args.test_filename)
    param_dict = load_checkpoint(check_point)
    load_param_into_net(tsm, param_dict)

    input_shp = [1, 24, 224, 224]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    if args.enable_modelarts:
        import moxing as mox
        export(tsm, input_array, file_name='./export_model/tsm', file_format='MINDIR')
        mox.file.copy_parallel(src_url='./export_model/', dst_url=args.train_url)
    else:
        export(tsm, input_array, file_name='tsm', file_format='MINDIR')

if __name__ == '__main__':
    main()
