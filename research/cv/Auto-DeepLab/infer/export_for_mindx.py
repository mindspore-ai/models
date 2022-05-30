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
# ===========================================================================
"""export checkpoint file into mindir models"""
import sys
import os
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export


context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.core.model import AutoDeepLab
    from util.mindx_config import obtain_autodeeplab_args
    from util.mindx_utils import InferWithFlipNetwork

    args = obtain_autodeeplab_args()
    args.total_iters = 0

    # net
    autodeeplab = AutoDeepLab(args)

    # load checkpoint
    param_dict = load_checkpoint(args.ckpt_name)

    # load the parameter into net
    load_param_into_net(autodeeplab, param_dict)
    network = InferWithFlipNetwork(autodeeplab, flip=args.infer_flip, input_format=args.input_format)

    input_data = np.random.uniform(0.0, 1.0, size=[1, 1024, 2048, 3]).astype(np.float32)
    export(network, Tensor(input_data), file_name='Auto-DeepLab-s_NHWC_BGR', file_format=args.file_format)
