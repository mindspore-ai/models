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
"""export checkpoint file into mindir models"""
import numpy as np
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export, Tensor, context
from src.models.non_local import I3DResNet50
from src.models.resnet import resnet56
from src.utils.opts import parse_opts

if __name__ == "__main__":
    opt = parse_opts()

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target='Ascend', device_id=opt.device_id)
    if opt.dataset == 'kinetics':
        if opt.mode == 'multi':
            feature = Tensor(np.zeros([8, 10, 3, 3, 32, 256, 256]).astype(np.float32))
        else:
            feature = Tensor(np.zeros([8, 3, 32, 256, 256]).astype(np.float32))
        net = I3DResNet50()
    else:
        feature = Tensor(np.zeros([10000, 3, 32, 32]).astype(np.float32))
        net = resnet56(non_local=True)
    # load checkpoint
    param_dict = load_checkpoint(ckpt_file_name=opt.ckpt)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    export(net, feature, file_name='nl_{}'.format(opt.dataset), file_format='MINDIR')
