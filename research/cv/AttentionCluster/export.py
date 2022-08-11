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
"""export checkpoint file into mindir models"""
import numpy as np
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export, Tensor, context
from src.models.attention_cluster import AttentionCluster
from src.utils.config import config as cfg


if __name__ == "__main__":

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=cfg.device, device_id=cfg.device_id)

    fdim = [50]
    natt = [cfg.natt]
    nclass = 1024
    feature = Tensor(np.zeros([10240, 25, 50]).astype(np.float32))

    # define net
    net = AttentionCluster(fdims=fdim, natts=natt, nclass=nclass, fc=cfg.fc)

    # load checkpoint
    param_dict = load_checkpoint(ckpt_file_name=cfg.ckpt)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    export(net, feature, file_name='attention_cluster_{}'.format(cfg.natt), file_format='MINDIR')
