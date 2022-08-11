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
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, export
from src.model_utils.config import config
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                    device_id=config.device_id, save_graphs=False)
def run_export():
    if config.dataset == "cifar10":
        from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
    elif config.dataset == "imagenet":
        from model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel
    else:
        raise ValueError("Unsupported dataset.")

    net = ResidualAttentionModel()

    assert config.ckpt_file is not None, "ckpt_file is None."
    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, config.width, config.height]), ms.float32)
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)
    if config.enable_modelarts:
        import moxing as mox
        workroot = '/home/work/user-job-dir/'
        obs_train_url = config.train_url
        mox.file.copy_parallel(workroot, obs_train_url)
        print("Successfully Upload {} to {}".format(workroot, obs_train_url))

if __name__ == '__main__':
    run_export()
