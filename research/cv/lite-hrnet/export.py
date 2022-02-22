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
"""
Export checkpoints into MINDIR model files
"""
import os
import argparse
import numpy as np
from mindspore import export, load_checkpoint, Tensor, context
from src.config import experiment_cfg, model_cfg
from src.model import get_posenet_model


# Set DEVICE_ID
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
parser.add_argument("--train_url", type=str, default='./checkpoints/', help="Storage path of training results.")
args = parser.parse_args()


def main():
    local_train_url = args.train_url
    model_name = experiment_cfg['model_config']
    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target=experiment_cfg['device'], device_id=args.device_id)
    # Create network
    model = get_posenet_model(model_cfg.model)
    # Load parameters from checkpoint into network
    last_checkpoint = os.path.join(local_train_url, experiment_cfg['experiment_tag'], f"{model_name}-final.ckpt")
    print(f'Loading checkpoint from {last_checkpoint}')
    load_checkpoint(last_checkpoint, net=model)
    # Initialize dummy inputs
    inputs = np.random.uniform(0.0, 1.0, size=[model_cfg.data['samples_per_gpu'],
                                               3,
                                               model_cfg.data_cfg['image_size'][1],
                                               model_cfg.data_cfg['image_size'][0]]).astype(np.float32)
    # Export network into MINDIR model file
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    file_name = model_name + "_" + experiment_cfg['experiment_tag']
    path = os.path.join('outputs', file_name)
    export(model, Tensor(inputs), file_name=path, file_format='MINDIR')
    print("==========================================")
    print(file_name + ".mindir exported successfully!")
    print("==========================================")

if __name__ == "__main__":
    main()
