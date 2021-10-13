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
"""
Export checkpoints into MINDIR models
"""
import os
import numpy as np
from src.gan import Generator
from src.param_parse import parameter_parser

from mindspore import export, load_checkpoint, load_param_into_net, Tensor, context
from mindspore.common import dtype as mstype

# Set DEVICE_ID
args = parameter_parser()

# Runtime
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# Create network
generator = Generator(args.latent_dim)

# Initialize dummy inputs
test_latent_code_parzen = Tensor(np.random.normal(size=(25, args.latent_dim)), dtype=mstype.float32)

# Load parameters from checkpoint into network
file_name = str(args.n_epochs-1) + ".ckpt"
param_dict = load_checkpoint(os.path.join('checkpoints', file_name))
load_param_into_net(generator, param_dict)


# Export network into MINDIR model file
if not os.path.exists('outputs'):
    os.mkdir('outputs')

path = os.path.join('outputs', file_name)
export(generator, test_latent_code_parzen, file_name=path, file_format='MINDIR')
print("==========================================")
print(file_name + ".mindir exported successfully!")
print("==========================================")
