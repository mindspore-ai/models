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
##############export checkpoint file into air, mindir models#################
python export.py
"""
import os

import numpy as np
from mindspore import Tensor, export, context
from mindspore import load_checkpoint, load_param_into_net

from src.modelarts_utils.config import config
from src.models.networks import ResnetGenerator
from src.utils.tools import check_folder

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

def run_export():
    """export"""
    genA2B = ResnetGenerator(input_nc=3,
                             output_nc=3,
                             ngf=config.ch,
                             n_blocks=config.n_res,
                             img_size=config.img_size)
    genA2B.set_train(True)
    # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
    # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d
    param_GA2B = load_checkpoint(config.genA2B_ckpt)
    load_param_into_net(genA2B, param_GA2B)

    input_shp = [config.batch_size, 3, config.img_size, config.img_size]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))

    check_folder(config.MINDIR_outdir)
    G_A2B_file = os.path.join(config.MINDIR_outdir, "UGATIT_AtoB")
    export(genA2B, input_array, file_name=G_A2B_file, file_format=config.export_file_format)
    print(f"export into {config.export_file_format} format")

if __name__ == '__main__':
    run_export()
