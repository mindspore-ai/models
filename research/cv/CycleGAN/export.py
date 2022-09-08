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

"""export file."""

import numpy as np
import mindspore as ms
from src.models.cycle_gan import get_generator
from src.utils.args import get_args
from src.utils.tools import load_ckpt, enable_batch_statistics


if __name__ == '__main__':
    args = get_args("export")
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.platform)
    G_A = get_generator(args)
    G_B = get_generator(args)
    # Use BatchNorm2d with batchsize=1, affine=False, use_batch_statistics=True instead of InstanceNorm2d
    # Use real mean and variance rather than moving_mean and moving_varance in BatchNorm2d
    enable_batch_statistics(G_A)
    enable_batch_statistics(G_B)
    load_ckpt(args, G_A, G_B)

    input_shp = [args.export_batch_size, 3, args.image_size, args.image_size]
    input_array = ms.Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    G_A_file = f"{args.export_file_name}_AtoB"
    ms.export(G_A, input_array, file_name=G_A_file, file_format=args.export_file_format)
    G_B_file = f"{args.export_file_name}_BtoA"
    ms.export(G_B, input_array, file_name=G_B_file, file_format=args.export_file_format)
