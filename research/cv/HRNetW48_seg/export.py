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
"""Export checkpoint into mindir or air for inference."""
import argparse
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.seg_hrnet import get_seg_model
from src.config import hrnetw48_config as model_config
from src.dataset.dataset_generator import create_seg_dataset


class InferModel(nn.Cell):
    """Add resize and exp behind HRNet."""
    def __init__(self, num_classes):
        super(InferModel, self).__init__()
        self.model = get_seg_model(model_config, num_classes)
        self.resize = ops.ResizeBilinear((1024, 2048))
        self.exp = ops.Exp()

    def construct(self, x):
        """Model construction."""
        out = self.model(x)
        out = self.resize(out)
        out = self.exp(out)
        return out


def main():
    """Export mindir for 310 inference."""
    parser = argparse.ArgumentParser("HRNet Semantic Segmentation exporting.")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID. ")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file path. ")
    parser.add_argument("--file_name", type=str, help="Output file name. ")
    parser.add_argument("--file_format", type=str, default="MINDIR",
                        choices=["AIR", "MINDIR"], help="Output file format. ")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend", "GPU", "CPU"], help="Device target.")
    parser.add_argument("--dataset", type=str, default="cityscapes")

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)
    image_size, num_classes = create_seg_dataset(args.dataset, is_train=False)
    net = InferModel(num_classes)
    pd = load_checkpoint(args.checkpoint_file)
    params_dict = {}
    for k, v in pd.items():
        params_dict["model." + k] = v
    load_param_into_net(net, params_dict, strict_load=True)
    net.set_train(False)
    height, width = image_size
    input_data = Tensor(np.zeros([1, 3, height, width], dtype=np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)


if __name__ == "__main__":
    main()
