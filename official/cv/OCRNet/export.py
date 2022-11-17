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

"""Export checkpoint into mindir or air for 310 inference."""
import argparse

import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

from src.config import config_hrnetv2_w48 as config
from src.seg_hrnet_ocr import get_seg_model


def main():
    """Export process."""
    parser = argparse.ArgumentParser("OCRNet Semantic Segmentation exporting.")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID. ")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Target device [Ascend, GPU]")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file path. ")
    parser.add_argument("--file_name", type=str, help="Output file name. ")
    parser.add_argument("--file_format", type=str, default="MINDIR",
                        choices=["AIR", "MINDIR"], help="Output file format. ")

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    net = get_seg_model(config)
    params_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(net, params_dict)
    net.set_train(False)
    height, width = config.eval.image_size[0], config.eval.image_size[1]
    input_data = Tensor(np.zeros([1, 3, height, width], dtype=np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)


if __name__ == "__main__":
    main()
