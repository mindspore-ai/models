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
"""export checkpoint file into air models"""
import argparse
import numpy as np
from mindspore import Tensor, context, load_checkpoint, export, load_param_into_net
from src.dgcn import DGCN
from src.config import ConfigDGCN

parser = argparse.ArgumentParser(description="DGCN export")
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer", "pubmed"], help="Dataset.")
parser.add_argument("--file_name", type=str, default="dgcn", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend"], help="device target (default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == "__main__":
    config = ConfigDGCN()

    if args.dataset == "cora":
        input_dim = 1433
        output_dim = 7
        diffusions = Tensor(np.zeros((2708, 2708), np.float32))
        ppmi = Tensor(np.zeros((2708, 2708), np.float32))
        features = Tensor(np.zeros((2708, 1433), np.float32))
    if args.dataset == "citeseer":
        input_dim = 3703
        output_dim = 6
        diffusions = Tensor(np.zeros((3312, 3312), np.float32))
        ppmi = Tensor(np.zeros((3312, 3312), np.float32))
        features = Tensor(np.zeros((3312, 3703), np.float32))
    if args.dataset == "pubmed":
        input_dim = 3703
        output_dim = 3
        diffusions = Tensor(np.zeros((19717, 19717), np.float32))
        ppmi = Tensor(np.zeros((19717, 19717), np.float32))
        features = Tensor(np.zeros((19717, 500), np.float32))


    dgcn_net = DGCN(input_dim=input_dim, hidden_dim=config.hidden1, output_dim=output_dim, dropout=config.dropout)
    dgcn_net.set_train(False)
    dgcn_net.add_flags_recursive(fp16=True)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(dgcn_net, param_dict)
    export(dgcn_net, diffusions, ppmi, features, file_name=args.file_name, file_format=args.file_format)
    print("==========================================")
    print(args.file_name + ".mindir exported successfully!")
    print("==========================================")
