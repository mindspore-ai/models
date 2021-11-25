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
"""eval dice and Hausdorff distance"""
import os
import argparse
from src.config import vnet_cfg as cfg
from src.dataset import InferImagelist
from src.vnet import VNet
from src.utils import evaluation
from mindspore import context, load_param_into_net, load_checkpoint


parser = argparse.ArgumentParser(description='Vnet eval running')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--dev_id', type=int, default=0, help='choose device id')
parser.add_argument("--ckpt_path", type=str, default=None, help="Path of pretrained module, default is None")
parser.add_argument("--data_path", type=str, default="./promise", help="Path of dataset, default is ./promise")
parser.add_argument("--eval_split_file_path", type=str, default="./split/eval.csv",
                    help="Path of dataset, default is ./split/eval.csv")

def main():
    """Main entrance for eval"""
    args = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.dev_id)
    dataInferlist = InferImagelist(cfg, args.data_path, args.eval_split_file_path)
    dataManagerInfer = dataInferlist.dataManagerInfer
    model = VNet(dropout=False)
    model.set_train(False)
    assert args.ckpt_path is not None, 'No ckpt file!'
    print("=> loading checkpoint '{}'".format(args.ckpt_path))
    params_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(model, params_dict)
    for i in range(dataInferlist.__len__()):
        img, img_id = dataInferlist.__getitem__(i)
        output = model(img)
        output = output.asnumpy()
        output = output[0, ...]
        print("save predicted label for test '{}'".format(img_id))
        dataManagerInfer.writeResultsFromNumpyLabel(output, img_id, '_test', '.mhd')
    evaluation(os.path.join(args.data_path, 'gt'), cfg['dirPredictionImage'])

if __name__ == '__main__':
    main()
