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
"""ntsnet eval."""
import argparse
import ast
import os

import mindspore.common.dtype as mstype
from mindspore import context, set_seed, Tensor, load_checkpoint, load_param_into_net, ops

from src.configgpu import config
from src.dataset_gpu import create_dataset_test
from src.network import NTS_NET

parser = argparse.ArgumentParser(description='ntsnet eval running')
parser.add_argument("--run_modelart", type=ast.literal_eval, default=False, help="Run on modelArt, default is false.")
parser.add_argument('--data_url', default="./data", help='Directory contains CUB_200_2011 dataset.')
parser.add_argument('--train_url', default="./ckpt_0", help='Directory contains checkpoint file and eval.log')
parser.add_argument('--ckpt_filename', default=None, help='checkpoint file name')
parser.add_argument("--device_target", type=str, default="Ascend", help="Device Target, default Ascend",
                    choices=["Ascend", "GPU"])
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
args = parser.parse_args()
run_modelart = args.run_modelart
if not run_modelart:
    device_id = args.device_id
else:
    device_id = int(os.getenv('DEVICE_ID'))

batch_size = config.batch_size

resnet50Path = ""

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
context.set_context(device_id=device_id)

if args.ckpt_filename is None:
    files = os.listdir(args.train_url)
    time = 0.
    for file in files:
        if file.endswith('.ckpt'):
            time_temp = os.path.getatime(os.path.join(args.train_url, file))
            if time_temp > time:
                time = time_temp
                args.ckpt_filename = file
    print(f"Get ckpt file {args.ckpt_filename}")

if run_modelart:
    import moxing as mox

    local_input_url = '/cache/data' + str(device_id)
    local_output_url = '/cache/ckpt' + str(device_id)
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_input_url)
    mox.file.copy_parallel(src_url=os.path.join(args.train_url, args.ckpt_filename),
                           dst_url=os.path.join(local_output_url, args.ckpt_filename))
    mox.file.copy_parallel(src_url=os.path.join(args.train_url, "eval.log"),
                           dst_url=os.path.join(local_output_url, "eval.log"))
else:
    local_input_url = args.data_url
    local_output_url = args.train_url


def print2file(obj1, obj2):
    with open(os.path.join(local_output_url, 'eval.log'), 'a') as f:
        f.write(str(obj1))
        f.write(' ')
        f.write(str(obj2))
        f.write(' \r\n')


if __name__ == '__main__':
    set_seed(1)
    test_data_set = create_dataset_test(test_path=os.path.join(local_input_url, "CUB_200_2011/test"),
                                        batch_size=batch_size)
    test_data_loader = test_data_set.create_dict_iterator(output_numpy=True)

    ntsnet = NTS_NET(topK=6, resnet50Path=resnet50Path)
    param_dict = load_checkpoint(os.path.join(local_output_url, args.ckpt_filename))
    load_param_into_net(ntsnet, param_dict)
    ntsnet.set_train(False)
    success_num = 0.0
    total_num = 0.0
    for _, data in enumerate(test_data_loader):
        image_data = Tensor(data['image'], mstype.float32)
        label = Tensor(data["label"], mstype.int32)
        _, scrutinizer_out, _, _ = ntsnet(image_data)
        result_label, _ = ops.ArgMaxWithValue(1)(scrutinizer_out)
        success_num = success_num + sum((result_label == label).asnumpy())
        total_num = total_num + float(image_data.shape[0])
    print2file("ckpt file name: ", args.ckpt_filename)
    print2file("accuracy: ", round(success_num / total_num, 3))
    print("accuracy: ", round(success_num / total_num, 3))
    print("ckpt file name: ", args.ckpt_filename)
    if run_modelart:
        mox.file.copy_parallel(src_url=os.path.join(local_output_url, "eval.log"),
                               dst_url=os.path.join(args.train_url, "eval.log"))
