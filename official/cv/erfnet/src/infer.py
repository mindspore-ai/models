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
import os
from argparse import ArgumentParser
import cv2
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from dataset import getInferDataLoader_fromfile
from show import Colorize_cityscapes
from util import seed_seed
from model import ERFNet

def infer(network_1, eval_dataloader, ckptPath, output_path_1):
    colorize = Colorize_cityscapes()

    # load model checkpoint
    if ckptPath is None:
        print("no model checkpoint!")
    elif not os.path.exists(ckptPath):
        print("not exist {}".format(ckptPath))
    else:
        print("load model checkpoint {}!".format(ckptPath))
        param_dict = load_checkpoint(ckptPath)
        load_param_into_net(network_1, param_dict)

    for index, images in enumerate(eval_dataloader):
        images = images[0]
        preds = network_1(images)
        preds = np.argmax(preds.asnumpy(), axis=1).astype(np.uint8)
        for i, pred in enumerate(preds):
            colorized_pred = colorize(pred)
            cv2.imwrite(os.path.join(output_path_1, str(index)+"_"+str(i)+".jpg"), \
                colorized_pred)

# example:
# python infer.py \
#   --data_path /path/to/cityscapes \
#   --model_path /path/to/ERFNet.ckpt \
#   --output_path /path/to/output \
#   --device_id 0 > log_infer.txt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device_target', default="Ascend", type=str)
    parser.add_argument('--device_id', type=int)

    config = parser.parse_args()
    model_path = config.model_path
    device_target = config.device_target
    device_id = config.device_id
    data_path = config.data_path
    output_path = config.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    seed_seed()
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(device_target=device_target)
    context.set_context(device_id=device_id)
    context.set_context(save_graphs=False)
    dataloader = getInferDataLoader_fromfile(data_path, 32, 512)

    weight_init = "XavierUniform"
    network = ERFNet(1, 20, weight_init, run_distribute=False, train=False)
    network.set_train(False)

    infer(network, dataloader, model_path, output_path)
