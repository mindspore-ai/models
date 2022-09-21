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
"""Train"""

import os
import argparse

from src.models import FaceNetModelwithLoss
from src.config import facenet_cfg
# from src.data_loader import get_dataloader
from src.data_loader import get_dataloader
from src.eval_metrics import evaluate
# from src.eval_callback import EvalCallBack
# from src.LFWDataset import get_lfw_dataloader

import numpy as np

import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
# from mindspore import  load_checkpoint, load_param_into_net
from mindspore import context




parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument("--data_url", type=str, default='/home/facenet_dataset/dataset/')
parser.add_argument("--train_url", type=str, default="./")
parser.add_argument("--pretrain_ckpt_path", type=str, default="")
parser.add_argument("--is_distributed", type=str, default='False')
parser.add_argument("--run_online", type=str, default='False')
parser.add_argument("--device_id", type=int, default=0)

args = parser.parse_args()



class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def validate_lfw(model, lfw_dataloader):
    distances, labels = [], []

    # print("Validating on LFW! ...")
    for data in lfw_dataloader.create_dict_iterator():
        distance = model.evaluate(data['img1'], data['img2'])
        label = data['issame']
        distances.append(distance)
        labels.append(label)

    labels = np.array([sublabel.asnumpy() for label in labels for sublabel in label])
    distances = np.array([subdist.asnumpy() for distance in distances for subdist in distance])

    _, _, accuracy, _, _, _ = evaluate(distances=distances, labels=labels)
    # Print statistics and add to log
    print("Accuracy on LFW: {:.4f}+-{:.4f}\n".format(np.mean(accuracy), np.std(accuracy)))

    return accuracy

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def main():
    cfg = facenet_cfg

    run_online = args.run_online
    rank = 0
    group_size = 1

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=False)
    # device_id = int(os.getenv('DEVICE_ID'))
    # device_num = int(os.environ.get("RANK_SIZE", 1))

    if args.is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank = get_rank()
        group_size = get_group_size()
        context.set_context(device_id=args.device_id)
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)
        context.set_auto_parallel_context(parameter_broadcast=True)

    else:
        if cfg.device_target == "Ascend":
            device_id = get_device_id()
            context.set_context(device_id=device_id)

        elif cfg.device_target == "GPU":
            context.set_context(enable_graph_kernel=True)

    if run_online == 'True':
        import moxing as mox
        local_data_url = '/cache/data/'
        mox.file.copy_parallel("obs://nanhang/shenao/data/dataset_eval/", local_data_url)
        #local_triplets = local_data_url+"/vggface2.csv"
        local_train_url = "/cache/train_out/"
    else:
        local_data_url = args.data_url
        local_train_url = args.train_url
        #local_triplets = local_data_url+"/vggface2.csv"
        local_csv = local_data_url+"/triplets.csv"

    train_root_dir = local_data_url
    valid_root_dir = local_data_url
    train_csv = local_csv
    valid_csv = local_csv

    ckpt_path = local_train_url

    data_loaders, _ = get_dataloader(train_root_dir, valid_root_dir, train_csv, valid_csv,
                                     cfg.batch_size, cfg.num_workers, group_size, rank,
                                     shuffle=True, mode="train")
    data_loader = data_loaders['train']

    net = FaceNetModelwithLoss(num_classes=1001, margin=cfg.margin, mode='train', ckpt_path=args.pretrain_ckpt_path)

    optimizer = nn.Adam(net.trainable_params(), learning_rate=cfg.learning_rate)

    loss_cb = LossMonitor(per_print_times=cfg.per_print_times)
    time_cb = TimeMonitor(data_size=cfg.per_print_times)

    # checkpoint save
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.per_print_times, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(f"facenet-rank{rank}", ckpt_path + 'rank_' + str(rank), config_ck)

    callbacks = [loss_cb, time_cb, ckpoint_cb]

    model = Model(net, optimizer=optimizer)

    print("============== Starting Training ==============")
    model.train(cfg.num_epochs, data_loader, callbacks=callbacks, dataset_sink_mode=True)

if __name__ == '__main__':
    main()
