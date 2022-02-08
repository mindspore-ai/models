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

"""train FasterRcnn and get checkpoint files."""

import os
import time

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import SGD
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from src.FasterRcnn.faster_rcnn import Faster_Rcnn
from src.dataset import create_fasterrcnn_dataset
from src.dataset import data_to_mindrecord_byte_image
from src.lr_schedule import dynamic_lr
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.device_adapter import get_device_num
from src.model_utils.device_adapter import get_rank_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.network_define import LossCallBack
from src.network_define import LossNet
from src.network_define import TrainOneStepCell
from src.network_define import WithLossCell

set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

if config.device_target == "GPU":
    context.set_context(enable_graph_kernel=False)
if config.run_distribute:
    if config.device_target == "Ascend":
        rank = get_rank_id()
        device_num = get_device_num()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
    else:
        init("nccl")
        context.reset_auto_parallel_context()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
else:
    rank = 0
    device_num = 1


def train_fasterrcnn_():
    """ train_fasterrcnn_ """
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                if not os.path.exists(config.coco_root):
                    print("Please make sure config:coco_root is valid.")
                    raise ValueError(config.coco_root)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                if not os.path.exists(config.image_dir):
                    print("Please make sure config:image_dir is valid.")
                    raise ValueError(config.image_dir)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_fasterrcnn():
    """ train_fasterrcnn """
    dataset_size, dataset = train_fasterrcnn_()
    net = Faster_Rcnn(net_config=config)
    net = net.set_train()

    load_path = config.pre_trained
    param_dict = load_checkpoint(load_path)
    param_dict['rcnn.cls_scores.weight'] = ms.Parameter(param_dict['rcnn.cls_scores.weight'][:, :2])
    param_dict['rcnn.cls_scores.bias'] = ms.Parameter(param_dict['rcnn.cls_scores.bias'][:2])
    param_dict['rcnn.reg_scores.weight'] = ms.Parameter(param_dict['rcnn.reg_scores.weight'][:, :8])
    param_dict['rcnn.reg_scores.bias'] = ms.Parameter(param_dict['rcnn.reg_scores.bias'][:8])
    # For compatibility with old checkpoints
    keys_for_replace = [key for key in param_dict if 'fpn_ncek' in key]
    for w_name in keys_for_replace:
        weight = param_dict.pop(w_name)
        new_key_name = w_name.replace('fpn_ncek', 'fpn_neck')
        param_dict[new_key_name] = ms.Parameter(weight)

    load_param_into_net(net, param_dict)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)

    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if config.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank, output_path=config.save_checkpoint_path)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    if config.run_eval:
        from src.eval_callback import EvalCallBack
        from src.eval_utils import create_eval_mindrecord, apply_eval
        config.prefix = "FasterRcnn_eval.mindrecord"
        anno_json = os.path.join(config.coco_root, "annotations/instances_val2017.json")
        mindrecord_path = os.path.join(config.coco_root, "FASTERRCNN_MINDRECORD", config.prefix)
        config.instance_set = "annotations/instances_val2017.json"

        if not os.path.exists(mindrecord_path):
            config.mindrecord_file = mindrecord_path
            create_eval_mindrecord(config)
        eval_net = Faster_Rcnn(config)
        eval_cb = EvalCallBack(config, eval_net, apply_eval, dataset_size, mindrecord_path, anno_json,
                               save_checkpoint_path)
        cb += [eval_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)


if __name__ == '__main__':
    train_fasterrcnn()
