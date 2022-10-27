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

import os
import sys
import time
from pprint import pprint
from mindspore import Tensor, ParameterTuple
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.train.callback import Callback
from mindspore.context import ParallelMode
from mindspore.nn import SGD
from mindspore.common import set_seed

from mindspore.ops import composite as C
from mindspore.train.callback import SummaryCollector
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype

from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr
from src.PVANet.pva_faster_rcnn import PVANet
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

time_stamp_init = False
time_stamp_first = 0


class LossCallBack(Callback):
    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.loss_sum = 0
        self.rank_id = rank_id

        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = time.time()
            time_stamp_init = True

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        self.count += 1
        self.loss_sum += float(loss)

        if self.count >= 1:
            global time_stamp_first
            time_stamp_current = time.time()
            total_loss = self.loss_sum / self.count

            loss_file = open("./loss_{}.log".format(self.rank_id), "a+")
            loss_file.write("%lu s | epoch: %s step: %s total_loss: %.5f" %
                            (time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             total_loss))
            loss_file.write("\n")
            loss_file.close()

            self.count = 0
            self.loss_sum = 0


class LossNet(nn.Cell):
    def construct(self, x1, x2, x3, x4, x5, x6):
        return x1 + x2


class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        loss1, loss2, loss3, loss4, loss5, loss6 = self._backbone(x, img_shape, gt_bboxe, gt_label, gt_num)
        return self._loss_fn(loss1, loss2, loss3, loss4, loss5, loss6)

    @property
    def backbone_network(self):
        return self._backbone


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ms.ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = Tensor([sens,], mstype.float32)
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        weights = self.weights
        loss = self.network(x, img_shape, gt_bboxe, gt_label, gt_num)
        grads = self.grad(self.network, weights)(x, img_shape, gt_bboxe, gt_label, gt_num, self.sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        grads = C.clip_by_global_norm(grads, clip_norm=1.0)
        return ms.ops.depend(loss, self.optimizer(grads))


def prepare_datasets():
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "PVANet.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if _rank == 0 and not os.path.exists(mindrecord_file):
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
                                        device_num=device_num, rank_id=_rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    return dataset_size, dataset


def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_pva_fasterrcnn(rank):
    """ train_pva_fasterrcnn """
    print(f"\n[{rank}] - rank id of process")
    dataset_size, dataset = prepare_datasets()

    print(f"\n[{rank}]", "===> Creating network...")
    net = PVANet(config=config)
    net = net.set_train()
    print(f"[{rank}]", "\t Done!\n")

    load_path = config.pre_trained
    if load_path != "":
        print(f"\n[{rank}]", "===> Loading from checkpoint:", load_path)
        param_dict = ms.load_checkpoint(load_path)

        # key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
        #                'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
        #                'down_sample_layer.0.weight': 'conv_down_sample.weight',
        #                'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
        #                'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
        #                }
        for oldkey in list(param_dict.keys()):
            if oldkey.startswith(('rcnn.cls', 'rcnn.reg', 'accum', 'stat', 'end_point', 'global_step',
                                  'learning_rate', 'moments', 'momentum')):
                param_dict.pop(oldkey)
                continue
            if not oldkey.startswith(('backbone')):
                data = param_dict.pop(oldkey)
                newkey = 'backbone.' + oldkey
                param_dict[newkey] = data
                # oldkey = newkey
            # for k, v in key_mapping.items():
            #     if k in oldkey:
            #         newkey = oldkey.replace(k, v)
            #         param_dict[newkey] = param_dict.pop(oldkey)
            #         break
        # for item in list(param_dict.keys()):
        #     if not item.startswith('backbone'):
        #         param_dict.pop(item)

        # for key, value in param_dict.items():
        #     tensor = value.asnumpy().astype(np.float32)
        #     param_dict[key] = Parameter(tensor, key)
        ms.load_param_into_net(net, param_dict)
    print(f"[{rank}]", "\tDone!\n")

    device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"
    print(f"\n[{rank}]", "===> Device type:", device_type, "\n")
    if device_type == "Ascend":
        net.to_float(ms.float16)
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
                cell.to_float(ms.float32)

    print(f"\n[{rank}]", "===> Creating loss, lr and opt objects...")
    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset_size), ms.float32)

    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    print(f"[{rank}]", "\tDone!\n")
    if config.run_distribute:
        print(f"\n[{rank}]", "===> Run distributed training...\n")
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        print(f"\n[{rank}]", "===> Run single GPU training...\n")
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    print(f"\n[{rank}]", "===> Creating callbacks...")
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(per_print_times=100, rank_id=rank)
    if config.Log_summary:
        summary_collector = SummaryCollector(summary_dir)
        cb = [time_cb, loss_cb, summary_collector]
    else:
        cb = [time_cb, loss_cb]
    print(f"[{rank}]", "\tDone!\n")

    print(f"\n[{rank}]", "===> Configurating checkpoint saving...")
    if rank == 0 and config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='pvanet', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]
    print(f"[{rank}]", "\tDone!\n")

    if config.run_eval:
        from src.eval_callback import EvalCallBack
        from src.eval_utils import create_eval_mindrecord, apply_eval
        config.prefix = "PVANet_eval.mindrecord"
        anno_json = config.anno_path
        mindrecord_path = os.path.join(config.coco_root, "mindrecords", config.prefix)
        config.instance_set = "annotations/instances_val2017.json"

        if not os.path.exists(mindrecord_path):
            config.mindrecord_file = mindrecord_path
            create_eval_mindrecord(config)
        eval_net = PVANet(config)
        eval_cb = EvalCallBack(config, eval_net, apply_eval, dataset_size, mindrecord_path, anno_json,
                               save_checkpoint_path)
        cb += [eval_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb)


if __name__ == '__main__':
    set_seed(1024)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id(),
                   save_graphs=True, save_graphs_path='./graph_path')

    local_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    summary_dir = local_path + "/../train/summary/"

    if config.device_target == "GPU":
        ms.set_context(enable_graph_kernel=True)
    if config.run_distribute:
        init()
        _rank = get_rank()
        device_num = get_group_size()
        ms.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        summary_dir += "thread_num_" + str(_rank) + "/"
    else:
        _rank = 0
        device_num = 1

    print()  # just for readability
    pprint(config)
    print(f"\n[{_rank}] Please check the above information for the configurations\n\n", flush=True)

    train_pva_fasterrcnn(_rank)
