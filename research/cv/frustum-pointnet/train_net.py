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
import argparse
import json
import datetime

from pathlib import Path

import mindspore as ms
from mindspore import nn, context, dtype
from mindspore.ops import functional as F

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn import Metric
from mindspore.communication.management import init, get_rank
from mindspore.nn import rearrange_inputs
from mindspore.train.callback import Callback

import numpy as np
from train import provider, datautil

BASE_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR = BASE_DIR.parent

sys.path.append(str(BASE_DIR))


sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.joinpath('src')))

DEBUG = True

enable_check = False
parser = argparse.ArgumentParser()
parser.add_argument('--name',
                    type=str,
                    default='ckpt',
                    help='tensorboard writer name')
parser.add_argument('--model',
                    default='frustum_pointnets_v1_ms',
                    help='Model name [default: frustum_pointnets_v1_ms]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--device_target', default='Ascend', type=str,
                    help='device_target [Ascned, GPU]')
parser.add_argument('--num_point',
                    type=int,
                    default=1024,
                    help='Point Number [default: 2048]')
parser.add_argument('--max_epoch',
                    type=int,
                    default=200,
                    help='Epoch to run [default: 200]')
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer',
                    default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step',
                    type=int,
                    default=20,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--loss_per_epoch',
                    type=int,
                    default=100,
                    help='times to print loss value per epoch')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--objtype',
                    type=str,
                    default='caronly',
                    help='caronly or carpedcyc')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0,
                    help='Weight Decay of Adam [default: 1e-4]')
parser.add_argument('--no_intensity',
                    action='store_true',
                    help='Only use XYZ for training')
parser.add_argument('--train_sets', type=str, default='train')
parser.add_argument('--val_sets', type=str, default='val')
parser.add_argument(
    '--restore_model_path',
    default=None,
    help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--keep_checkpoint_max',
                    type=int,
                    default=5,
                    help='max checkpoints to save [default: 5]')
parser.add_argument('--disable_datasink_mode',
                    default=False,
                    action="store_false",
                    help='disable datasink mode [default: False]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate

MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
DATASINK = FLAGS.disable_datasink_mode
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4  # point feature channel
NUM_CLASSES = 2  # segmentation has two classes

if FLAGS.objtype == 'carpedcyc':
    n_classes = 3
elif FLAGS.objtype == 'caronly':
    n_classes = 1

sys.path.append("../src")

MODEL_FILE = ROOT_DIR.joinpath('src').joinpath(FLAGS.model + '.py')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = DECAY_STEP
BN_DECAY_CLIP = 0.99

LOG_DIR = Path(FLAGS.log_dir)
if not LOG_DIR.exists():
    LOG_DIR.mkdir(exist_ok=True)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w', encoding="utf8")
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


EDLR = nn.ExponentialDecayLR(BASE_LEARNING_RATE,
                             DECAY_RATE,
                             DECAY_STEP,
                             is_stair=True)


def get_learning_rate(step_per_epoch):
    """ learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE! """
    lr_list = nn.exponential_decay_lr(BASE_LEARNING_RATE,
                                      DECAY_RATE,
                                      FLAGS.max_epoch * step_per_epoch,
                                      1,
                                      DECAY_STEP,
                                      is_stair=True)
    lr_list = np.maximum(lr_list, 0.00001)

    return lr_list


def content_init(device_id, device_num, device_target):
    '''content_init'''
    if device_target in ("Ascend", "GPU"):
        ms.common.set_seed(1234)
    else:
        raise ValueError("Unsupported platform {}".format(device_target))

    if not DEBUG:
        context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)
    if device_num > 1:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()

    context.set_context(device_id=device_id)

def pack(d):
    point_cloud = d["data"].astype(dtype.float32)
    one_hot_vec = d["one_hot_vec"].astype(dtype.float32)
    mask_label = d["label"].astype(dtype.int32)
    center_label = d["center"].astype(dtype.float32)
    heading_class_label = d["hclass"].astype(dtype.int32)
    heading_residual_label = d["hres"].astype(dtype.float32)
    size_class_label = d["sclass"].astype(dtype.int32)
    size_residual_label = d["sres"].astype(dtype.float32)
    return point_cloud, one_hot_vec, mask_label, \
        center_label, heading_class_label, heading_residual_label, \
            size_class_label, size_residual_label


class Fmatrix(Metric):

    def __init__(self):
        super(Fmatrix, self).__init__()
        self.clear()
        self.total_correct = 0
        self.total_seen = 0
        self.loss_sum = 0
        self.total_seen_class = [0 for _ in range(NUM_CLASSES)]
        self.total_correct_class = [0 for _ in range(NUM_CLASSES)]
        self.iou2ds_sum = 0
        self.iou3ds_sum = 0
        self.iou3d_acc = 0
        self.iou3d_correct_cnt = 0
        self.argmax_op = ms.ops.Argmax(axis=2)
        self.batch_num = 0
        self.acc = 0
        self.best_iou3d = 0
        self.best_iou3d_epoch = 0
        self.epoch = 0

    def clear(self):
        """Clears the internal evaluation result."""
        self.total_correct = 0
        self.total_seen = 0
        self.loss_sum = 0
        self.total_seen_class = [0 for _ in range(NUM_CLASSES)]
        self.total_correct_class = [0 for _ in range(NUM_CLASSES)]
        self.iou2ds_sum = 0
        self.iou3ds_sum = 0
        self.iou3d_correct_cnt = 0
        self.argmax_op = ms.ops.Argmax(axis=2)
        self.batch_num = 0
        self.accuracy_point = 0
        self.acc = 0
        self.iou3d_acc = 0

    def update_torch(self, output, label):
        self.batch_num += 1
        (_, batch_label, batch_center, batch_hclass, batch_hres, batch_sclass,
         batch_sres, _) = label

        logits, _, _, _, \
            heading_scores, _, heading_residuals, \
            size_scores, _, size_residuals, box3d_center = output
        mask_logits = logits

        iou2ds, iou3ds = provider.compute_box3d_iou(
            box3d_center.asnumpy(), heading_scores.asnumpy(),
            heading_residuals.asnumpy(), size_scores.asnumpy(),
            size_residuals.asnumpy(), batch_center.asnumpy(),
            batch_hclass.asnumpy(), batch_hres.asnumpy(),
            batch_sclass.asnumpy(), batch_sres.asnumpy())
        self.iou2ds_sum += np.sum(iou2ds)
        self.iou3ds_sum += np.sum(iou3ds)
        self.iou3d_acc += np.sum(iou3ds >= 0.7)

        t1 = self.argmax_op(mask_logits)
        t2 = batch_label.astype(ms.dtype.int32)
        correct = F.equal(t1, t2)
        temp = correct.asnumpy()
        acc = np.sum(temp)
        self.acc += acc

    @rearrange_inputs
    def update(self, output, label):
        self.update_torch(output, label)

    def eval_torch(self):
        self.epoch += 1
        n_sample = self.batch_num * BATCH_SIZE
        accuracy = self.acc / (n_sample * NUM_POINT)
        eval_box_IoU = (self.iou2ds_sum / n_sample, self.iou3ds_sum / n_sample)
        eval_box_estimation_accuracy = self.iou3d_acc / n_sample
        if eval_box_estimation_accuracy >= self.best_iou3d:
            self.best_iou3d = eval_box_estimation_accuracy
            self.best_iou3d_epoch = self.epoch
        ans = {
            'eval_accuracy': accuracy,
            "eval_box_IoU_(ground/3D)": eval_box_IoU,
            "eval_box_estimation_accuracy_(IoU=0.7)":
            eval_box_estimation_accuracy,
            'Best Test acc: %f(Epoch %d)':
            (self.best_iou3d, self.best_iou3d_epoch)
        }
        return ans

    def eval(self):
        return self.eval_torch()


class myCallback(Callback):
    """Callback base class"""

    def __init__(self, eval_model, eval_ds, ckpt_path):
        super(myCallback, self).__init__()
        self.model = eval_model
        self.eval_ds = eval_ds
        self.best_iou3d = -1
        self.ckpt_path = ckpt_path

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        log_string("##################################")
        cb_params = run_context.original_args()
        print(f"now learning rate: {lr_func(cb_params.cur_epoch_num)}")


    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        cb_params = run_context.original_args()
        result = self.model.eval(self.eval_ds, dataset_sink_mode=False)
        ans = json.dumps(result["Fmatrix"])
        if float(result["Fmatrix"]["eval_box_estimation_accuracy_(IoU=0.7)"]) > self.best_iou3d:
            self.best_iou3d = float(result["Fmatrix"]["eval_box_estimation_accuracy_(IoU=0.7)"])

            ms.save_checkpoint(save_obj=cb_params.train_network, \
                ckpt_file_name=os.path.join(self.ckpt_path, \
                "best.ckpt"))
            log_string(f"save checkpoint acc {self.best_iou3d:.2f} > best.ckpt")
        log_string(ans)


def lr_func(epoch, _init=BASE_LEARNING_RATE, step_size=DECAY_STEP, gamma=DECAY_RATE, eta_min=0.00001):
    f = gamma**((epoch)//step_size)
    if _init*f > eta_min:
        return _init*f
    return 0.01#0.001*0.01 = eta_min


def train_ms_v3():
    log_string("init environment ... ")
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))
    content_init(device_id, device_num, FLAGS.device_target)
    if FLAGS.device_target == "Ascend":
        loss_scale_manager = FixedLossScaleManager(1024, drop_overflow_update=False)

    import frustum_pointnets_v1 as MODEL_OLD
    rank_id = get_rank() if device_num > 1 else 0
    log_string(f"device_num:{device_num}, rank_id:{rank_id}")
    log_string(
        f"{datetime.datetime.now().isoformat()}:loading train dataset ...")
    overwritten_data_path = 'kitti/frustum_' + \
        FLAGS.objtype+'_'+FLAGS.train_sets+'.pickle'
    TRAIN_DATASET = provider.FrustumDataset(
        npoints=NUM_POINT,
        split=FLAGS.train_sets,
        rotate_to_center=True,
        random_flip=True,
        random_shift=True,
        one_hot=True,
        overwritten_data_path=overwritten_data_path)
    TEST_DATASET = provider.FrustumDataset(
        npoints=NUM_POINT,
        split=FLAGS.val_sets,
        rotate_to_center=True,
        one_hot=True,
        overwritten_data_path='kitti/frustum_'+FLAGS.objtype+'_'+FLAGS.val_sets+'.pickle')
    train_data_set = datautil.get_train_data(TRAIN_DATASET,
                                             BATCH_SIZE,
                                             device_num=device_num,
                                             rank_id=rank_id)
    batch_step = train_data_set.get_dataset_size()

    test_data_set = datautil.get_test_data(TEST_DATASET,
                                           BATCH_SIZE,
                                           device_num=device_num,
                                           rank_id=rank_id)
    log_string(f"{datetime.datetime.now().isoformat()}:construct net ...")
    # build model
    net: nn.Cell = MODEL_OLD.FrustumPointNetv1(n_classes=n_classes,
                                               n_channel=4)
    lossfn = MODEL_OLD.FrustumPointNetLoss(enable_summery=False)
    _ = MODEL_OLD.FrustumPointNetLoss(return_all=True, enable_summery=False)

    net_eval = MODEL_OLD.FpointWithEval(net, lossfn)
    net_with_criterion: nn.Cell = MODEL_OLD.FpointWithLoss_old(net, lossfn)

    lr = [lr_func(i//batch_step) for i in range(FLAGS.max_epoch*batch_step)]
    optimizer = nn.Adam(net.trainable_params(),
                        learning_rate=lr,
                        weight_decay=FLAGS.weight_decay)
    ckpt_path = f"./{FLAGS.name}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if FLAGS.device_target == "Ascend":
        model = ms.Model(net_with_criterion,
                         loss_fn=None,
                         eval_network=net_eval,
                         optimizer=optimizer,
                         loss_scale_manager=loss_scale_manager,
                         metrics={"Fmatrix": Fmatrix()})
    else:
        model = ms.Model(net_with_criterion,
                         loss_fn=None,
                         eval_network=net_eval,
                         optimizer=optimizer,
                         metrics={"Fmatrix": Fmatrix()})

    cb = []

    cb += [TimeMonitor()]
    cb += [LossMonitor(100)]

    train_epoch = FLAGS.max_epoch

    config_ck = CheckpointConfig(
        save_checkpoint_steps=int(train_data_set.get_dataset_size() * \
            train_epoch // FLAGS.keep_checkpoint_max),
        keep_checkpoint_max=FLAGS.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="fpoint_v1",
                              directory=ckpt_path,
                              config=config_ck)

    if rank_id == 0:
        cb += [ckpt_cb]
        cb += [myCallback(model, test_data_set, ckpt_path)]


    model.train(train_epoch,
                train_data_set,
                dataset_sink_mode=DATASINK,
                callbacks=cb)
    print("train down!")

if __name__ == "__main__":
    log_string(f"开始训练 pid: {os.getpid()}")
    train_ms_v3()
