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

"""task distill script"""

import datetime
import os
import argparse
import ast

import re
import numpy as np

from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.nn.optim import AdamWeightDecay
from mindspore import set_seed
from mindspore.train.callback import TimeMonitor
from mindspore.train.callback import LossMonitor
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.dataset import create_dataset
from src.utils import StepCallBack, ModelSaveCkpt, EvalCallBack, BertLearningRate
from src.config import train_cfg, eval_cfg, teacher_net_cfg, student_net_cfg, task_cfg, cfg_cfg
from src.cell_wrapper import BertNetworkWithLoss, BertTrainOneStepWithLossScaleCell
from src.tinybert_model import BertModelCLS


WEIGHTS_NAME = cfg_cfg.WEIGHTS_NAME
EVAL_DATA_NAME = cfg_cfg.EVAL_DATA_NAME
TRAIN_DATA_NAME = cfg_cfg.TRAIN_DATA_NAME
DEFAULT_NUM_LABELS = cfg_cfg.DEFAULT_NUM_LABELS
DEFAULT_SEQ_LENGTH = cfg_cfg.DEFAULT_SEQ_LENGTH
DEFAULT_BS = cfg_cfg.DEFAULT_BS

def parse_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(description='ternarybert task distill')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='Device where the code will be implemented. (Default: GPU)')
    parser.add_argument('--do_eval', type=ast.literal_eval, default=True,
                        help='Do eval task during training or not. (Default: True)')
    parser.add_argument('--epoch_size', type=int, default=5, help='Epoch size for train phase. (Default: 3)')
    parser.add_argument('--device_id', type=int, default=0, help='Device id. (Default: 0)')
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument('--do_shuffle', type=ast.literal_eval, default=True,
                        help='Enable shuffle for train dataset. (Default: True)')
    parser.add_argument('--enable_data_sink', type=ast.literal_eval, default=True,
                        help='Enable data sink. (Default: True)')
    parser.add_argument('--save_ckpt_step', type=int, default=50,
                        help='If do_eval is False, the checkpoint will be saved every save_ckpt_step. (Default: 50)')
    parser.add_argument('--max_ckpt_num', type=int, default=50,
                        help='The number of checkpoints will not be larger than max_ckpt_num. (Default: 50)')
    parser.add_argument('--data_sink_steps', type=int, default=50, help='Sink steps for each epoch. (Default: 1)')
    parser.add_argument('--teacher_model_dir', type=str, default='', help='The checkpoint directory of teacher model.')
    parser.add_argument('--student_model_dir', type=str, default='', help='The checkpoint directory of student model.')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory.')
    parser.add_argument('--output_dir', type=str, default='', help='The output checkpoint directory.')
    parser.add_argument('--task_name', type=str, default='sts-b', choices=['sts-b', 'qnli', 'mnli'],
                        help='The name of the task to train. (Default: sts-b)')
    parser.add_argument('--dataset_type', type=str, default='tfrecord', choices=['tfrecord', 'mindrecord'],
                        help='The name of the task to train. (Default: tfrecord)')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Eval Batch size in callback')
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--file_name", type=str, default="ternarybert", help="The name of the output file.")
    parser.add_argument("--file_format", type=str, default="MINDIR", choices=["AIR", "MINDIR"],
                        help="output model type")
    # model art
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False,
                        help='Do modelarts or not. (Default: False)')
    parser.add_argument("--data_url", type=str, default="./dataset", help='real input file path')
    parser.add_argument("--train_url", type=str, default="", help='real output file path include .ckpt and .air') # modelarts -> obs
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset", help='modelart input path')
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result", help='modelart output path.')
    parser.add_argument("--result_dir", type=str, default="./output", help='output')
    parser.add_argument("--modelarts_attrs", type=str, default="")

    return parser.parse_args()

def obs_data2modelarts(args_opt):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    import moxing as mox
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args_opt.data_url, args_opt.modelarts_data_dir))
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=args_opt.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    if not mox.file.exists(args_opt.result_dir):
        mox.file.make_dirs(args_opt.result_dir)

def modelarts_result2obs(args_opt):
    """
    Copy result data from modelarts to obs.
    """
    import moxing as mox
    train_url = args_opt.train_url
    if not mox.file.exists(train_url):
        print(f"train_url[{train_url}] not exist!")
        mox.file.make_dirs(train_url)
    save_ckpt_dir = os.path.join(args_opt.result_dir, args_opt.task_name)
    mox.file.copy_parallel(src_url=save_ckpt_dir, dst_url=os.path.join(train_url, args_opt.task_name))
    files = os.listdir(args_opt.result_dir)
    print("===>>>current Files:", files)
    print("===>>>Copy Event or Checkpoint from modelarts dir: ./ckpt to obs:{}".format(train_url))
    if args_opt.file_format == "MINDIR":
        mox.file.copy(src_url='ternarybert.mindir',
                      dst_url=os.path.join(train_url, 'ternarybert.mindir'))
    else:
        mox.file.copy(src_url='ternarybert.air',
                      dst_url=os.path.join(train_url, 'ternarybert.air'))

def export_MODEL(args_opt):
    """
    start modelarts export
    """
    class Task:
        """
        Encapsulation class of get the task parameter.
        """

        def __init__(self, task_name):
            self.task_name = task_name

        @property
        def num_labels(self):
            if self.task_name in task_cfg and "num_labels" in task_cfg[self.task_name]:
                return task_cfg[self.task_name]["num_labels"]
            return DEFAULT_NUM_LABELS

        @property
        def seq_length(self):
            if self.task_name in task_cfg and "seq_length" in task_cfg[self.task_name]:
                return task_cfg[self.task_name]["seq_length"]
            return DEFAULT_SEQ_LENGTH

    task = Task(args_opt.task_name)
    student_net_cfg.seq_length = task.seq_length
    student_net_cfg.batch_size = DEFAULT_BS
    student_net_cfg.do_quant = False

    ckpt_file = os.path.join(args_opt.result_dir, args_opt.task_name, WEIGHTS_NAME)
    eval_model = BertModelCLS(student_net_cfg, False, task.num_labels, 0.0, phase_type='student')
    param_dict = load_checkpoint(ckpt_file)
    new_param_dict = {}
    for key, value in param_dict.items():
        new_key = re.sub('tinybert_', 'bert_', key)
        new_key = re.sub('^bert.', '', new_key)
        new_param_dict[new_key] = value
    load_param_into_net(eval_model, new_param_dict)
    eval_model.set_train(False)
    input_ids = Tensor(np.zeros((student_net_cfg.batch_size, task.seq_length), np.int32))
    token_type_id = Tensor(np.zeros((student_net_cfg.batch_size, task.seq_length), np.int32))
    input_mask = Tensor(np.zeros((student_net_cfg.batch_size, task.seq_length), np.int32))

    input_data = [input_ids, token_type_id, input_mask]

    export(eval_model, *input_data, file_name=args_opt.file_name, file_format=args_opt.file_format)


def run_task_distill(args_opt):
    """
    run task distill
    """
    if args_opt.enable_modelarts:
        args.student_model_dir = os.path.join(args.modelarts_data_dir, args.student_model_dir) #args.student_model_dir = '/data/weights/student_model/'
        args.teacher_model_dir = os.path.join(args.modelarts_data_dir, args.teacher_model_dir) #args.teacher_model_dir = '/data/weights/teacher_model/'
        args.data_dir = os.path.join(args.modelarts_data_dir, args.data_dir) #args.data_dir = '/data'
        args.output_dir = args.result_dir
    task = task_cfg[args_opt.task_name]
    teacher_net_cfg.seq_length = task.seq_length
    student_net_cfg.seq_length = task.seq_length
    train_cfg.batch_size = args_opt.train_batch_size
    eval_cfg.batch_size = args_opt.eval_batch_size
    teacher_ckpt = os.path.join(args_opt.teacher_model_dir, args_opt.task_name, WEIGHTS_NAME)
    student_ckpt = os.path.join(args_opt.student_model_dir, args_opt.task_name, WEIGHTS_NAME)
    train_data_dir = os.path.join(args_opt.data_dir, args_opt.task_name, TRAIN_DATA_NAME)
    eval_data_dir = os.path.join(args_opt.data_dir, args_opt.task_name, EVAL_DATA_NAME)
    save_ckpt_dir = os.path.join(args_opt.output_dir, args_opt.task_name)
    if args_opt.distribute == "true":
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=device_id)
        D.init()
        device_num = args_opt.device_num
        rank = device_id % device_num
        print("device_id is {}, rank_id is {}".format(device_id, rank))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        save_ckpt_dir = save_ckpt_dir + '_ckpt_' + str(rank)
    else:
        if args_opt.device_target == "Ascend" or args_opt.device_target == "GPU":
            context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                                device_id=args_opt.device_id)
        else:
            raise Exception("Target error, GPU or Ascend is supported.")
        rank = 0
        device_num = 1
    train_dataset = create_dataset(batch_size=train_cfg.batch_size, device_num=device_num, rank=rank,
                                   do_shuffle=args_opt.do_shuffle, data_dir=train_data_dir,
                                   data_type=args_opt.dataset_type, seq_length=task.seq_length,
                                   task_type=task.task_type, drop_remainder=True)
    dataset_size = train_dataset.get_dataset_size()
    print('train dataset size:', dataset_size)
    eval_dataset = create_dataset(batch_size=eval_cfg.batch_size, device_num=1, rank=0,
                                  do_shuffle=args_opt.do_shuffle, data_dir=eval_data_dir,
                                  data_type=args_opt.dataset_type, seq_length=task.seq_length,
                                  task_type=task.task_type, drop_remainder=False)
    print('eval dataset size:', eval_dataset.get_dataset_size())
    repeat_count = args_opt.epoch_size
    time_monitor_steps = dataset_size
    netwithloss = BertNetworkWithLoss(teacher_config=teacher_net_cfg, teacher_ckpt=teacher_ckpt,
                                      student_config=student_net_cfg, student_ckpt=student_ckpt,
                                      is_training=True, task_type=task.task_type, num_labels=task.num_labels)
    params = netwithloss.trainable_params()
    optimizer_cfg = train_cfg.optimizer_cfg
    lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size * args_opt.epoch_size *
                                                    optimizer_cfg.AdamWeightDecay.warmup_ratio),
                                   decay_steps=int(dataset_size * args_opt.epoch_size),
                                   power=optimizer_cfg.AdamWeightDecay.power)
    decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0}, {'order_params': params}]
    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 20, scale_factor=2.0, scale_window=1000)
    netwithgrads = BertTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    callback_size = dataset_size
    if args_opt.do_eval:
        eval_dataset = list(eval_dataset.create_dict_iterator())
        callback = [TimeMonitor(time_monitor_steps), LossMonitor(callback_size),
                    EvalCallBack(network=netwithloss.bert, dataset=eval_dataset,
                                 eval_ckpt_step=dataset_size,
                                 save_ckpt_dir=save_ckpt_dir,
                                 embedding_bits=student_net_cfg.embedding_bits,
                                 weight_bits=student_net_cfg.weight_bits,
                                 clip_value=student_net_cfg.weight_clip_value,
                                 metrics=task.metrics)]
    else:
        callback = [TimeMonitor(time_monitor_steps), StepCallBack(), LossMonitor(callback_size),
                    ModelSaveCkpt(network=netwithloss.bert, save_ckpt_step=args_opt.save_ckpt_step,
                                  max_ckpt_num=args_opt.max_ckpt_num, output_dir=save_ckpt_dir,
                                  embedding_bits=student_net_cfg.embedding_bits,
                                  weight_bits=student_net_cfg.weight_bits,
                                  clip_value=student_net_cfg.weight_clip_value)]
    model = Model(netwithgrads)
    model.train(repeat_count, train_dataset, callbacks=callback,
                dataset_sink_mode=args_opt.enable_data_sink)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    if args.enable_modelarts:
        obs_data2modelarts(args)
    run_task_distill(args)
    print("===========training success================")
    if args.enable_modelarts:
        ## start export air
        export_MODEL(args)
        print("===========export success================")
        ## copy result from modelarts to obs
        modelarts_result2obs(args)
    print("===========Done!!!!!================")
