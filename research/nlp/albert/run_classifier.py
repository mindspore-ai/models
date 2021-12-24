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

'''
Albert finetune and evaluation script.
'''

import os
from src.Albert_Callback_class import albert_callback
from src.albert_for_finetune import AlbertFinetuneCell, AlbertCLS
from src.dataset import create_classification_dataset
from src.assessment_method import Accuracy, F1, MCC, Spearman_Correlation
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, AlbertLearningRate

from src.model_utils.config import config as args_opt, optimizer_cfg, albert_net_cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num

from mindspore import context
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.common import set_seed

_cur_dir = os.getcwd()


def do_train(dataset=None, ds_eval=None, network=None, load_checkpoint_path="",
             save_checkpoint_path="", epoch_num=1, args=None):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    print("========steps_per_epoch: ========", steps_per_epoch)
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = AlbertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                         end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                         warmup_steps=optimizer_cfg.AdamWeightDecay.warmup_steps,
                                         decay_steps=steps_per_epoch * epoch_num,
                                         power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = AlbertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                         end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                         warmup_steps=optimizer_cfg.Lamb.warmup_steps,
                                         decay_steps=steps_per_epoch * epoch_num,
                                         power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="classifier",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)

    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 32, scale_factor=2, scale_window=1000)
    netwithgrads = AlbertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)

    eval_callback = albert_callback(netwithgrads, args, steps_per_epoch, ds_eval, save_checkpoint_path)
    model.train(epoch_num, dataset, callbacks=[TimeMonitor(dataset.get_dataset_size()), eval_callback,
                                               LossCallBack(dataset.get_dataset_size()), ckpoint_cb])


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    f1 = 0.0
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
        f1 = round(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN), 6)
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")
    return f1


def do_eval(dataset=None, network=None, num_class=2, assessment_method="accuracy", load_checkpoint_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(albert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(False, num_class)
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = Spearman_Correlation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")


def modelarts_pre_process():
    '''modelarts pre process function.'''
    args_opt.device_id = get_device_id()
    args_opt.device_num = get_device_num()
    args_opt.load_pretrain_checkpoint_path = os.path.join(args_opt.load_path, args_opt.load_pretrain_checkpoint_path)
    args_opt.load_finetune_checkpoint_path = os.path.join(args_opt.output_path, args_opt.load_finetune_checkpoint_path)
    args_opt.save_finetune_checkpoint_path = os.path.join(args_opt.output_path, args_opt.save_finetune_checkpoint_path)
    if args_opt.schema_file_path:
        args_opt.schema_file_path = os.path.join(args_opt.data_path, args_opt.schema_file_path)
    args_opt.train_data_file_path = os.path.join(args_opt.data_path, args_opt.train_data_file_path)
    args_opt.eval_data_file_path = os.path.join(args_opt.data_path, args_opt.eval_data_file_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_classifier():
    """run classifier task"""
    set_seed(45556)
    epoch_num = args_opt.epoch_num
    assessment_method = args_opt.assessment_method.lower()
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    if args_opt.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    context.set_context(reserve_class_name_in_scope=False)

    if args_opt.device_target == "GPU":
        # Enable graph kernel
        context.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")

    if args_opt.distribute == 'true':
        if args_opt.device_target == "Ascend":
            # rank = args_opt.rank_id
            device_num = args_opt.device_num
            print("device_num: ", device_num)
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
        elif args_opt.device_target == "GPU":
            init("nccl")
            context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                              gradients_mean=True)
        else:
            raise ValueError(args_opt.device_target)
        save_ckpt_path = os.path.join(args_opt.save_finetune_checkpoint_path, 'ckpt_' + str(get_rank()) + '/')
    else:
        rank = 0
        device_num = 1
        save_ckpt_path = os.path.join(args_opt.save_finetune_checkpoint_path, 'ckpt_0/')

    make_directory(save_ckpt_path)

    netwithloss = AlbertCLS(albert_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1,
                            assessment_method=assessment_method)

    if args_opt.do_train.lower() == "true":
        print("create_classification_dataset")
        ds = create_classification_dataset(batch_size=args_opt.train_batch_size, repeat_count=1,
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.train_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                           rank_size=args_opt.device_num,
                                           rank_id=rank)
        ds_eval = create_classification_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                                assessment_method=assessment_method,
                                                data_file_path=args_opt.eval_data_file_path,
                                                schema_file_path=args_opt.schema_file_path,
                                                do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"),
                                                rank_size=1,
                                                rank_id=0)

        do_train(ds, ds_eval, netwithloss, load_pretrain_checkpoint_path, save_ckpt_path, epoch_num, args_opt)

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_ckpt_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, "classifier")

    if args_opt.do_eval.lower() == "true":
        ds = create_classification_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.eval_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        do_eval(ds, AlbertCLS, args_opt.num_class, assessment_method, load_finetune_checkpoint_path)


if __name__ == "__main__":
    run_classifier()
