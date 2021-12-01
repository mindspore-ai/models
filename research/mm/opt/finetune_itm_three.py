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
"""
UNITER pre-training
"""
import os
import time
import argparse
import math

import mindspore
from mindspore import context, ops
from mindspore.train.model import Model
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.nn import TrainOneStepCell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from src.data import data_column, create_dataset, get_batch_data
from src.model_mindspore.pretrain_ms import UniterThreeForPretrainingForRetFinetune
from src.model_mindspore.optim_ms import build_optimizer
from src.tools.logger import LOGGER
from src.tools import parse_with_config, set_random_seed
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
import numpy as np

from pathlib2 import Path

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


class LearningRate(LearningRateSchedule):
    """ LearningRate """

    def __init__(self,
                 start_learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(start_learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(start_learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, start_learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """ construct """
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def guard_val(val):
    """ guard_val """
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


def main(opts):
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))
    profiling_path = f'/cache/{local_rank}-graphs/'
    if not os.path.exists(profiling_path):
        Path(profiling_path).mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    save_graphs_path = os.path.join(profiling_path, "graph")
    if not os.path.exists(save_graphs_path):
        Path(save_graphs_path).mkdir(parents=True, exist_ok=True)
    strategy_ckpt_save_file = save_graphs_path + "strategy" + str(local_rank) + ".ckpt"
    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
    os.system('ulimit -s 102400')
    set_random_seed(opts.seed)

    print('local_rank:{}, device id:{} start to run...'.format(local_rank, device_id), flush=True)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    if opts.use_parallel:
        init()
        LOGGER.info("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=opts.full_batch,
            enable_alltoall=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=True,
            pipeline_stages=1,
            strategy_ckpt_save_file=strategy_ckpt_save_file)
        context.set_context(max_call_depth=3000)
    else:
        device_num = 1
        rank = 0
        opts.rank = rank

    ds = create_dataset(opts, device_num=device_num, rank=rank, column_name=data_column,
                        token_size=opts.train_batch_size, full_batch=opts.full_batch, is_train=False)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)

    opts.epochs = 40
    opts.sink_size = 100

    if opts.sink_size > 0:
        new_epoch = opts.epochs * dataset_size // opts.sink_size
        callback_size = opts.sink_size
    else:
        new_epoch = opts.epochs
        callback_size = dataset_size

    net_with_loss = UniterThreeForPretrainingForRetFinetune(opts.model_config, img_dim=opts.img_dim,
                                                            img_label_dim=IMG_LABEL_DIM,
                                                            audio_dim=opts.audio_dim, audio_label_dim=AUDIO_LABEL_DIM,
                                                            use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                            full_batch=opts.full_batch, use_moe=opts.use_moe)
    net_with_loss.init_output()

    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)
    net_with_grads = TrainOneStepCell(net_with_loss, optimizer)

    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]
    model = Model(net_with_grads)
    print("start_training...")
    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=True, sink_size=callback_size)


def validate_itm_matching(model, val_ds):
    '''
    :param model:
    :param val_ds:
    :return: result
    '''
    topk = ops.TopK()
    LOGGER.info("start running ITM validation...")

    score_vec = np.zeros((1000000,))
    k = 0

    n_ex = 0
    for batch in val_ds.create_dict_iterator():
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
         audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks,
         taskId) = get_batch_data(batch)
        scores = model.predict(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                               audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                               txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                               mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                               txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                               txt_masks, img_token_gts, img_token_masks,
                               taskId)

        bs = batch['input_ids'].shape[0]

        score_vec[n_ex:n_ex + bs] = scores[:, k]
        n_ex += bs

    score_mat = score_vec[:n_ex].reshape((int(math.sqrt(n_ex)), -1))

    print(score_mat)
    print(score_mat.shape)
    max_targets = np.arange(0, int(math.sqrt(n_ex)), dtype=np.int64)
    _, topk_indices = topk(score_mat, 10)
    topk_ind = topk_indices.asnumpy()
    gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
    _, rank = np.nonzero(topk_ind == gt_img_j)
    ir_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
    ir_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
    ir_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))
    print(ir_r1, ir_r5, ir_r10)

    score_mat = score_mat.T
    _, topk_indices = topk(score_mat, 10)
    topk_ind = topk_indices.asnumpy()
    gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
    _, rank = np.nonzero(topk_ind == gt_img_j)
    ir_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
    ir_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
    ir_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))
    print(ir_r1, ir_r5, ir_r10)

    acc1 = (score_mat.max(dim=-1)[1] == max_targets).sum().item()
    acc2 = (score_mat.max(dim=0)[1] == max_targets).sum().item()
    print(acc1, acc2)

    return {}


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default="/home/work/user-job-dir/uniter-three/config/ \
                                pretrain_three_modal_txt_img_audio_config.json",
                        help='JSON config files')

    parser.add_argument("--start_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=120000, type=int,
                        help="The decay step.")
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=1024, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=5000, type=int, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=2, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
