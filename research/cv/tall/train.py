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
Use this file for standalone training and distributed training
"""

import argparse
import ast
import os

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
import mindspore.dataset as ds

from mindspore.train.model import Model
from mindspore.train.serialization import save_checkpoint
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn import WithLossCell, TrainOneStepCell
from src.config import CONFIG
from src.dataset import TrainDataset, TestingDataSet
from src.utils import AverageMeter, compute_IoU_recall_top_n_forreg
from src.ctrl import CTRL, CTRL_Loss
import numpy as np

class EvalCallBack(Callback):
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.dataset = TestingDataSet(cfg.test_feature_dir, cfg.test_csv_path, cfg.test_batch_size)
        self.bestR5 = 0

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch >= 1:
            print(f"Start eval... at epoch {cur_epoch}")
            IoU_thresh = [0.1, 0.3, 0.5]
            all_correct_num_5 = [0.0] * 5
            all_retrievd = 0.0

            for movie_name in self.dataset.movie_names:
                batch = cfg.test_batch_size
                movie_clip_featmaps, movie_clip_sentences = self.dataset.load_movie_slidingclip(movie_name, 16)
                sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
                sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
                for k in range(0, len(movie_clip_sentences), batch):
                    sent_vec = [x[1] for x in movie_clip_sentences[k:k + batch]]
                    length_k = len(sent_vec)
                    sent_vec = np.array(sent_vec)
                    if length_k < batch:
                        padding = np.zeros(shape=[batch - length_k, sent_vec.shape[1]], dtype=np.float32)
                        sent_vec = np.concatenate((sent_vec, padding), axis=0)
                    for t in range(0, len(movie_clip_featmaps), batch):
                        featmap = [x[1] for x in movie_clip_featmaps[t:t + batch]]
                        length_t = len(featmap)
                        featmap = np.array(featmap)
                        visual_clip_name = [x[0] for x in movie_clip_featmaps[t:t + batch]]

                        start = np.array([int(x.split("_")[1]) for x in visual_clip_name])
                        end = np.array([int(x.split("_")[2].split("_")[0]) for x in visual_clip_name])
                        if length_t < batch:
                            padding = np.zeros(shape=[batch - length_t, featmap.shape[1]], dtype=np.float32)
                            featmap = np.concatenate((featmap, padding), axis=0)
                        input_feat = np.concatenate((featmap, sent_vec), axis=1)
                        output = self.ctrl.construct(Tensor(input_feat))
                        output_np = output.asnumpy()
                        sentence_image_mat[k:k + length_k, t:t + length_t] = output_np[:length_k, :length_t, 0]

                        reg_end = end + output_np[:length_k, :length_t, 2]
                        reg_start = start + output_np[:length_k, :length_t, 1]

                        sentence_image_reg_mat[k:k + length_k, t:t + length_t, 0] = reg_start
                        sentence_image_reg_mat[k:k + length_k, t:t + length_t, 1] = reg_end

                iclips = [b[0] for b in movie_clip_featmaps]
                sclips = [b[0] for b in movie_clip_sentences]

                # calculate Recall@m, IoU=n
                for k in range(len(IoU_thresh)):
                    IoU = IoU_thresh[k]
                    correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat,
                                                                    sentence_image_reg_mat, sclips, iclips)
                    all_correct_num_5[k] += correct_num_5
                    break
                all_retrievd += len(sclips)
            for k in range(len(IoU_thresh)):
                print("Result IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd))
                break
            if self.bestR5 < all_correct_num_5[0] / all_retrievd:
                self.bestR5 = all_correct_num_5[0] / all_retrievd
                print("save best model...")
                save_checkpoint(self.ctrl, f"{cfg.log_dir}/best.ckpt")


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='Train CTRL')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend'],
                        help='device target, only support Ascend.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend.')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute.')
    parser.add_argument('--train_data_dir', type=str, default=None, help='the directory of train data.')
    parser.add_argument('--check_point_dir', type=str, default=None, help='the directory of train check_point.')
    parser.add_argument('--train_url', default=None, help='Cloudbrain Location of training outputs.\
                        This parameter needs to be set when running on the cloud brain platform.')
    parser.add_argument('--data_url', default=None, help='Cloudbrain Location of data.\
                        This parameter needs to be set when running on the cloud brain platform.')
    parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=False,
                        help='Whether it is running on CloudBrain platform.')
    parser.add_argument('--train_output_path', type=str, default=None)
    return parser.parse_args()


args = get_args()
local_data_url = './cache/data'
local_train_url = './cache/train'
_local_train_url = local_train_url
if args.run_cloudbrain:
    import moxing as mox

    args.train_data_dir = local_data_url
    device_id = int(os.getenv('DEVICE_ID'))
    args.train_output_path = os.path.join(local_train_url, f"logs_{int(device_id)}")
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
else:
    device_id = args.device_id
cfg = CONFIG(data_dir=args.train_data_dir, log_dir=args.train_output_path)
set_seed(cfg.seed)
if __name__ == '__main__':
    print("Set Context...")
    rank_size = int(os.getenv('RANK_SIZE')) if args.run_distribute else 1
    rank_id = int(os.getenv('RANK_ID')) if args.run_distribute else 0
    print(f"device_id:{device_id}, rank_id:{rank_id}")
    print(f"args.device_id:{args.device_id}")
    context.set_context(mode=cfg.mode, device_target=args.device_target,
                        device_id=device_id, save_graphs=False)
    if args.run_distribute:
        print("Init distribute train...")
        cfg.batch_size = 8
        cfg.max_epoch = 10
        cfg.optimizer = 'Momentum'
        init()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)

    print('Done.')
    print("Get Dataset...")

    dataset = TrainDataset(cfg.train_feature_dir, cfg.train_csv_path, cfg.valid_csv_path,
                           cfg.visual_dim, cfg.sentence_embed_dim, cfg.IoU, cfg.nIoL,
                           cfg.context_num, cfg.context_size)
    if args.run_distribute:
        dataset = ds.GeneratorDataset(dataset, ["vis_sent", "offset"], shuffle=False,
                                      num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = ds.GeneratorDataset(dataset, ["vis_sent", "offset"], shuffle=False)
    dataset = dataset.shuffle(buffer_size=cfg.buffer_size)
    dataset = dataset.batch(batch_size=cfg.batch_size)
    print('Done.')

    print("Get Model...")
    net = CTRL(cfg.visual_dim, cfg.sentence_embed_dim, cfg.semantic_dim, cfg.middle_layer_dim)
    loss = CTRL_Loss(cfg.lambda_reg)

    evalCallBack = EvalCallBack(ctrl=net)

    if cfg.optimizer == 'Adam':
        net_opt = nn.Adam(net.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Momentum':
        net_opt = nn.Momentum(net.trainable_params(), cfg.lr, cfg.momentum)
    elif cfg.optimizer == 'SGD':
        net_opt = nn.SGD(net.trainable_params(), learning_rate=cfg.lr)
    else:
        raise ValueError("cfg.optimizer is null")
    net = WithLossCell(net, loss)
    net = TrainOneStepCell(net, net_opt)
    model = Model(net)
    print('Done.')

    print("Train Model...")
    loss_meter = AverageMeter('loss')
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint = ModelCheckpoint(prefix="checkpoint_CTRL", directory=cfg.log_dir, config=config_ck)

    if cfg.mode == context.GRAPH_MODE:
        model.train(cfg.max_epoch, dataset,
                    callbacks=[ckpoint, LossMonitor(), TimeMonitor(), evalCallBack], dataset_sink_mode=True)
    else:
        model.train(cfg.max_epoch, dataset,
                    callbacks=[ckpoint, LossMonitor(), TimeMonitor(), evalCallBack], dataset_sink_mode=False)
    print('Done.')

    print("End.")
    if args.run_cloudbrain:
        mox.file.copy_parallel(src_url=_local_train_url, dst_url=args.train_url)
