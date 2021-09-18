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
Use this file for model inference and accuracy evaluation
"""

import argparse
import ast
import os

import numpy as np
from mindspore import context, Tensor
from mindspore.common import set_seed

from mindspore.train.model import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.config import CONFIG
from src.dataset import TestingDataSet
from src.utils import compute_IoU_recall_top_n_forreg
from src.ctrl import CTRL, CTRL_Loss


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='Train CTRL')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend'],
                        help='device target, only support Ascend.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend.')
    parser.add_argument('--eval_data_dir', type=str, default=None,
                        help='the directory of train data.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint file path.')
    parser.add_argument('--train_url', default=None, help='Cloudbrain Location of training outputs.\
                            This parameter needs to be set when running on the cloud brain platform.')
    parser.add_argument('--data_url', default=None, help='Cloudbrain Location of data.\
                            This parameter needs to be set when running on the cloud brain platform.')
    parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=False,
                        help='Whether it is running on CloudBrain platform.')
    return parser.parse_args()



args = get_args()
local_data_url = './cache/data'
local_train_url = './cache/train'
if args.run_cloudbrain:
    import moxing as mox
    args.eval_data_dir = local_data_url
    args.train_output_path = local_train_url
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
    mox.file.copy_parallel(src_url=args.checkpoint_path, dst_url=local_data_url)
    args.checkpoint_path = os.path.join(local_data_url, "latest.ckpt")

cfg = CONFIG(data_dir=args.eval_data_dir,
             log_dir=args.checkpoint_path)
set_seed(cfg.seed)

if __name__ == '__main__':
    print("Set Context...")
    context.set_context(mode=cfg.mode, device_target=args.device_target,
                        device_id=args.device_id)
    print('Done.')

    print("Get Dataset...")
    dataset = TestingDataSet(cfg.test_feature_dir, cfg.test_csv_path, cfg.test_batch_size)
    print('Done.')

    print("Get Model...")
    net = CTRL(cfg.visual_dim, cfg.sentence_embed_dim, cfg.semantic_dim, cfg.middle_layer_dim)
    loss = CTRL_Loss(cfg.lambda_reg)
    model = Model(net, loss, metrics=None)

    param_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net, param_dict)

    print(f"loading {args.checkpoint_path}...")
    print('Done.')

    print("Start eval...")
    IoU_thresh = [0.1, 0.3, 0.5]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0

    for movie_name in dataset.movie_names:
        batch = cfg.test_batch_size

        print("Test movie: " + movie_name + "....loading movie data")
        movie_clip_featmaps, movie_clip_sentences = dataset.load_movie_slidingclip(movie_name, 16)
        print("sentences: " + str(len(movie_clip_sentences)))
        print("clips: " + str(len(movie_clip_featmaps)))
        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(0, len(movie_clip_sentences), batch):
            sent_vec = [x[1] for x in movie_clip_sentences[k:k + batch]]
            length_k = len(sent_vec)
            sent_vec = np.array(sent_vec)
            batch = length_k
            for t in range(0, len(movie_clip_featmaps), batch):
                featmap = [x[1] for x in movie_clip_featmaps[t:t + batch]]
                length_t = len(featmap)
                featmap = np.array(featmap)
                visual_clip_name = [x[0] for x in movie_clip_featmaps[t:t + batch]]
                if length_t < batch:
                    padding = np.zeros(shape=[batch - length_t, featmap.shape[1]], dtype=np.float32)
                    featmap = np.concatenate((featmap, padding), axis=0)

                start = np.array([int(x.split("_")[1]) for x in visual_clip_name])
                end = np.array([int(x.split("_")[2].split("_")[0]) for x in visual_clip_name])

                input_feat = np.concatenate((featmap, sent_vec), axis=1)
                output = net.construct(Tensor(input_feat))
                output_np = output.asnumpy()

                sentence_image_mat[k:k + length_k, t:t + length_t] = output_np[:, :length_t, 0]

                reg_end = end + output_np[:, :length_t, 2]
                reg_start = start + output_np[:, :length_t, 1]

                sentence_image_reg_mat[k:k + length_k, t:t + length_t, 0] = reg_start
                sentence_image_reg_mat[k:k + length_k, t:t + length_t, 1] = reg_end

        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat,
                                                             sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat,
                                                            sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat,
                                                            sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) +
                  "; IoU=" + str(IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" +
                  str(IoU) + ", R@1: " + str(correct_num_1 / len(sclips)))
            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
    for k in range(len(IoU_thresh)):
        print("IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) +
              "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) +
              "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
    print('Done.')
    print("End.")
