"""test"""
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
import sys
import time
import datetime
import argparse
import os
import os.path as osp
import numpy as np

import mindspore
from mindspore.common import set_seed
from mindspore import context
from mindspore import Tensor
from mindspore import ops
from mindspore.train.serialization import load_param_into_net
import mindspore.nn as nn

from src.utils import Logger
from src.eval_metrics import evaluate
from src import init_model
from src.dataset_loader import create_test_dataset

parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')

parser.add_argument('--run_modelarts', type=lambda x: x.lower() == 'true', default=False, help='train on the modelarts')
parser.add_argument('--modelarts_ckpt_name', type=str, default='resnet50-300_23.ckpt')

# remote server
parser.add_argument('--data_url', type=str, default='/root/lh/', help="root path to data directory")
parser.add_argument('--checkpoint_path', type=str, default='/root/lh/lhresnet50-300_23.ckpt', metavar='PATH')
parser.add_argument('--dataset', type=str, default='market1501')
parser.add_argument('--workers', default=4, type=int, help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128, help="width of an image (default: 128)")
parser.add_argument('--device_id', type=int, default=0)

parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")

parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--train_url', type=str, default='log')
parser.add_argument('--reranking', type=lambda x: x.lower() == 'true', default=True, help='re_rank')
parser.add_argument('--test_distance', type=str, default='global_local', help='test distance type')
parser.add_argument('--unaligned', action='store_true')

args = parser.parse_args()
set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(device_id=args.device_id)

def test(model, queryload, galleryload):
    """test part 1"""
    model.set_train(False)
    qf, q_pids, q_camids, lqf = [], [], [], []
    for subtarget in queryload.create_dict_iterator():
        imgs = subtarget["img"].asnumpy()
        pids = subtarget["pid"].asnumpy()
        camids = subtarget["camid"].asnumpy()
        t_imgs = Tensor(imgs)
        features, local_features = model(t_imgs)
        squeeze = ops.Squeeze(3)
        local_features = squeeze(local_features)
        qf.append(features)
        lqf.append(local_features)
        q_pids.extend(pids)
        q_camids.extend(camids)

    op_concat = ops.Concat(axis=0)
    qf = op_concat(qf)
    lqf = op_concat(lqf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))

    gf, g_pids, g_camids, lgf = [], [], [], []
    for subtarget in galleryload.create_dict_iterator():
        imgs = subtarget["img"].asnumpy()
        pids = subtarget["pid"].asnumpy()
        camids = subtarget["camid"].asnumpy()
        t_imgs = Tensor(imgs)
        features, local_features = model(t_imgs)
        local_features = squeeze(local_features)

        features = features.asnumpy()
        local_features = local_features.asnumpy()
        gf.append(features)
        lgf.append(local_features)
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = np.concatenate(gf, 0)
    lgf = np.concatenate(lgf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    gf = Tensor(gf)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))
    test_part2(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids)

def test_part2(qf, gf, lqf, lgf, q_pids, g_pids, q_camids, g_camids):
    """test part 2"""
    ranks = [1, 5, 10, 20]
    # feature normlization
    m_norm = nn.Norm(axis=-1, keep_dims=True)
    broadcast_toq = ops.BroadcastTo(qf.shape)
    broadcast_tog = ops.BroadcastTo(gf.shape)

    m_qf = m_norm(qf)
    m_qf = broadcast_toq(m_qf)
    qf = 1.*qf /(m_qf + 1e-12)
    m_gf = m_norm(gf)
    m_gf = broadcast_tog(m_gf)
    gf = 1.*gf /(m_gf + 1e-12)

    m, n = qf.shape[0], gf.shape[0]
    op_pow = ops.Pow()
    op_sum = ops.ReduceSum(keep_dims=True)
    op_broadq = ops.BroadcastTo((m, n))
    op_broadg = ops.BroadcastTo((n, m))

    distqf = op_pow(qf, 2)
    distqf = op_sum(distqf, 1)
    distqf = op_broadq(distqf)
    distgf = op_pow(gf, 2)
    distgf = op_sum(distgf, 1)
    distgf = op_broadg(distgf)

    op_transpose = ops.Transpose()
    transpose_distgf = op_transpose(distgf, (1, 0))
    distmat = distqf + transpose_distgf

    transpose_gf = op_transpose(gf, (1, 0))
    temp = ops.matmul(qf, transpose_gf)
    temp = temp*(-2)
    distmat = distmat+temp  # global_distmat

    if not args.test_distance == 'global':
        from src.distance import low_memory_local_dist
        lqf = lqf.asnumpy()
        lqf = np.transpose(lqf, (0, 2, 1))
        lgf = np.transpose(lgf, (0, 2, 1))
        local_distmat = low_memory_local_dist(lqf, lgf, aligned=not args.unaligned) # local_distmat
        if args.test_distance == 'local':
            distmat = local_distmat
        if args.test_distance == 'global_local':
            print("Using global and local branches")
            distmat = distmat.asnumpy()
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))

    if args.reranking:
        from src.re_ranking import re_ranking
        if args.test_distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf, lqf, aligned=not args.unaligned)
            local_gg_distmat = low_memory_local_dist(lgf, lgf, aligned=not args.unaligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if args.test_distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3, local_distmat=local_dist, only_local=True)
            elif args.test_distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3, local_distmat=local_dist, only_local=False)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    return cmc[0]

if __name__ == '__main__':
    start_time = time.time()
    sys.stdout = Logger(osp.join(args.train_url, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("Initializing dataset {}".format(args.dataset))

    real_path_data = args.data_url
    if args.run_modelarts:
        import moxing as mox

        real_path_data = '/cache/datapath/'
        os.system("rm -rf {0}".format(real_path_data))
        os.system("mkdir {0}".format(real_path_data))
        mox.file.copy_parallel(args.data_url, real_path_data)
        mox.file.copy_parallel(args.checkpoint_path, real_path_data)
        print("training data finish copy to %s." % real_path_data)
        print("test checkpoint finish copy to %s." % real_path_data)

    queryloader, galleryloader, num_train_pids = create_test_dataset(real_path_data, args)
    net = init_model(name=args.arch, num_classes=num_train_pids, loss='softmax and metric', is_train=False)

    checkpoint_file = args.checkpoint_path
    if args.run_modelarts:
        checkpoint_file = real_path_data + args.modelarts_ckpt_name
    if checkpoint_file:
        checkpoint = mindspore.load_checkpoint(checkpoint_file)
        load_param_into_net(net, checkpoint)
    test(net, queryloader, galleryloader)

    train_time = time.time()-start_time
    total_time = str(datetime.timedelta(seconds=train_time))
    print("total_time", total_time)
