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
pretrain_eval
"""
import os
import argparse
import mindspore.dataset as ds
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
import numpy as np
import src.util as util
from src.model.meta_eval import MetaEval
from src.model.classifier import Classifier
from src.data.IterSamplers import CategoriesSampler
from src.data.mini_Imagenet import MiniImageNet
from tqdm import tqdm


class Eval:
    """
    Eval meta-baseline and EGNN
    """

    def __init__(self):
        pass

    def set_exp_name(self):
        """
        :return: experience setting
        """
        exp_name = 'D-{}'.format(args.dataset)
        exp_name += '_backbone-{}'.format(args.backbone)
        exp_name += '_N-{}_K-{}'.format(args.num_ways, args.num_shots)
        exp_name += '_L-{}_B-{}'.format(args.num_layers, args.meta_batch_size)
        return exp_name

    def pretrain_eval(self):
        """
        :return: meta-baseline eval
        """
        param_dict = load_checkpoint(args.load_encoder)
        net = Classifier(64)

        load_param_into_net(net, param_dict)
        n_way = 5
        n_query = 15
        n_shots = [args.num_shots]
        eval_net = MetaEval()
        root_path = os.path.join(args.root_path, args.dataset)
        testset = MiniImageNet(root_path, 'test')

        fs_loaders = []
        for n_shot in n_shots:
            test_sampler = CategoriesSampler(testset.data, testset.label, n_way, n_shot + n_query,
                                             200,
                                             args.ep_per_batch)
            test_loader = ds.GeneratorDataset(test_sampler, ['data'], shuffle=True)
            fs_loaders.append(test_loader)

        aves_keys = ['tl', 'ta', 'vl', 'va']
        for n_shot in n_shots:
            aves_keys += ['fsa-' + str(n_shot)]
        aves = {k: util.Averager() for k in aves_keys}

        print("few-shot eval start")
        net.set_train(mode=False)
        for i, n_shot in enumerate(n_shots):
            np.random.seed(0)
            for data in tqdm(fs_loaders[i].create_dict_iterator(), desc='test', leave=False):
                x_shot, x_query = data['data'][:, :, :n_shot], data['data'][:, :, n_shot:]
                img_shape = x_query.shape[-3:]
                x_query = x_query.view(args.ep_per_batch, -1,
                                       *img_shape)  # bs*(way*n_query)*3*84*84
                label = util.make_nk_label(n_way, n_query, args.ep_per_batch)  # bs*(way*n_query)
                acc_val, _ = eval_net.eval(x_shot, x_query, label, net.encoder)
                aves['fsa-' + str(n_shot)].add(acc_val.asnumpy())

        for k, v in aves.items():
            aves[k] = v.item()
        for n_shot in n_shots:
            key = 'fsa-' + str(n_shot)
            print("epoch {}, {}-shot, val acc {:.4f}".format(str(1), n_shot, aves[key]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='configs/train_classifier_mini.yaml')  root_path
    parser.add_argument('--name', default=None)
    parser.add_argument('--root_path', default='./dataset/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', default='mini-imagenet')
    parser.add_argument('--backbone', default='convnet', choices=['convnet', 'resnet12'])
    parser.add_argument('--load_encoder',
                        default='./save/epoch-max.ckpt')
    parser.add_argument('--resume', type=str, default="False")
    parser.add_argument('--ep_per_batch', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--visualize_datasets', default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--classifier', default='linear-classifier')
    parser.add_argument('--n_classes', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--meta_batch_size', type=int, default=40)
    parser.add_argument('--save_epoch', type=int, default=200)
    parser.add_argument('--eval_fs_epoch', type=int, default=3)

    parser.add_argument('--num_ways', type=int, default=5)
    parser.add_argument('--num_shots', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    parser.add_argument('--run_offline', type=str, default="False", help='run in offline')

    args = parser.parse_args()

    eval_model = Eval()
    eval_model.set_exp_name()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    if args.device_target == 'GPU' or args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)
        if args.run_offline == "True":
            import moxing as mox
            mox.file.copy_parallel(src_url=args.data_url, dst_url=args.root_path)
    else:
        raise ValueError("Unsupported platform.")

    eval_model.pretrain_eval()
