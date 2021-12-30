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
train_meta
"""
import argparse
import os
import numpy as np
from mindspore import nn, ops, ParameterTuple, load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.dataset import context, ds
from tqdm import tqdm
import src.util as util
from src.data.IterSamplers import CategoriesSampler
from src.data.mini_Imagenet import MiniImageNet
from src.model.classifier import Classifier
from src.model.meta_baseline import MetaBaseline, MetaBaselineWithLossCell


class TrainOneStepCell(nn.Cell):
    """
    TrainOneStepCell
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=True)
        self.network = network
        self.optimizer = optimizer
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        """
        :param value:
        :return:
        """
        self.sens = value

    def construct(self, x_shot, x_query, label):
        """
        :param x_shot:
        :param x_query:
        :param label:
        :return:
        """
        weights = self.weights
        loss = self.network(x_shot, x_query, label)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(x_shot, x_query, label, sens)
        return ops.depend(loss, self.optimizer(grads))


def main():
    """
    :return:
    """
    svname = args.name
    if svname is None:  # svname = classifier_mini-imagenet_resnet12
        svname = 'train-meta_{}'.format(args.dataset)
        svname += '_ways_' + str(args.num_ways)
        svname += '_shots_' + str(args.num_shots)
        svname += '_' + args.encoder
    # loader dataset
    save_path = os.path.join('./save/', svname)
    util.ensure_path(save_path)
    root_path = os.path.join(args.root_path, args.dataset)
    if args.dataset == 'mini-imagenet':
        trainset = MiniImageNet(root_path, 'train')
        testset = MiniImageNet(root_path, 'test')

    n_query = 15

    train_sampler = CategoriesSampler(trainset.data, trainset.label, args.num_ways,
                                      args.num_shots + n_query, args.ep_batch, args.ep_per_batch)
    train_loader = ds.GeneratorDataset(train_sampler, ['data'], shuffle=True)

    test_sampler = CategoriesSampler(testset.data, testset.label, args.num_ways,
                                     args.num_shots + n_query, args.ep_batch, args.ep_per_batch)
    test_loader = ds.GeneratorDataset(test_sampler, ['data'], shuffle=True)

    classifier = Classifier(args.n_classes)
    param_dict = load_checkpoint(args.load_encoder)
    load_param_into_net(classifier, param_dict)

    net = MetaBaseline()
    load_param_into_net(net.encoder, classifier.encoder.parameters_dict())
    net_with_loss = MetaBaselineWithLossCell(net)
    net_opt = nn.SGD(params=net.trainable_params(), learning_rate=args.lr,
                     weight_decay=args.weight_decay, momentum=0.9)
    train_cell = TrainOneStepCell(net_with_loss, net_opt)
    # eval model
    max_va = 0.
    timer_used = util.Timer()
    timer_epoch = util.Timer()
    trlog = dict()
    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    for k in aves_keys:
        trlog[k] = []
    for epoch in range(1, args.max_epoch):
        timer_epoch.s()
        aves = {k: util.Averager() for k in aves_keys}
        net.set_train(True)
        np.random.seed(epoch)
        # train
        for data in tqdm(train_loader.create_dict_iterator(), desc='train', leave=False):
            x_shot, x_query = data['data'][:, :, :args.num_shots], data['data'][:, :,
                                                                                args.num_shots:]
            img_shape = x_query.shape[-3:]
            x_query = x_query.view(args.ep_per_batch, -1, *img_shape)  # bs*(way*n_query)*3*84*84
            label = util.make_nk_label(args.num_ways, n_query,
                                       args.ep_per_batch)  # bs*(way*n_query)
            loss = train_cell(x_shot, x_query, label)

            aves['tl'].add(loss.asnumpy())
            aves['ta'].add(net_with_loss.acc.asnumpy())
        # test
        net.set_train(False)
        # train_cell.set_train(False)
        for data in tqdm(test_loader.create_dict_iterator(), desc='test', leave=False):
            x_shot, x_query = data['data'][:, :, :args.num_shots], data['data'][:, :,
                                                                                args.num_shots:]
            img_shape = x_query.shape[-3:]
            x_query = x_query.view(args.ep_per_batch, -1, *img_shape)  # bs*(way*n_query)*3*84*84
            label = util.make_nk_label(args.num_ways, n_query,
                                       args.ep_per_batch)  # bs*(way*n_query)
            loss_val = net_with_loss(x_shot, x_query, label)
            aves['vl'].add(loss_val.asnumpy())
            aves['va'].add(net_with_loss.acc.asnumpy())
        _sig = int(-1)
        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append([aves[k]])
        t_epoch = util.time_str(timer_epoch.t())
        t_used = util.time_str(timer_used.t())
        t_estimate = util.time_str(timer_used.t() / epoch * args.max_epoch)
        print('epoch {}, train {:.4f}|{:.4f}, '
              'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                  epoch, aves['tl'], aves['ta'], aves['vl'], aves['va'],
                  t_epoch, t_used, t_estimate, _sig))

        if epoch % args.save_epoch == 0:
            path = os.path.join(save_path, 'epoch-{}.ckpt'.format(epoch))
            save_checkpoint(net, path)
            if max_va < aves['va']:
                path = os.path.join(save_path, 'max-va.ckpt')
                save_checkpoint(net, path)
        if args.run_offline == "True":
            md_save_path = os.path.join(args.train_url, save_path)
            mox.file.make_dirs(md_save_path)
            mox.file.copy_parallel(src_url=save_path, dst_url=md_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--root_path', default='./dataset/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', default='mini-imagenet')
    parser.add_argument('--encoder', default='resnet12')
    parser.add_argument('--load_encoder',
                        default='./save/classifier5_mini-imagenet_resnet12/epoch-17.ckpt')
    parser.add_argument('--ep_per_batch', type=int, default=4)
    parser.add_argument('--ep_batch', type=int, default=200)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--visualize_datasets', default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5.e-4)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--classifier', default='linear-classifier')
    parser.add_argument('--n_classes', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--eval_fs_epoch', type=int, default=4)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--milestones', default=[90])
    parser.add_argument('--num_ways', type=int, default=5)
    parser.add_argument('--num_shots', type=int, default=1)  #
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    parser.add_argument('--run_offline', type=str, default=False, help='run in offline')

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    if args.device_target == 'GPU' or args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)
        if args.run_offline == "True":
            print("run_online--")
            import moxing as mox
            mox.file.copy_parallel(src_url=args.data_url, dst_url=args.root_path)
    else:
        raise ValueError("Unsupported platform.")
    main()
