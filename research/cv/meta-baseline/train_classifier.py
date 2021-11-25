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
train_classifier
"""
import os
import argparse
import numpy as np
import mindspore.dataset as ds
from mindspore import nn
from mindspore import save_checkpoint
from mindspore.nn import piecewise_constant_lr
from mindspore import context
from mindspore import ParameterTuple
from mindspore import ops
import src.util as util
from src.data.IterSamplers import CategoriesSampler
from src.data.mini_Imagenet import MiniImageNet
from src.model.classifier import Classifier, ClassifierWithLossCell
from src.model.meta_eval import MetaEval
from tqdm import tqdm


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

    def construct(self, data, label):
        """
        :param data:
        :param label:
        :return:
        """
        weights = self.weights
        loss = self.network(data, label)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        return ops.depend(loss, self.optimizer(grads))


def main():
    """
    train
    :return:
    """
    util.ensure_path(save_path)
    # train set
    root_path = os.path.join(args.root_path, args.dataset)
    n_way, n_query, n_shots = 5, 15, [1, 5]
    if args.dataset == 'mini-imagenet':
        trainset = MiniImageNet(root_path, 'train')
        testset = MiniImageNet(root_path, 'test')
    else:
        print('not found error')

    trainloader = ds.GeneratorDataset(trainset, ['data', 'label'], shuffle=True).batch(
        args.batch_size)
    # test set
    fs_loaders = []
    for n_shot in n_shots:
        fs_loader = CategoriesSampler(testset.data, testset.label, n_way, n_shot + n_query,
                                      args.ep_batch, args.ep_per_batch)
        fs_loaders.append(ds.GeneratorDataset(fs_loader, ['data'], shuffle=True))

    # define network
    net = Classifier(args.n_classes)
    net_with_loss = ClassifierWithLossCell(net)
    # define opt
    train_batch = trainset.len // args.batch_size
    multiStepLR = piecewise_constant_lr([(args.max_epoch - 10) * train_batch,
                                         args.max_epoch * train_batch], [args.lr, args.lr * 0.1])

    net_opt = nn.SGD(params=net.trainable_params(), learning_rate=multiStepLR,
                     weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    train_cell = TrainOneStepCell(net_with_loss, net_opt)

    eval_loss_fn = MetaEval()
    timer_used = util.Timer()
    timer_epoch = util.Timer()
    max_va = 0.

    for epoch in range(1, args.max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta']
        for n_shot in n_shots:
            aves_keys += ['fsa-' + str(n_shot)]
        aves = {k: util.Averager() for k in aves_keys}
        # pre train
        net.set_train(mode=True)
        for data in tqdm(trainloader.create_dict_iterator(), desc='train', leave=False):
            loss = train_cell(data['data'], data['label'])
            acc = net_with_loss.acc
            aves['tl'].add(loss.asnumpy())
            aves['ta'].add(acc.asnumpy())
        # few-shot eval
        if epoch == args.max_epoch or epoch % args.save_epoch == 0:
            net.set_train(mode=False)
            for i, n_shot in enumerate(n_shots):
                np.random.seed(0)
                for data in tqdm(fs_loaders[i].create_dict_iterator(), desc='test-' + str(n_shot),
                                 leave=False):
                    x_shot, x_query = data['data'][:, :, :n_shot], data['data'][:, :, n_shot:]
                    img_shape = x_query.shape[-3:]
                    x_query = x_query.view(args.ep_per_batch, -1, *img_shape)
                    label = util.make_nk_label(n_way, n_query, args.ep_per_batch)
                    acc_val, _ = eval_loss_fn.eval(x_shot, x_query, label, net.encoder)
                    aves['fsa-' + str(n_shot)].add(acc_val.asnumpy())
        # post
        for k, v in aves.items():
            aves[k] = v.item()
        t_epoch = util.time_str(timer_epoch.t())
        t_used = util.time_str(timer_used.t())
        t_estimate = util.time_str(timer_used.t() / epoch * args.max_epoch)

        print("epoch {},train loss {:.4f}, train acc {:.4f}".format(str(epoch), aves['tl'],
                                                                    aves['ta']))
        if epoch == args.max_epoch or epoch % args.save_epoch == 0:
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                print("epoch {}, {}-shot, val acc {:.4f}".format(str(epoch), n_shot, aves[key]))

        if epoch <= args.max_epoch:
            print("{} {}/{}".format(t_epoch, t_used, t_estimate))
        else:
            print("{}".format(t_epoch))

        path = os.path.join(save_path, 'epoch-{}.ckpt'.format(epoch))
        if epoch >= 15 and epoch % args.save_epoch == 0:
            save_checkpoint(net, path)
        if aves['fsa-' + str(5)] > max_va:
            max_va = aves['fsa-' + str(5)]
            save_checkpoint(net, os.path.join(save_path, 'max-va.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--root_path', default='./dataset/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', default='mini-imagenet')
    parser.add_argument('--encoder', default='resnet12')
    parser.add_argument('--load_encoder',
                        default='./save/classifier3_mini-imagenet_resnet12/epoch-70.ckpt')
    parser.add_argument('--ep_per_batch', type=int, default=4)
    parser.add_argument('--ep_batch', type=int, default=200)
    parser.add_argument('--max_epoch', type=int, default=25)
    parser.add_argument('--visualize_datasets', default=True)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5.e-4)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--classifier', default='linear-classifier')
    parser.add_argument('--n_classes', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--eval_fs_epoch', type=int, default=3)
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    parser.add_argument('--run_offline', type=str, default="False", help='run in offline')
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

    svname = 'classifier_{}'.format(args.dataset)
    svname += '_' + args.encoder
    save_path = os.path.join('./save/', svname)
    main()

    if args.run_offline == "True":
        md_save_path = os.path.join(args.train_url, save_path)
        mox.file.make_dirs(md_save_path)
        mox.file.copy_parallel(src_url=save_path, dst_url=md_save_path)
