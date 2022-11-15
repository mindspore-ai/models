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
"""
eval.
"""
import argparse
import logging
import os
from model.wide_resnet import WideResNet32
from model.resnet import ResNet18
from utils import evaluate_standard, evaluate_standard_random_norms
from utils import set_norm_list, set_random_norm, set_random_norm_mixed
from datasets import get_loaders
from mindspore import load_checkpoint, load_param_into_net
import mindspore as ms
from attacks import PGD, FGSM, MIFGSM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='ResNet18', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--epsilon', default=8, type=int)

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save log')

    parser.add_argument('--attack_type', default='pgd')

    parser.add_argument('--tau', default=0.1, type=float, help='tau in cw inf')

    parser.add_argument('--max_iterations', default=100, type=int, help='max iterations in cw attack')

    parser.add_argument('--c', default=1e-4, type=float, help='c in torchattacks')
    parser.add_argument('--steps', default=1000, type=int, help='steps in torchattacks')

    parser.add_argument('--norm_type', default='gn_32', type=str,
                        help='type of normalization to use. E.g., bn, in, gn_(group num), gbn_(group num)')

    # random setting
    parser.add_argument('--random_norm_training', action='store_true',
                        help='enable random norm training')
    parser.add_argument('--num_group_schedule', default=None, type=int, nargs='*',
                        help='group schedule for gn/gbn in random gn/gbn training')
    parser.add_argument('--random_type', default='None', type=str,
                        help='type of normalizations to be included besides gn/gbn, e.g. bn/in/bn_in')
    parser.add_argument('--gn_type', default='gn', type=str,
                        choices=['gn', 'gnr', 'gbn', 'gbnr', 'gn_gbn', 'gn_gbnr', 'gnr_gbn', 'gnr_gbnr'],
                        help='type of gn/gbn to use')
    parser.add_argument('--mixed', action='store_true', help='if use different norm for different layers')

    return parser.parse_args()


def evaluate_attack(model, test_loader, args, atk, atk_name, logger):
    test_acc = 0
    total_num = 0
    if args.dataset == 'cifar10':
        for _, (X, y) in enumerate(test_loader):
            # random select a path to attack
            if args.random_norm_training:
                if args.mixed:
                    set_random_norm_mixed(args, model)
                else:
                    set_random_norm(args, model)
            # get attack image
            X_adv = atk.forward(X, y)
            # random select a path to infer
            if args.random_norm_training:
                if args.mixed:
                    set_random_norm_mixed(args, model)
                else:
                    set_random_norm(args, model)
            output = model(X_adv)
            test_acc += (output.argmax(1) == y).sum().asnumpy()
            total_num += y.shape[0]
    elif args.dataset == 'cifar100':
        for _, (X, _, y) in enumerate(test_loader):
            # random select a path to attack
            if args.random_norm_training:
                if args.mixed:
                    set_random_norm_mixed(args, model)
                else:
                    set_random_norm(args, model)
            # get attack image
            X_adv = atk.forward(X, y)
            # random select a path to infer
            if args.random_norm_training:
                if args.mixed:
                    set_random_norm_mixed(args, model)
                else:
                    set_random_norm(args, model)
            output = model(X_adv)
            test_acc += (output.argmax(1) == y).sum().asnumpy()
            total_num += y.shape[0]

    pgd_acc = test_acc / total_num

    logger.info(atk_name)
    logger.info('adv: %.4f \t', pgd_acc)


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU', save_graphs=False)

    logger = logging.getLogger(__name__)

    args = get_args()

    args.save_dir = os.path.join('logs', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)
    log_path = os.path.join(args.save_dir, 'output_test.log')

    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)

    logger.info(args)

    assert isinstance(args.pretrain, str) and os.path.exists(args.pretrain)

    # get datasets
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()
    logger.info('Dataset: %s', args.dataset)
    _, test_loader, ds_norm = get_loaders(args.data_dir,
                                          args.batch_size, dataset=args.dataset, worker=args.worker, norm=False)

    # setup network
    if args.network == 'ResNet18':
        net = ResNet18
    elif args.network == 'WideResNet32':
        net = WideResNet32
    else:
        print('Wrong network:', args.network)

    if args.random_norm_training:
        assert args.num_group_schedule is not None
        norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type,
                                  args.gn_type)
        model = net(norm_list, num_classes=args.num_classes, normalize=ds_norm)
    else:
        model = net(args.norm_type, num_classes=args.num_classes, normalize=ds_norm)

    norm_list = set_norm_list(args.num_group_schedule[0], args.num_group_schedule[1], args.random_type,
                              args.gn_type)

    # load parameters
    params_dict = load_checkpoint(args.pretrain)
    load_param_into_net(model, params_dict)
    model.set_train(False)
    print('successful load from {}'.format(args.pretrain))

    # nature testing
    if args.random_norm_training:
        logger.info('Evaluating with standard images with random norms...')
        _, nature_acc = evaluate_standard_random_norms(test_loader, model, args)
        logger.info('Nature Acc: %.4f \t', nature_acc)
    else:
        logger.info('Evaluating with standard images...')
        _, nature_acc = evaluate_standard(test_loader, model, args)
        logger.info('Nature Acc: %.4f \t', nature_acc)

    logger.info('Nature Testing done.')

    # attack testing
    if args.attack_type == 'pgd':
        atk = PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
        evaluate_attack(model, test_loader, args, atk, 'pgd', logger)
    elif args.attack_type == 'fgsm':
        atk = FGSM(model, eps=8/255)
        evaluate_attack(model, test_loader, args, atk, 'fgsm', logger)
    elif args.attack_type == 'mifgsm':
        atk = MIFGSM(model, eps=8/255, alpha=2/255, steps=5, decay=1.0)
        evaluate_attack(model, test_loader, args, atk, 'mifgsm', logger)

    logger.info('Attack Testing done.')

if __name__ == "__main__":
    main()
