# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""eval cgan"""

import argparse
import numpy as np
from mindspore import Tensor
from mindspore import context
import mindspore.ops.operations as P
from mindspore.common import dtype as mstype
from src.dataset import get_real_valued_mnist
from src.tools import load_ckpt
from src.reporter import Reporter
from src.model import Generator
from src.parzen_numpy import cross_validate_sigma, parzen_estimation, get_nll

def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='device target, Ascend or GPU (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size id of training (Default: 200)')
    parser.add_argument('--G_ckpt_path', type=str,
                        default=None, help='checkpoint path of generator')
    parser.add_argument('--data_path', type=str, default='data/train',
                        help='dataset dir (default data/train)')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='output path of training (default ./output)')
    parser.add_argument("--real_valued_mnist", type=bool, default=True,
                        help='If load real valued MNIST dataset(default False)')
    args = parser.parse_args()

    context.set_context(device_id=args.device_id,
                        mode=context.GRAPH_MODE,
                        device_target=args.device_target)

    return args


def main():
    # before training, we should set some arguments
    args = preLauch()

    # training argument
    sigma = 0.1
    classes_num = 10
    img_channels = 1
    input_dim = 100
    embed_size = 100
    generator_size = 10000
    batch_size = args.batch_size
    G_ckpt_path = args.G_ckpt_path
    data_path = args.data_path
    output_path = args.output_path
    if_parzen_estimates = True
    if_cross_validate_sigma = True

    # create G Cell
    netG = Generator(input_dim, img_channels, classes_num, embed_size=embed_size)
    if G_ckpt_path is not None:
        load_ckpt(G_ckpt_path, netG)
    netG.set_train(False)

    reporter = Reporter(output_path=output_path, batch_size=batch_size, stage='eval')
    reporter.info('==========start predict %s===============')
    reporter.start_predict()

    steps = np.ceil(generator_size / batch_size).astype(np.int32)
    sample_list = []
    resize = P.ResizeBilinear((28, 28))
    for step in range(steps):
        noise = Tensor(np.random.normal(size=(batch_size, input_dim)),
                       dtype=mstype.float32)
        label = Tensor((np.arange(noise.shape[0]) % 10), dtype=mstype.int32)
        samples = netG(noise, label)  # 10000

        samples = resize(samples)
        samples = samples.asnumpy()
        sample_list.append(samples)
        reporter.visualizer_eval(samples, step)
    reporter.end_predict(step)

    samples = np.concatenate(sample_list, axis=0)
    samples = samples[:generator_size]
    sample_data = samples.reshape(-1, 784).astype('float32')

    if if_parzen_estimates:
        reporter.info('start Gaussian Parzen window log-likelihood estimate')
        dataset_train = get_real_valued_mnist(dataset_dir=data_path, usage='train', out_as_numpy=True)
        dataset_train = dataset_train[0]

        valid_data = dataset_train[10000:10000 + generator_size].reshape(-1, 784).astype('float32') / 255
        test_data = dataset_train[50000:50000 + generator_size].reshape(-1, 784).astype('float32') / 255

        if if_cross_validate_sigma:
            sigma_range = np.logspace(-1, 0, 10)
            # get appropriate sigmas
            sigma, _ = cross_validate_sigma(sample_data, valid_data, sigma_range, batch_size, 2)
        else:
            sigma = sigma
        print(sigma)

        parzen = parzen_estimation(sample_data, sigma, mode='gauss')  # the density estimate p(x)
        ll = get_nll(test_data, parzen, batch_size=batch_size)  # get ll on multi batch
        se = ll.std() / np.sqrt(test_data.shape[0])
        info = "Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se)
        reporter.info(info)


if __name__ == '__main__':
    main()
