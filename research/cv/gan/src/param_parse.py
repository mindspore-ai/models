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
'''GAN parameter parser'''

import argparse


def parameter_parser():
    '''parameter parser'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_ascend", type=int, default=8,
                        help="number of ascend threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    parser.add_argument("-d", default="mnist")
    parser.add_argument("-l", type=float, default=1000)
    parser.add_argument("-c", "--cross_val", default=10, type=int, help="Number of cross valiation folds")
    parser.add_argument("--sigma_start", default=-1, type=float)
    parser.add_argument("--sigma_end", default=0, type=float)
    parser.add_argument("-s", "--sigma", default=None)
    parser.add_argument("--batch_size_t", type=int, default=10, help="size of the test batches")
    parser.add_argument("--batch_size_v", type=int, default=1000, help="size of the valid batches")
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')
    parser.add_argument("--data_path", type=str, default="mnist/", help="dataset path")  # change to train data path
    parser.add_argument("--ckpt_path", type=str, default="", help="eval ckpt path")  # change to eval ckpt path
    parser.add_argument("--distribute", type=bool, default=False, help="Run distribute, default is false.")
    return parser.parse_args()
