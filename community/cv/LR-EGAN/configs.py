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
""" config module of LR-EGAN  """
import argparse


def parse_args():
    ''' parse args '''
    parser = argparse.ArgumentParser(description="Run Ensemble.")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Wether to train model or eval model')
    parser.add_argument('--resume_epoch', type=int, default=134,
                        help='Which epoch of saved model to load')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Data path')
    parser.add_argument('--data_name', nargs='?', default='shuttle',
                        help='Input data name.')
    parser.add_argument('--data_format', nargs='?', default='mat',
                        help='data format.')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--print_epochs', type=int, default=1,
                        help='print the loss per print_epochs.')
    parser.add_argument('--lr_g', type=float, default=0.0003,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_d', type=float, default=0.0003,
                        help='Learning rate of discriminator.')
    parser.add_argument('--active_rate', type=float, default=1,
                        help='the proportion of instances that need to be labeled.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch size.')
    parser.add_argument('--LR_flag', type=int, default=0,
                        help='wether use LR strategy.')
    parser.add_argument('--refurbish_rate', type=float, default=0.2,
                        help='refurbish_rate.')
    parser.add_argument('--noiseLabelRate', type=float, default=0.8,
                        help='noiseLabelRate.')
    parser.add_argument('--kFold', type=int, default=5,
                        help='k for k-fold cross validation.')
    parser.add_argument('--dim_z', type=int, default=128,
                        help='dim for latent noise.')
    parser.add_argument('--dis_layer', type=int, default=1,
                        help='hidden_layer number in dis.')
    parser.add_argument('--dis_activation_func', type=str, default="relu",
                        help='activation function on discriminator, include relu,sigmoid,tanh')
    parser.add_argument('--gen_layer', type=int, default=2,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_num', type=int, default=10,
                        help='the number of dis in ensemble.')
    parser.add_argument('--device', choices=['CPU', 'GPU', 'Ascend'], default='CPU',
                        help='used device,including CPU,GPU,Ascend')
    parser.add_argument('--mindspore_mode', choices=['PYNATIVE_MODE', 'GRAPH_MODE'], default='GRAPH_MODE',
                        help='mode mindspore used,including PYNATIVE_MODE,GRAPH_MODE')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of used device')
    parser.add_argument('--input_path', type=str, default='data_csv')
    parser.add_argument('--init_type', nargs='?', default="N02",
                        help='init method for both gen and dis, including ortho,N02,xavier')
    parser.add_argument('--print', type=int, default=0,
                        help='Print the learning procedure')
    return parser.parse_args()
