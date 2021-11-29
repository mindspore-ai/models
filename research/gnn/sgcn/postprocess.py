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
postprocess.
"""
import argparse
import os

import numpy as np
from mindspore import context
from mindspore import load_checkpoint

from src.ms_utils import calculate_auc


def softmax(x):
    """Softmax"""
    t_max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - t_max)  # subtracts each row with its max value
    t_sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / t_sum
    return f_x


def score_model(preds, test_pos, test_neg, weight, bias):
    """Score the model on the test set edges."""
    score_positive_edges = np.array(test_pos, dtype=np.int32).T
    score_negative_edges = np.array(test_neg, dtype=np.int32).T
    test_positive_z = np.concatenate((preds[score_positive_edges[0, :], :],
                                      preds[score_positive_edges[1, :], :]), axis=1)
    test_negative_z = np.concatenate((preds[score_negative_edges[0, :], :],
                                      preds[score_negative_edges[1, :], :]), axis=1)
    # operands could not be broadcast together with shapes (4288,128) (128,3)
    scores = np.dot(np.concatenate((test_positive_z, test_negative_z), axis=0), weight) + bias
    probability_scores = np.exp(softmax(scores))
    predictions = probability_scores[:, 0] / probability_scores[:, 0: 2].sum(1)
    # predictions = predictions.asnumpy()
    targets = [0] * len(test_pos) + [1] * len(test_neg)
    auc, f1 = calculate_auc(targets, predictions)
    return auc, f1


def get_acc():
    """get infer Accuracy."""
    parser = argparse.ArgumentParser(description='postprocess')
    parser.add_argument('--dataset_name', type=str, default='bitcoin-otc', choices=['bitcoin-otc', 'bitcoin-alpha'],
                        help='dataset name')
    parser.add_argument('--result_path', type=str, default='./ascend310_infer/input/', help='result Files')
    parser.add_argument('--label_path', type=str, default='', help='y_test npy Files')
    parser.add_argument('--mask_path', type=str, default='', help='test_mask npy Files')
    parser.add_argument("--checkpoint_file", type=str, default='sgcn_alpha_f1.ckpt', help="Checkpoint file path.")
    parser.add_argument("--edge_path", nargs="?",
                        default="./input/bitcoin_alpha.csv", help="Edge list csv.")
    parser.add_argument("--features-path", nargs="?",
                        default="./input/bitcoin_alpha.csv", help="Edge list csv.")
    parser.add_argument("--test-size", type=float,
                        default=0.2, help="Test dataset size. Default is 0.2.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")
    parser.add_argument("--spectral-features", default=True, dest="spectral_features", action="store_true")
    parser.add_argument("--reduction-iterations", type=int,
                        default=30, help="Number of SVD iterations. Default is 30.")
    parser.add_argument("--reduction-dimensions", type=int,
                        default=64, help="Number of SVD feature extraction dimensions. Default is 64.")
    args_opt = parser.parse_args()

    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=0)
    # Create network
    test_pos = np.load(os.path.join(args_opt.result_path, 'pos_test.npy'))
    test_neg = np.load(os.path.join(args_opt.result_path, 'neg_test.npy'))
    # Load parameters from checkpoint into network
    param_dict = load_checkpoint(args_opt.checkpoint_file)
    print(type(param_dict))
    print(param_dict)
    print(type(param_dict['regression_weights']))
    print(param_dict['regression_weights'])
    pred = np.fromfile('./result_Files/repos_0.bin', np.float32)

    if args_opt.dataset_name == 'bitcoin-otc':
        pred = pred.reshape(5881, 64)
    else:
        pred = pred.reshape(3783, 64)

    auc, f1 = score_model(pred, test_pos, test_neg, param_dict['regression_weights'].asnumpy(),
                          param_dict['regression_bias'].asnumpy())
    print("Test set results:", "auc=", "{:.5f}".format(auc), "f1=", "{:.5f}".format(f1))


if __name__ == '__main__':
    get_acc()
