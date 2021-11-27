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
Evaluation script
"""
import argparse
import time
import mindspore
from mindspore import Tensor
from mindspore import context, set_seed, load_checkpoint
import numpy as np
import src.data_process as dp
from src.dgcn import DGCN
from src.metrics import LossAccuracyWrapper
from src.utilities import diffusion_fun_sparse, diffusion_fun_improved_ppmi_dynamic_sparsity
from src.config import ConfigDGCN

def main():
    """
    run eval
    """
    parser = argparse.ArgumentParser(description="DGCN eval")
    parser.add_argument("--device_id", help="device_id", default=0, type=int)
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend", "GPU"], help="device target (default: Ascend)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed for sklearn pre-training. Default is 42.")
    parser.add_argument("--dataset", type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'], help="Dataset.")

    args = parser.parse_args()
    set_seed(args.seed)
    # Runtime
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    # Create network
    config = ConfigDGCN()
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = dp.load_graph_data(args.dataset)
    test_mask = np.reshape(test_mask, (-1, len(test_mask)))
    train_mask = np.reshape(train_mask, (-1, len(test_mask)))
    val_mask = np.reshape(val_mask, (-1, len(test_mask)))
    #### obtain diffusion and ppmi matrices
    diffusions = diffusion_fun_sparse(adj.tocsc())  # Normalized adjacency matrix
    ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(adj, path_len=config.path_len, k=1.0)  # ppmi matrix

    adj = Tensor(adj.toarray(), mindspore.float32)
    features = Tensor(features.toarray(), mindspore.float32)
    diffusions = Tensor(diffusions.toarray(), mindspore.float32)
    labels = Tensor(labels, mindspore.float32)
    y_test = Tensor(y_test, mindspore.float32)
    y_val = Tensor(y_val, mindspore.float32)
    ppmi = Tensor(ppmi.toarray(), mindspore.float32)

    #### construct the classifier model ####

    feature_size = features.shape[1]
    label_size = y_train.shape[1]
    layer_sizes = [(feature_size, config.hidden1), (config.hidden1, label_size)]

    print("Convolution Layers:" + str(layer_sizes))

    dgcn_net_eval = DGCN(input_dim=features.shape[1], hidden_dim=config.hidden1, output_dim=y_train.shape[1],
                         dropout=config.dropout)
    ckpt_path = args.checkpoint
    load_checkpoint(ckpt_path, net=dgcn_net_eval)
    dgcn_net_eval.add_flags_recursive(fp16=True)
    eval_net = LossAccuracyWrapper(dgcn_net_eval, y_test, test_mask, weight_decay=config.weight_decay)
    t_eval = time.time()
    eval_net.set_train(False)
    ret = 0.5
    eval_result = eval_net(diffusions, ppmi, features, ret)
    eval_loss = eval_result[0].asnumpy()
    eval_accuracy = eval_result[1].asnumpy()
    print("Eval results:", "loss=", "{:.5f}".format(eval_loss),
          "accuracy=", "{:.5f}".format(eval_accuracy), "time=", "{:.5f}".format(time.time() - t_eval))

if __name__ == "__main__":
    main()
      