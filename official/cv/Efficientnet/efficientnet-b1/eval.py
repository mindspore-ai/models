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
"""eval efficientnet."""
import ast
import timeit
import argparse

import mindspore.nn as nn
from mindspore import context, Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.loss import CrossEntropySmooth
from src.dataset import create_imagenet
from src.models.effnet import EfficientNet
from src.model_utils.moxing_adapter import moxing_wrapper
from src.config import organize_configuration
from src.config import efficientnet_b1_config_ascend as config


set_seed(1)


def parse_args():
    """Get parameters from command line."""
    parser = argparse.ArgumentParser("Evaluate efficientnet.")
    parser.add_argument("--data_url", type=str, default=None,
                        help="Storage path of dataset in OBS.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Storage path of dataset in offline machine.")
    parser.add_argument("--train_url", type=str, default=None,
                        help="Storage path of outputs in OBS.")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Storage path of outputs in offline machine.")
    parser.add_argument("--checkpoint_url", type=str, default=None,
                        help="Storage path of checkpoint in OBS.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Storage path of checkpoint in OBS.")
    parser.add_argument("--model", type=str, default="efficientnet-b1",
                        help="Specify the model to be trained.")
    parser.add_argument("--modelarts", type=ast.literal_eval, default=False,
                        help="Run on ModelArts or offline machines.")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "CPU", "GPU"],
                        help="Training platform.")
    args_opt = parser.parse_args()

    return args_opt


@moxing_wrapper(config)
def main():
    """Main function for model evaluation."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    dataset = create_imagenet(dataset_path=config.data_path, do_train=False, repeat_num=1,
                              input_size=config.input_size, batch_size=config.batchsize,
                              target=config.device_target, distribute=config.run_distribute)
    net = EfficientNet(width_coeff=config.width_coeff, depth_coeff=config.depth_coeff,
                       dropout_rate=config.dropout_rate, drop_connect_rate=config.drop_connect_rate,
                       num_classes=config.num_classes)
    params = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, params)
    net.set_train(False)

    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.num_classes)
    metrics = {"Loss": nn.Loss(),
               "Top_1_Acc": nn.Top1CategoricalAccuracy(),
               "Top_5_Acc": nn.Top5CategoricalAccuracy()}
    model = Model(network=net, loss_fn=loss, metrics=metrics)
    start_time = timeit.default_timer()
    res = model.eval(dataset)
    end_time = timeit.default_timer()
    print(res, flush=True)
    print("The time spent is {}s.".format(end_time - start_time), flush=True)


if __name__ == "__main__":
    args = parse_args()
    organize_configuration(cfg=config, args=args)
    main()
