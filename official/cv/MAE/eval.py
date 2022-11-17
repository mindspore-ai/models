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

import argparse

from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint

from src.models.vit import FineTuneVit
from src.datasets.dataset import get_dataset
from src.logger import get_logger
from src.helper import parse_with_config, str2bool, cloud_context_init
from src.models.eval_engine import get_eval_engine


def main(args):
    local_rank, device_num = cloud_context_init(seed=args.seed, use_parallel=args.use_parallel,
                                                context_config=args.context, parallel_config=args.parallel)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))

    # evaluation dataset
    eval_dataset = get_dataset(args, is_train=False)
    per_step_size = eval_dataset.get_dataset_size()
    if args.per_step_size:
        per_step_size = args.per_step_size
    args.logger.info("Create eval dataset finish, data size:{}".format(per_step_size))

    net = FineTuneVit(batch_size=args.batch_size, patch_size=args.patch_size,
                      image_size=args.image_size, dropout=args.dropout,
                      num_classes=args.num_classes, **args.model_config)
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # define optimizer
    optimizer = nn.AdamWeightDecay(net.trainable_params(),
                                   learning_rate=args.learning_rate,
                                   weight_decay=args.weight_decay,
                                   beta1=args.beta1,
                                   beta2=args.beta2)

    # load eval ckpt
    if args.use_ckpt:
        params_dict = load_checkpoint(args.use_ckpt)
        net_not_load = net.init_weights(params_dict)
        args.logger.info(f"===============net_not_load================{net_not_load}")

    # define Model and begin training
    model = Model(net, metrics=eval_engine.metric, optimizer=optimizer,
                  eval_network=eval_engine.eval_network,
                  loss_scale_manager=None, amp_level="O3")

    eval_engine.set_model(model)
    # equal to model._init(dataset, sink_size=per_step_size)
    eval_engine.compile(sink_size=per_step_size)
    eval_engine.eval()
    output = eval_engine.get_result()
    args.logger.info('accuracy={:.6f}'.format(float(output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_path', default="/home/work/user-job-dir/mae_mindspore/config/eval.yaml", help='YAML config files')
    parser.add_argument(
        '--use_parallel', default=False, type=str2bool, help='use parallel config.')

    args_ = parse_with_config(parser)

    main(args_)
