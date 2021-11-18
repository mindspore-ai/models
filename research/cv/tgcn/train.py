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
Training script
"""
import os
import time
import argparse
from mindspore.communication import init, get_rank
from mindspore.context import ParallelMode
from mindspore import dtype as mstype
from mindspore import set_seed, nn, context, Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from src.config import ConfigTGCN
from src.dataprocess import load_adj_matrix, load_feat_matrix, generate_dataset_ms, generate_dataset_ms_distributed
from src.task import SupervisedForecastTask
from src.model.loss import TGCNLoss
from src.callback import RMSE, SaveCallback


def run_train(args):
    """
    Run training
    """
    # Config initialization
    config = ConfigTGCN()
    # Set global seed for MindSpore and NumPy
    set_seed(config.seed)
    # ModelArts runtime, datasets and network initialization
    if args.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=device_id)
        mox.file.copy_parallel(src_url=args.data_url, dst_url='./data')
        if args.distributed:
            device_num = int(os.getenv('RANK_SIZE'))
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            training_set = generate_dataset_ms_distributed(config, training=True, abs_path=config.data_path)
            eval_set = generate_dataset_ms_distributed(config, training=False, abs_path=config.data_path)
            _, max_val = load_feat_matrix(config.dataset, config.data_path)
            net = SupervisedForecastTask(load_adj_matrix(config.dataset, config.data_path),
                                         config.hidden_dim, config.pre_len)
        else:
            training_set = generate_dataset_ms(config, training=True, abs_path=config.data_path)
            eval_set = generate_dataset_ms(config, training=False, abs_path=config.data_path)
            _, max_val = load_feat_matrix(config.dataset)
            net = SupervisedForecastTask(load_adj_matrix(config.dataset), config.hidden_dim, config.pre_len)
    # Offline runtime, datasets and network initialization
    else:
        if args.distributed:
            if config.device == 'Ascend':
                device_id = int(os.getenv('DEVICE_ID'))
                context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)
                init()
            elif config.device == 'GPU':
                context.set_context(mode=context.GRAPH_MODE, device_target=config.device)
                init('nccl')
            else:
                raise RuntimeError("Wrong device type. Supported devices: 'Ascend', 'GPU'")
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            training_set = generate_dataset_ms_distributed(config, training=True, abs_path=config.data_path)
            eval_set = generate_dataset_ms_distributed(config, training=False, abs_path=config.data_path)
            _, max_val = load_feat_matrix(config.dataset, config.data_path)
            net = SupervisedForecastTask(load_adj_matrix(config.dataset, config.data_path),
                                         config.hidden_dim, config.pre_len)
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target=config.device, device_id=args.device_id)
            training_set = generate_dataset_ms(config, training=True, abs_path=config.data_path)
            eval_set = generate_dataset_ms(config, training=False, abs_path=config.data_path)
            _, max_val = load_feat_matrix(config.dataset, config.data_path)
            net = SupervisedForecastTask(
                load_adj_matrix(config.dataset, config.data_path),
                config.hidden_dim,
                config.pre_len,
            )
    # Mixed precision
    net.tgcn.tgcn_cell.graph_conv1.matmul.to_float(mstype.float16)
    net.tgcn.tgcn_cell.graph_conv2.matmul.to_float(mstype.float16)
    # Loss function
    loss_fn = TGCNLoss()
    # Optimizer
    optimizer = nn.Adam(net.trainable_params(), config.learning_rate, weight_decay=config.weight_decay)
    # Create model
    model = Model(net, loss_fn, optimizer, {'RMSE': RMSE(max_val)})
    # Training
    time_start = time.time()
    callbacks = [LossMonitor(), TimeMonitor()]
    if args.distributed:
        print("==========Distributed Training Start==========")
        save_callback = SaveCallback(model, eval_set, config)
        if get_rank() == 0:
            callbacks = [LossMonitor(), TimeMonitor(), save_callback]
        elif config.save_best:
            callbacks = [LossMonitor()]
    else:
        print("==========Training Start==========")
        save_callback = SaveCallback(model, eval_set, config)
        callbacks.append(save_callback)
    model.train(config.epochs, training_set,
                callbacks=callbacks,
                dataset_sink_mode=config.data_sink)
    time_end = time.time()
    if args.distributed:
        print("==========Distributed Training End==========")
    else:
        print("==========Training End==========")
    print("Training time in total:", '{:.6f}'.format(time_end - time_start), "s")
    # Save outputs (checkpoints) on ModelArts
    if args.run_modelarts:
        mox.file.copy_parallel(src_url='./checkpoints', dst_url=args.train_url)


if __name__ == '__main__':
    # Set universal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', help="DEVICE_ID", type=int, default=0)
    parser.add_argument('--distributed', help="distributed training", type=bool, default=False)
    # Set ModelArts arguments
    parser.add_argument('--run_modelarts', help="ModelArts runtime", type=bool, default=False)
    parser.add_argument('--data_url', help='ModelArts location of data', type=str, default=None)
    parser.add_argument('--train_url', help='ModelArts location of training outputs', type=str, default=None)
    run_args = parser.parse_args()
    # Training
    run_train(run_args)
