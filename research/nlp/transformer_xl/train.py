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
"""Transformer training script."""

import math
import os
import argparse
import numpy as np
import mindspore as ms
from mindspore import DynamicLossScaleManager
from mindspore.communication import init
import mindspore.nn.optim as optim
import mindspore.context as context
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model
from src.callback.eval import EvalDuringTrain, doEval
from src.callback.log import TrainLogger
from src.callback.flag import FlagModifiedCallback
from src.model.mem_transformer import MemTransformerLM
from src.model.mem_transformer_for_ascend import MemTransformerLM as MemTransformerLMAscend
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id
from src.model_utils.config import config
from src.utils.dataset_util import get_dataset
from src.utils.nnUtils import uniform_, normal_, constant_
from src.metric.calc import bpc


def init_weight(weight, _config):
    if _config.init == 'uniform':
        uniform_(weight, -_config.init_range, _config.init_range)
    elif _config.init == 'normal':
        normal_(weight, 0.0, _config.init_std)


def init_bias(bias):
    constant_(bias, 0.0)


def weights_init_AdaptiveEmbedding(m, config1):
    if hasattr(m, 'emb_projs'):
        for i in range(len(m.emb_projs)):
            if m.emb_projs[i] is not None:
                normal_(m.emb_projs[i], 0.0, config1.proj_init_std)


def weights_init_ProjectedAdaptiveLogSoftmax(m, config1):
    if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
        init_weight(m.cluster_weight, config1)
    if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
        init_bias(m.cluster_bias)
    if hasattr(m, 'out_projs'):
        for i in range(len(m.out_projs)):
            if m.out_projs[i] is not None:
                normal_(m.out_projs[i], 0.0, config1.proj_init_std)


def weights_init_LayerNorm(m, config1):
    if hasattr(m, 'weight'):
        normal_(m.weight, 1.0, config1.init_std)
    if hasattr(m, 'bias') and m.bias is not None:
        init_bias(m.bias)


def weights_init_TransformerLM(m, config1):
    if hasattr(m, 'r_emb'):
        init_weight(m.r_emb, config1)
    if hasattr(m, 'r_w_bias'):
        init_weight(m.r_w_bias, config1)
    if hasattr(m, 'r_r_bias'):
        init_weight(m.r_r_bias, config1)
    if hasattr(m, 'r_bias'):
        init_bias(m.r_bias)


def weights_init(m, config1):
    classname = m.__class__.__name__
    if classname.find('AdaptiveEmbedding') != -1:
        weights_init_AdaptiveEmbedding(m, config1)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, config1)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        weights_init_ProjectedAdaptiveLogSoftmax(m, config1)
    elif classname.find('LayerNorm') != -1:
        weights_init_LayerNorm(m, config1)
    elif classname.find('TransformerLM') != -1:
        weights_init_TransformerLM(m, config1)


def get_optimizer(_config, net, scheduler):
    """
    get optimizer: adam,sgd
    Args:
        _config:
        net:
        scheduler:

    Returns:
        optimizer:
        optimizer_sparse: default is None
    """
    optimizer = optimizer_sparse = None
    lr = dynamic_lr()
    if _config.optim.lower() == 'sgd':
        if _config.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in net.trainable_params():
                if len(param) == len(net.word_emb.embedding_table):
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, learning_rate=_config.lr * 2)
            optimizer = optim.SGD(dense_params, learning_rate=_config.lr, momentum=_config.mom)
        else:
            optimizer = optim.SGD(net.trainable_params(), learning_rate=lr,
                                  momentum=_config.mom)
    elif _config.optim.lower() == 'adam':
        if _config.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in net.trainable_params():
                if len(param) == len(net.word_emb.embedding_table):
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=lr)
            optimizer = optim.Adam(dense_params, learning_rate=lr)
        else:
            optimizer = optim.Adam(net.trainable_params(), learning_rate=lr)
    elif _config.optim.lower() == 'adamw':
        filter_word = ['norm', 'bias']

        def decay_filter(p):
            name = p.name
            for fw in filter_word:
                if name.find(fw) != -1:
                    return False
            return True

        params = net.trainable_params()
        decay_params = list(filter(decay_filter, params))
        other_params = list(filter(lambda x: not decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': _config.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': params}]
        optimizer = optim.AdamWeightDecay(group_params, learning_rate=lr)
    elif _config.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(net.trainable_params(), learning_rate=lr)
    return optimizer, optimizer_sparse


def rsqrt_decay(warmup_steps, current_step):
    return float(max([current_step, warmup_steps])) ** -0.5


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, total_steps):
    decay_steps = total_steps - warmup_steps
    linear_decay = (total_steps - current_step) / decay_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * current_step / decay_steps))
    decayed = linear_decay * cosine_decay + 0.00001
    learning_rate = decayed * base_lr
    return learning_rate


def dynamic_lr():
    """dynamic learning rate generator"""
    base_lr = config.lr
    min_lr = config.lr_min
    total_steps = int(config.max_step)
    warmup_steps = int(config.warmup_step)
    lr = []
    for i in range(total_steps):
        curr_lr = min_lr
        if i < warmup_steps:
            curr_lr = max(curr_lr, linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * config.warmup_ratio))
        else:
            curr_lr = max(curr_lr, a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))
        lr.append(curr_lr)
    return lr


def get_scheduler(_config):
    scheduler = scheduler_sparse = None
    if _config.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        from src.utils.additional_algorithms import CosineAnnealingLR

        scheduler = CosineAnnealingLR(total_step=_config.max_step, lr=_config.lr, min_lr=_config.eta_min)

    elif _config.scheduler == 'inv_sqrt':
        pass

    elif _config.scheduler == 'dev_perf':
        pass
    elif _config.scheduler == 'constant':
        pass
    return scheduler, scheduler_sparse


def set_seed():
    np.random.seed(config.seed)
    ms.set_seed(config.seed)


def construct_args():
    parser = argparse.ArgumentParser(description='Transformer-XL train running')
    parser.add_argument('--datadir', default='./data/enwik8',
                        help='Directory contains enwik8 dataset.')
    parser.add_argument('--dataset', default='enwik8',
                        help='Dataset Name.', choices=["enwik8", "text8"])
    parser.add_argument('--train_url', default="./", help='Directory of training output.')
    parser.add_argument("--device_target", type=str, default="GPU", help="Device Target, default GPU",
                        choices=["Ascend", "GPU"])

    args = parser.parse_args()
    return args


def main():
    # Set the random seed manually for reproducibility.
    set_seed()

    args = construct_args()
    datadir = args.datadir
    dataset = args.dataset
    config.device_target = args.device_target

    device_id = get_device_id()
    device_num = get_device_num()

    if args.device_target == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()

    elif args.device_target == 'GPU':
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", max_device_memory="39.0GB",
                            enable_graph_kernel=True)
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    ###############################################################################
    # Load data
    ###############################################################################

    dataset = get_dataset(datadir, dataset)
    ntokens = len(dataset.vocab)
    config.n_token = ntokens

    # adaptive softmax / embedding
    cutoffs = []

    ###############################################################################
    # Build the model
    ###############################################################################
    if args.device_target == 'Ascend':
        net = MemTransformerLMAscend(ntokens, config.n_layer, config.n_head, config.d_model,
                                     config.d_head, config.d_inner, config.dropout, config.dropatt,
                                     batch_size=config.batch_size,
                                     d_embed=config.d_embed, div_val=config.div_val,
                                     pre_lnorm=config.pre_lnorm, tgt_len=config.tgt_len,
                                     ext_len=config.ext_len, mem_len=config.mem_len, eval_tgt_len=config.eval_tgt_len,
                                     cutoffs=cutoffs, same_length=config.same_length, clamp_len=config.clamp_len)
    else:
        net = MemTransformerLM(ntokens, config.n_layer, config.n_head, config.d_model,
                               config.d_head, config.d_inner, config.dropout, config.dropatt,
                               batch_size=config.batch_size,
                               d_embed=config.d_embed, div_val=config.div_val,
                               pre_lnorm=config.pre_lnorm, tgt_len=config.tgt_len,
                               ext_len=config.ext_len, mem_len=config.mem_len, eval_tgt_len=config.eval_tgt_len,
                               cutoffs=cutoffs, same_length=config.same_length, clamp_len=config.clamp_len)

    # ensure embedding init is not overridden by out_layer in case of weight sharing
    weights_init(net, config)
    weights_init(net.word_emb, config)

    config.n_all_param = sum([p.size for p in net.trainable_params()])
    config.n_nonemb_param = sum([p.size for p in net.layers.trainable_params()])

    # scheduler
    scheduler, _ = get_scheduler(config)
    # optimizer
    optimizer, _ = get_optimizer(config, net, scheduler)

    if device_id == 0:
        print('=' * 100)
        for k, v in config.__dict__.items():
            print('    - {} : {}'.format(k, v))
        print('=' * 100)
        print('#params = {}'.format(config.n_all_param))
        print('#non emb params = {}'.format(config.n_nonemb_param))

    ###############################################################################
    # Training code
    ###############################################################################

    config.n_batch = dataset.get_train_generator().n_batch
    config.max_epoch = math.ceil(config.max_step / config.n_batch)

    rank_size, rank_id = get_device_num(), get_rank_id()

    train_dataset = GeneratorDataset(source=dataset.get_train_generator(), column_names=['data', 'target'],
                                     num_shards=rank_size, shard_id=rank_id, shuffle=False)
    # Due to the mems mechanism, it is not possible to perform multi-card segmentation on the valid and test datasets
    valid_dataset = GeneratorDataset(source=dataset.get_valid_generator(), column_names=['data', 'target'],
                                     shuffle=False)
    test_dataset = GeneratorDataset(source=dataset.get_test_generator(), column_names=['data', 'target'],
                                    shuffle=False)

    # Train #

    flagModifiedCallback = FlagModifiedCallback()
    train_log = TrainLogger(per_print_times=config.log_interval, n_batch=config.n_batch)
    if args.device_target == 'Ascend':
        if device_id == 0:
            config_ck = CheckpointConfig(save_checkpoint_steps=4000, keep_checkpoint_max=4)
            modelCheckpoint = ModelCheckpoint(directory=os.path.join(config.checkpoint_url, 'device_' + str(device_id)),
                                              config=config_ck)
            callbacks = [flagModifiedCallback, train_log, modelCheckpoint]
        else:
            callbacks = [flagModifiedCallback, train_log]

        loss_scale_manager = DynamicLossScaleManager()

        model = Model(network=net, loss_fn=None, optimizer=optimizer, metrics=None,
                      loss_scale_manager=loss_scale_manager)
        model.train(config.max_step, train_dataset, sink_size=1, callbacks=callbacks)
    else:
        evalDuringTrain = EvalDuringTrain(dataset=valid_dataset, per_print_times=config.eval_interval,
                                          tgt_len=config.tgt_len, ext_len=config.ext_len, mem_len=config.mem_len,
                                          eval_tgt_len=config.eval_tgt_len)

        model = Model(network=net, loss_fn=None, optimizer=optimizer, metrics=None)
        model.train(config.max_step, train_dataset, sink_size=1,
                    callbacks=[flagModifiedCallback, train_log, evalDuringTrain])

    # Test #

    if device_id == 0 and args.device_target == 'GPU':
        test_loss = doEval(net=net, dataset=test_dataset, tgt_len=config.tgt_len, ext_len=config.ext_len,
                           mem_len=config.mem_len, eval_tgt_len=config.eval_tgt_len)
        print('=' * 100)
        if config.dataset in ['enwik8', 'text8']:
            print('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_loss, bpc(test_loss)))
        print('=' * 100)


if __name__ == '__main__':
    main()
