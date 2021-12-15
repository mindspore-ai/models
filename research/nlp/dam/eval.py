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
# ===========================================================================
"""Test function"""
import os
import time

import mindspore
import mindspore.dataset as ds
from mindspore import context, Model
from mindspore.communication import init
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.net import DAMNet, PredictWithNet, DAMNetWithLoss
from src.metric import EvalMetric
from src import config as conf

device_num = int(os.getenv('RANK_SIZE'))
device_id = int(os.getenv('DEVICE_ID'))
rank_id = int(os.getenv('RANK_ID'))


mindspore.set_seed(1)


def evaluate(config):
    """Evaluate function."""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=device_id)

    if config.modelArts:
        import moxing as mox
        mox.file.shift('os', 'mox')
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        root = "/cache/"
        obs_data_path = config.data_url
        obs_ckpt_path = config.ckpt_path
        if config.model_name == "DAM_ubuntu":
            local_data_path = os.path.join(root, "ubuntu_data")
        else:
            local_data_path = os.path.join(root, "douban_data")
        local_ckpt_path = os.path.join(local_data_path, "ckpt")
        local_test_path = os.path.join(root, "test")
        mox.file.make_dirs(local_data_path)
        mox.file.make_dirs(local_ckpt_path)
        mox.file.make_dirs(local_test_path)

        print("############## Downloading data from OBS ##############")
        mox.file.copy_parallel(src_url=obs_data_path, dst_url=local_data_path)
        mox.file.copy_parallel(src_url=obs_ckpt_path, dst_url=local_ckpt_path)
        print("############### Downloading is completed ##############")
    else:
        if config.parallel:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        local_data_path = config.data_root
        local_ckpt_path = config.ckpt_path
        local_test_path = config.output_path
        if not os.path.exists(local_test_path):
            os.makedirs(local_test_path)

    test_data_path = os.path.join(local_data_path, config.test_data)
    print("************Starting loading data: ", test_data_path)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    dataset = ds.MindDataset(test_data_path,
                             columns_list=["turns", "turn_len", "response", "response_len", "label"],
                             shuffle=False, num_shards=device_num, shard_id=rank_id)
    dataset = dataset.batch(config.eval_batch_size, drop_remainder=True)
    dataset = dataset.repeat(1)
    print("dataset_len: ", dataset.get_dataset_size() * config.eval_batch_size)
    print("dataset_size: ", dataset.get_dataset_size())
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    ckpt_name = config.ckpt_name
    ckpt_name = ckpt_name.split('.')[0]
    test_score_file = os.path.join(local_test_path, "score_" + ckpt_name + ".test")
    test_result_file = os.path.join(local_test_path, "result_" + ckpt_name + ".test")
    print("test_score_file: ", test_score_file)
    print("test_result_file: ", test_result_file)

    print("************Starting loading model: ", config.model_name)
    dam_net = DAMNet(config, is_emb_init=config.is_emb_init)
    train_net = DAMNetWithLoss(dam_net)
    eval_net = PredictWithNet(dam_net)
    metric = EvalMetric(config.model_name, score_file=test_score_file)
    model = Model(train_net, eval_network=eval_net, metrics={"Accuracy": metric})

    # loading checkpoint
    checkpoint_file = os.path.join(local_ckpt_path, config.ckpt_name)
    print('loading checkpoint: ', checkpoint_file)
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(dam_net, param_dict)

    print("############## Start testing ##############")
    res = model.eval(dataset, dataset_sink_mode=False)
    print(res)

    result = res["Accuracy"]
    with open(test_result_file, 'a+', encoding='utf-8') as file_out:
        file_out.write("checkpoint_file: " + config.ckpt_path + config.ckpt_name + '\n')
        result_str = ""
        for acc in result:
            result_str += str(acc) + '\t'
        file_out.write(result_str + '\n')

    if config.modelArts:
        mox.file.copy_parallel(src_url=local_test_path, dst_url=config.train_url)


if __name__ == '__main__':
    args = conf.parse_args()
    if args.model_name == "DAM_douban":
        args.vocab_size = 172130
        args.channel1_dim = 16
    print("args: ", args)
    evaluate(args)
