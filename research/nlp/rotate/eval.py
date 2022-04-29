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
""" Link Prediction Evaluation """
import os
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.rotate import ModelBuilder
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_dataset


class KGEModel(nn.Cell):
    """
    Generate sorted candidate entity id and positive sample.

    Args:
        network (nn.Cell): Trained model with entity embedding and relation embedding.
        mode (str): which negative sample mode ('head-mode' or 'tail-mode').

    Returns:
        argsort: entity id sorted by score
        positive_arg: positive sample entity id

    """
    def __init__(self, network, mode='head-mode'):
        super(KGEModel, self).__init__()
        self.network = network
        self.construct_head = self.network.construct_head
        self.construct_tail = self.network.construct_tail
        self.mode = mode
        self.sort = P.Sort(axis=1, descending=True)

    def construct(self, positive_sample, negative_sample, filter_bias):
        """ Sort candidate entity id and positive sample entity id. """
        if self.mode == 'head-mode':
            score = self.construct_head((positive_sample, negative_sample))
        else:
            score = self.construct_tail((positive_sample, negative_sample))
        score += filter_bias
        _, argsort = self.sort(score)
        return argsort


class EvalKGEMetric(nn.Cell):
    """
    Calculate metrics.

    Args:
        network (nn.Cell): Trained model with entity embedding and relation embedding.
        mode (str): which negative sample mode ('head-mode' or 'tail-mode').

    Returns:
        log (list): contain metrics of each triple

    """
    def __init__(self, network, mode='head-mode'):
        super(EvalKGEMetric, self).__init__()
        self.mode = mode
        self.kgemodel = KGEModel(network=network, mode=self.mode)

    def construct(self, positive_sample, negative_sample, filter_bias):
        """ Calculate metrics. """
        argsort = self.kgemodel(positive_sample, negative_sample, filter_bias)
        if self.mode == 'head-mode':
            positive_arg = positive_sample[:, 0]
        else:
            positive_arg = positive_sample[:, 2]
        return argsort, positive_arg


def modelarts_process():
    pass


def generate_log(argsort, positive_arg, batch_size):
    argsort, positive_arg = argsort.asnumpy(), positive_arg.asnumpy()
    log = []
    for i in range(batch_size):
        ranking = np.where(argsort[i, :] == positive_arg[i])[0][0]
        ranking = 1 + ranking
        log.append({
            'MRR': 1.0 / ranking,
            'MR': ranking,
            'HITS@1': 1.0 if ranking <= 1 else 0.0,
            'HITS@3': 1.0 if ranking <= 3 else 0.0,
            'HITS@10': 1.0 if ranking <= 10 else 0.0,
        })
    return log


@moxing_wrapper(pre_process=modelarts_process)
def eval_kge():
    """ Link Prediction Task for Knowledge Graph Embedding Model """
    if config.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.getenv('DEVICE_ID')))
    elif config.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    else:
        print("Unsupported device_target ", config.device_target)
        exit()

    num_entity, num_relation, test_dataloader_head, test_dataloader_tail = create_dataset(
        data_path=config.data_path,
        config=config,
        is_train=False
    )
    config.num_entity, config.num_relation = num_entity, num_relation

    model_builder = ModelBuilder(config)
    eval_net = model_builder.get_eval_net()

    model_params = load_checkpoint(ckpt_file_name=config.eval_checkpoint)
    load_param_into_net(net=eval_net, parameter_dict=model_params)

    logs = []
    eval_model_head = EvalKGEMetric(network=eval_net, mode='head-mode')
    eval_model_tail = EvalKGEMetric(network=eval_net, mode='tail-mode')

    for test_data in test_dataloader_head.create_dict_iterator():
        argsort, positive_arg = eval_model_head.construct(test_data["positive"], test_data["negative"],
                                                          test_data["filter_bias"])
        batch_size = test_data["positive"].shape[0]
        log_head = generate_log(argsort, positive_arg, batch_size)
        logs += log_head
    for test_data in test_dataloader_tail.create_dict_iterator():
        argsort, positive_arg = eval_model_tail.construct(test_data["positive"], test_data["negative"],
                                                          test_data["filter_bias"])
        batch_size = test_data["positive"].shape[0]
        log_tail = generate_log(argsort, positive_arg, batch_size)
        logs += log_tail

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    print(metrics)


if __name__ == '__main__':
    eval_kge()
