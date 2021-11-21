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
"""Evaluation HAKE."""
import time
import numpy as np

import mindspore
from mindspore.ops import Sort, Mul, Select
from mindspore import context, load_checkpoint, load_param_into_net, Tensor
import mindspore.dataset as ds

from src.HAKE_model import HAKE_GRAPH
from src.config import config
from src.dataset import DataReader, TestDataset, ModeType, BatchType


class hake_eval:
    """
    hake eval
    Args:
        hake_network: base hake model.
        head_dataloader: head negative dataloader. Type: ds.GeneratorDataset
        tail_dataloader: tail negative dataloader. Type: ds.GeneratorDataset
    """

    def __init__(self, hake_network, head_dataloader, tail_dataloader):
        self.hake = hake_network
        self.test_dataset_list = [head_dataloader, tail_dataloader]
        self.argsort = Sort(axis=1, descending=True)

        self.mul = Mul()
        self.select = Select()

    def eval(self):
        """eval hake model"""
        logs = []
        step = 0
        total_steps = sum([dataset.get_dataset_size() for dataset in self.test_dataset_list])
        head_test = True
        for test_dataset in self.test_dataset_list:
            for positive_sample, negative_sample, filter_bias in test_dataset:
                batch_size = positive_sample.shape[0]

                if head_test:
                    score = self.hake.construct_head((positive_sample, negative_sample))
                else:
                    score = self.hake.construct_tail((positive_sample, negative_sample))

                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = self.argsort(score)[1]

                if head_test:
                    positive_arg = positive_sample[:, 0]
                else:
                    positive_arg = positive_sample[:, 2]

                for i in range(batch_size):
                    ranking = (argsort[i, :] == positive_arg[i])
                    index = Tensor(np.arange(ranking.shape[0]), dtype=mindspore.float32)
                    ranking_index = self.mul(ranking, index).sum()

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking_index.asnumpy().item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': ranking,
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % 50 == 0:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('time: {}, Evaluating the model... ({}/{})'.format(current_time, step, total_steps))
                step += 1

            head_test = False  # set to tail negative

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        print(metrics)


def run_hake_test_eval():
    """run hake test eval"""
    data_reader = DataReader(config.data_path)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    test_dataloader_head = ds.GeneratorDataset(
        source=TestDataset(
            data_reader,
            ModeType.TEST,
            BatchType.HEAD_BATCH
        ),
        column_names=["positive_sample", "negative_sample", "filter_bias"],
        num_parallel_workers=8
    )

    test_dataloader_tail = ds.GeneratorDataset(
        source=TestDataset(
            data_reader,
            ModeType.TEST,
            BatchType.TAIL_BATCH
        ),
        column_names=["positive_sample", "negative_sample", "filter_bias"],
        num_parallel_workers=8
    )
    test_dataloader_head = test_dataloader_head.batch(config.test_batch_size)
    test_dataloader_tail = test_dataloader_tail.batch(config.test_batch_size)

    hake = HAKE_GRAPH(num_entity, num_relation, config.hidden_dim, config.gamma,
                      config.modulus_weight, config.phase_weight)
    parameter_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(hake, parameter_dict)

    hake_test_eval = hake_eval(hake, test_dataloader_head, test_dataloader_tail)
    hake_test_eval.eval()


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)
    print(config)
    run_hake_test_eval()
