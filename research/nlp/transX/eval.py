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
"""
Evaluate TransE/TransD/TransH/TransR models
"""

from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_evaluation_generator
from src.metric import HitAt10
from src.model_builder import create_model
from src.utils.logging import get_logger

set_seed(1)


def modelarts_pre_process():
    """modelarts pre process function."""


def _prepare_context():
    """Prepare the MindSpore context"""
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    context.set_context(device_id=config.device_id)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """Run evaluation"""
    # Prepare the Context
    _prepare_context()

    config.logger = get_logger(config.eval_output_dir, 0)

    data_generator = create_evaluation_generator(
        dataset_root=config.dataset_root,
        triplet_file_name=config.eval_triplet_file_name,
        entities_file_name=config.entities_file_name,
        relations_file_name=config.relations_file_name,
        triplets_filter_files=config.filter_triplets_files_names,
    )

    # network
    config.logger.important_info('start create network')

    # get network and init
    network = create_model(
        data_generator.entities_number,
        data_generator.relations_number,
        config,
    )

    # Loading weights
    load_param_into_net(network, load_checkpoint(config.ckpt_file))

    network.set_train(False)
    network.set_grad(False)

    # Evaluation
    head_hits = HitAt10()
    tails_hits = HitAt10()

    config.logger.info('start evaluation')

    for batch_data in data_generator:
        head_corrupted_batch = batch_data[0]
        head_corrupted_mask = batch_data[1]
        tail_corrupted_batch = batch_data[2]
        tail_corrupted_mask = batch_data[3]

        head_scores = network(Tensor(head_corrupted_batch)).asnumpy()
        tail_scores = network(Tensor(tail_corrupted_batch)).asnumpy()

        head_hits.update(head_scores, head_scores[-1], head_corrupted_mask)
        tails_hits.update(tail_scores, tail_scores[-1], tail_corrupted_mask)

    results_info = (
        f'Result: hit@10 = {(head_hits.hit10 + tails_hits.hit10) / 2:.4f} '
        f'hit@3 = {(head_hits.hit3 + tails_hits.hit3) / 2:.4f} '
        f'hit@1 = {(head_hits.hit1 + tails_hits.hit1) / 2:.4f}'
    )

    config.logger.info('evaluation finished')
    config.logger.info(results_info)


if __name__ == '__main__':
    run_eval()
