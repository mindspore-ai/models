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
predict
"""
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import RetrievalWithSoftmax
from src.bert import BertConfig
from src.dataset import create_dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config as cfg

set_seed(0)

@moxing_wrapper()
def run_duconv():
    """run duconv task"""

    context.set_context(mode=context.GRAPH_MODE)
    use_kn = bool("kn" in cfg.task_name)
    config = BertConfig(seq_length=cfg.max_seq_length, vocab_size=cfg.vocab_size)
    dataset = create_dataset(cfg.batch_size, data_file_path=cfg.eval_data_file_path,
                             do_shuffle=False, use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()
    print(steps_per_epoch)

    network = RetrievalWithSoftmax(config, use_kn)
    param_dict = load_checkpoint(cfg.load_checkpoint_path)
    not_loaded = load_param_into_net(network, param_dict)
    print(not_loaded)
    network.set_train(False)

    f = open(cfg.save_file_path, 'w')
    iterator = dataset.create_tuple_iterator()
    for item in iterator:
        output = network(*item[:-1])
        for i in output:
            f.write(str(i[1]) + '\n')
            f.flush()
    f.close()

if __name__ == '__main__':
    run_duconv()
