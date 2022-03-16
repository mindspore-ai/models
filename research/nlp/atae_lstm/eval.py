# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""infer"""
import os

import numpy as np

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Model
from mindspore import set_seed
from mindspore import context, Tensor
from mindspore.common import dtype as mstype

from src.config import config
from src.model import AttentionLstm
from src.atae_for_test import Infer
from src.load_dataset import load_dataset

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device,
                        device_id=0)

    data_menu = config.data_url

    if config.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset_menu')
        data_menu = '/cache/dataset_menu/'

    eval_dataset = os.path.join(data_menu, 'test.mindrecord')
    word_path = os.path.join(data_menu, 'weight.npz')
    ckpt_path = config.eval_ckpt

    dataset = load_dataset(input_files=eval_dataset,
                           batch_size=config.batch_size)

    set_seed(config.rseed)
    r = np.load(word_path)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float32)

    net = AttentionLstm(config, weight, is_train=False)
    net.set_train(False)

    max_acc = 0

    model_path = ckpt_path
    ms_ckpt = load_checkpoint(model_path)
    load_param_into_net(net, ms_ckpt)
    infer = Infer(net, batch_size=1)

    model = Model(infer)

    correct = 0
    count = 0

    for batch in dataset.create_dict_iterator():
        content = batch['content']
        sen_len = batch['sen_len']
        aspect = batch['aspect']
        solution = batch['solution']

        pred = model.predict(content, sen_len, aspect)

        polarity_pred = np.argmax(pred.asnumpy(), axis=1)
        polarity_label = np.argmax(solution.asnumpy(), axis=1)

        correct += (polarity_pred == polarity_label).sum()
        count += len(polarity_label)

    acc = correct / count
    print("\n---accuracy:", acc, "---\n")
