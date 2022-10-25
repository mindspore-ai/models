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

import numpy as np
from src.dialogXL import DialogXL, ERC_xlnet
from src.config import DialogXLConfig, init_args
from src.data import load_data
from mindspore import load_checkpoint, load_param_into_net, context
from sklearn.metrics import f1_score, accuracy_score

if __name__ == '__main__':
    args = init_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)
    config = DialogXLConfig()
    config.set_args(args)

    dialogXL = DialogXL(config)
    model = ERC_xlnet(args, dialogXL)
    param_dict = load_checkpoint('checkpoints/MELD_best.ckpt')
    load_param_into_net(model, param_dict)

    valsets = load_data('data/MELD_testsets.pkl')

    preds, labels = [], []
    print('********************** Evaluating **********************')
    for dataset in valsets:
        mems = None
        speaker_mask = None
        window_mask = None
        for data in dataset:
            content_ids, label, content_mask, content_lengths, speaker_ids = data
            logits, mems, speaker_mask, window_mask = model(content_ids, mems, content_mask, content_lengths,
                                                            speaker_ids, speaker_mask, window_mask)

            label = label.asnumpy()
            pred = logits.argmax(1).asnumpy()

            for l, p in zip(label, pred):
                if l != -1:
                    preds.append(p)
                    labels.append(l)

    preds = np.array(preds)
    labels = np.array(labels)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    print(f'test_acc: {avg_accuracy}, test_fscore: {avg_fscore}')
    print('********************** End Evaluation **********************')
