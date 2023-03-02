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

import time
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net, context, save_checkpoint
import numpy as np
from src.dialogXL import DialogXL, ERC_xlnet
from src.config import DialogXLConfig, init_args
from src.data import load_data
from sklearn.metrics import f1_score, accuracy_score

def forward_fn(content_ids, content_mask, content_lengths, speaker_ids,
               mems, speaker_mask, window_mask, label):
    logits, new_mems, new_speaker_mask, new_window_mask = model(content_ids, mems, content_mask, content_lengths,
                                                                speaker_ids, speaker_mask, window_mask)
    loss = loss_fn(logits, label)
    return loss, logits, new_mems, new_speaker_mask, new_window_mask

def train_step(content_ids, content_mask, content_lengths, speaker_ids,
               mems, speaker_mask, window_mask, label):
    (loss, logits, new_mems, new_speaker_mask, new_window_mask), grads = grad_fn(content_ids, content_mask,
                                                                                 content_lengths, speaker_ids, mems,
                                                                                 speaker_mask, window_mask, label)
    grads = ops.clip_by_global_norm(grads, 5.0)
    optimizer(grads)
    return loss, logits, new_mems, new_speaker_mask, new_window_mask

def train_or_eval(datasets, train=False):
    losses, preds, labels, cnt = [], [], [], 0
    begin_time = time.time()
    for dataset in datasets:
        mems = None
        speaker_mask = None
        window_mask = None
        for data in dataset:
            cnt += 1
            content_ids, label, content_mask, content_lengths, speaker_ids = data
            if train:
                loss, logits, mems, speaker_mask, window_mask = train_step(content_ids, content_mask,
                                                                           content_lengths, speaker_ids, mems,
                                                                           speaker_mask, window_mask, label)
            else:
                logits, mems, speaker_mask, window_mask = model(content_ids, mems, content_mask,
                                                                content_lengths, speaker_ids,
                                                                speaker_mask, window_mask)

            label = label.asnumpy()
            pred = logits.argmax(1).asnumpy()
            for l, p in zip(label, pred):
                if l != -1:
                    preds.append(p)
                    labels.append(l)

            if train:
                losses.append(loss.asnumpy())

    total_time = round((time.time() - begin_time) * 1000, 3)
    avg_time = round(total_time / float(cnt), 3)
    preds = np.array(preds)
    labels = np.array(labels)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    if train:
        avg_loss = round(np.sum(losses) / len(losses), 4)
    else:
        avg_loss = None
    return avg_loss, avg_accuracy, avg_fscore, avg_time, total_time

if __name__ == '__main__':
    args = init_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)
    config = DialogXLConfig()
    config.set_args(args)

    dialogXL = DialogXL(config)

    param_dict = load_checkpoint('pretrained/xlnet.ckpt')
    param_not_load, _ = load_param_into_net(dialogXL, param_dict)
    if len(param_not_load) == len(dialogXL.trainable_params()):
        print(param_not_load)
        raise ValueError

    model = ERC_xlnet(args, dialogXL)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr)

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    trainsets = load_data('data/MELD_trainsets.pkl')
    valsets = load_data('data/MELD_valsets.pkl')

    best_fscore = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f'********************** epoch {epoch} **********************')
        train_loss, train_acc, train_f1, train_avg_time, train_time = train_or_eval(trainsets, train=True)
        print(f'train_loss: {train_loss}, train_acc: {train_acc}, train_fscore: {train_f1}.')
        print(f'Train epoch time: {train_time} ms, per step time: {train_avg_time} ms.')

        _, eval_acc, eval_f1, eval_avg_time, eval_time = train_or_eval(valsets, train=False)
        print(f'eval_acc: {eval_acc}, eval_fscore: {eval_f1}.')
        print(f'Eval epoch time: {eval_time} ms, per step time: {eval_avg_time} ms.')
        if eval_f1 > best_fscore:
            best_fscore = eval_f1
            save_checkpoint(model, ckpt_file_name='checkpoints/MELD_best.ckpt')
            print('Best model saved!')
    print('********************** End Training **********************')
