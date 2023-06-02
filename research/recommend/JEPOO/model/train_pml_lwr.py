# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import stat
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.nn import dynamic_lr
from mindspore import ops
from tqdm import tqdm
from evaluate import evaluate
from model import DatasetGenerator, JMPML, focal_loss, cycle, MinNormSolver


def train(args):
    alpha_loss = args.alpha_FL
    gamma_loss = args.gamma_FL

    logdir = args.logdir
    hop_length = args.hop_length
    w_re = args.w_re
    sr = 16000
    max_midi = 108
    min_midi = 21
    n_mels = 229
    mel_fmin = 30
    mel_fmax = sr // 2
    iterations = args.iterations
    checkpoint_interval = args.checkpoint_interval

    batch_size = args.batch_size
    sequence_length = 204800
    model_complexity = 16

    learning_rate_decay_steps = 500
    learning_rate_decay_rate = 0.98
    learning_rate = dynamic_lr.exponential_decay_lr(5e-4, learning_rate_decay_rate, iterations,
                                                    learning_rate_decay_steps, 1)

    os.makedirs(logdir, exist_ok=True)
    train_groups, validation_groups = ['train'], ['test']
    dataset = DatasetGenerator(path=args.path_dataset, hop_length=hop_length, groups=train_groups,
                               sequence_length=sequence_length)
    print(len(dataset))
    validation_dataset = DatasetGenerator(path=args.path_dataset, hop_length=hop_length, groups=validation_groups,
                                          sequence_length=None)
    print(len(validation_dataset))
    loader = ds.GeneratorDataset(dataset, ['audio', 'onset', 'offset', 'frame'], shuffle=True).batch(batch_size)
    loader_val = ds.GeneratorDataset(validation_dataset, ['audio', 'onset', 'offset', 'frame']).create_dict_iterator()

    model = JMPML(n_mels, max_midi - min_midi + 1, sr, hop_length, mel_fmin, mel_fmax, model_complexity)
    model.set_train()
    optimizer = nn.Adam(model.trainable_params(), learning_rate)
    loss_fn = focal_loss
    resume_iteration = 0
    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    weight = ms.Tensor(np.ones([batch_size, sequence_length // hop_length, 88]), dtype=ms.float32)
    re_label = ms.Tensor(np.ones([3]), dtype=ms.float32)

    class WithLossCell(nn.Cell):
        def __init__(self, net, loss):
            super().__init__()
            self.net = net
            self.loss = loss

        def construct(self, inputs, onsets, offsets, frames, task_weights, w_re=None):
            if w_re is None:
                onset_pred, offset_pred, frame_pred = self.net(inputs)
                loss = self.loss(onset_pred, onsets, alpha_loss, gamma_loss, weight) * task_weights[0] + \
                       self.loss(offset_pred, offsets, alpha_loss, gamma_loss, weight) * task_weights[1] + \
                       self.loss(frame_pred, frames, alpha_loss, gamma_loss, weight) * task_weights[2]
            else:
                onset_pred, offset_pred, frame_pred, task_weights = self.net(inputs, task_weights)
                task_weights = task_weights[0]
                loss = self.loss(onset_pred, onsets, alpha_loss, gamma_loss, weight) * task_weights[0] + \
                       self.loss(offset_pred, offsets, alpha_loss, gamma_loss, weight) * task_weights[1] + \
                       self.loss(frame_pred, frames, alpha_loss, gamma_loss, weight) * task_weights[2] + \
                       w_re * ops.reduce_sum((task_weights - re_label) ** 2)
            return loss

    class TrainOneStepCell(nn.Cell):
        def __init__(self, net, optim):
            super(TrainOneStepCell, self).__init__()
            self.net = net
            self.net.set_grad()
            self.grad_op = ops.GradOperation(get_by_list=True)
            self.optimizer = optim
            self.weights = self.optimizer.parameters

        def construct(self, inputs, onsets, offsets, frames, task_weights, w_re, back=True):
            loss = self.net(inputs, onsets, offsets, frames, task_weights, w_re)
            grads = self.grad_op(self.net, self.weights)(inputs, onsets, offsets, frames, task_weights, w_re)
            if back:
                loss = ops.depend(loss, self.optimizer(grads))
                return loss
            return grads

    net_with_loss = WithLossCell(model, loss_fn)
    train_net = TrainOneStepCell(net_with_loss, optimizer)
    frame_f1, note_f1, note_off_f1, it = 0, 0, 0, 0

    for i, batch in zip(loop, cycle(loader.create_dict_iterator())):
        audio_label = batch['audio']
        onset_label = batch['onset']
        print(onset_label.shape, end='\t')
        frame_label = batch['frame']
        offset_label = batch['offset']

        grads = [[], [], []]
        grad_onset = train_net(audio_label, onset_label, offset_label, frame_label, [1, 0, 0], None, False)
        for each_grad in grad_onset:
            grads[0].append(each_grad.copy().asnumpy().flatten())
        grad_offset = train_net(audio_label, onset_label, offset_label, frame_label, [0, 1, 0], None, False)
        for each_grad in grad_offset:
            grads[1].append(each_grad.copy().asnumpy().flatten())
        grad_frame = train_net(audio_label, onset_label, offset_label, frame_label, [0, 0, 1], None, False)
        for each_grad in grad_frame:
            grads[2].append(each_grad.copy().asnumpy().flatten())
        weight_vec, _ = MinNormSolver.find_min_norm_element(grads)
        normalize_coeff = len(grads) / np.sum(np.absolute(weight_vec))

        weight_vec = ops.expand_dims(ms.Tensor(weight_vec * normalize_coeff, dtype=ms.float32), 0)
        loss = train_net(audio_label, onset_label, offset_label, frame_label, weight_vec, w_re, True)
        with os.fdopen(os.open(os.path.join(logdir, 'loss.txt'),
                               os.O_CREAT | os.O_APPEND | os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), 'a') as f:
            # with os.open(os.path.join(logdir, 'loss.txt'), 'a') as f:
            f.write(str(i) + '\t')
            f.write(str(loss) + '\n')

        if i % checkpoint_interval == 0:
            model.set_grad(requires_grad=False)
            model.set_train(False)
            metrics = evaluate(loader_val, model, hop_length, alpha_loss, gamma_loss)
            f_f1 = np.mean(metrics['metric/frame/f1'])
            n_f1 = np.mean(metrics['metric/note/f1'])
            nf_f1 = np.mean(metrics['metric/note-with-offsets/f1'])
            if f_f1 >= frame_f1 or n_f1 >= note_f1 or nf_f1 >= note_off_f1 or i - it > 10000:
                frame_f1, note_f1, note_off_f1, it = f_f1, n_f1, nf_f1, i
                with os.fdopen(os.open(os.path.join(logdir, 'result.txt'),
                                       os.O_CREAT | os.O_APPEND | os.O_WRONLY, stat.S_IWUSR | stat.S_IRUSR), 'a') as f:
                    # with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                    f.write(str(i) + '\t')
                    f.write(str(f_f1) + '\t')
                    f.write(str(n_f1) + '\t')
                    f.write(str(nf_f1) + '\n')
                ms.save_checkpoint(model, os.path.join(logdir, f'model-{i}.ckpt'))
            model.set_train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JEPOO')
    parser.add_argument('--path_dataset', type=str, default="./dataset/MAPS", help='Dataset path')
    parser.add_argument('--alpha_FL', type=int, default=5, help='')
    parser.add_argument('--gamma_FL', type=int, default=0, help='')
    parser.add_argument('--logdir', type=str, default='./runs/MAPS', help='')
    parser.add_argument('--hop_length', type=int, default=512, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='')
    parser.add_argument('--iterations', type=int, default=300000, help='')
    parser.add_argument('--w_re', type=float, default=0.04, help='')
    my_args = parser.parse_args()
    train(my_args)
