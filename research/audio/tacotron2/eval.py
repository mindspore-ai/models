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
"""evaluate."""

import time
import argparse
import os
from os.path import join

import matplotlib
import matplotlib.pylab as plt
import numpy as np

import mindspore
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor

from src.utils.audio import save_wav, inv_melspectrogram
from src.tacotron2 import Tacotron2
from src.hparams import hparams as hps
from src.text import text_to_sequence




matplotlib.use('Agg')

def load_model(ckpt_pth):
    '''
    load model
    '''
    net = Tacotron2()
    param_dict = load_checkpoint(ckpt_pth)

    load_param_into_net(net, param_dict)
    net.set_train(False)
    net.decoder.prenet.dropout.set_train(True)

    return net.to_float(mindspore.float32)


def infer(text, net):
    '''
    inference
    '''
    sequence = text_to_sequence(text, hps.text_cleaners)
    sequence = Tensor(sequence, mindspore.int32).view(1, -1)
    text_mask = Tensor(np.zeros(sequence.shape).astype('bool'))

    mel_outputs, mel_outputs_postnet, _, alignments = net.inference(
        sequence, text_mask)

    return (mel_outputs, mel_outputs_postnet, alignments)


def plot_data(data, figsize=(16, 4)):
    '''
    plot alignments
    '''
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for _, i in enumerate(range(len(data))):
        axes[i].imshow(data[i], aspect='auto', origin='lower')



def plot(output, dir_pth, filename):
    '''
    plot alignments
    '''
    mel_outputs, mel_outputs_postnet, alignments = output
    plot_data((mel_outputs.asnumpy()[0],
               mel_outputs_postnet.asnumpy()[0],
               alignments.asnumpy()[0].T))
    plt.savefig(join(dir_pth, filename + '.png'))


def audio(output, dir_pth, filename):
    '''
    save waveform
    '''
    _, mel_outputs_postnet, _ = output

    wav = inv_melspectrogram(mel_outputs_postnet.asnumpy()[0])
    np.save(join(dir_pth, filename + '-wave.npy'), wav, allow_pickle=False)
    save_wav(wav, join(dir_pth, filename + '.wav'))


def save_mel(output, dir_pth, filename):
    '''
    save mel spectrogram
    '''
    _, mel_outputs_postnet, _ = output
    np.save(
        join(
            dir_pth,
            filename +
            '-feats.npy'),
        mel_outputs_postnet.asnumpy()[0].T,
        allow_pickle=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, default='',
                        required=True, help='path to load checkpoints')
    parser.add_argument('-o', '--out_dir', type=str, default='output',
                        help='dirs to save')
    parser.add_argument('-n', '--fname', type=str, default='out',
                        help='fname to save')
    parser.add_argument('-d', '--device_id', type=int, default='0',
                        help='device id')
    parser.add_argument(
        '-t',
        '--text',
        type=str,
        default='',
        help='text to synthesize')

    args = parser.parse_args()
    context.set_context(
        mode=0,
        save_graphs=False,
        device_target="Ascend",
        device_id=int(
            args.device_id))
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.ckpt_pth)
    start = time.time()
    outputs = infer(args.text, model)
    end = time.time()
    print('inference elapsed :{}s'.format(end - start))

    plot(outputs, args.out_dir, args.fname)
    audio(outputs, args.out_dir, args.fname)
    save_mel(outputs, args.out_dir, args.fname)
