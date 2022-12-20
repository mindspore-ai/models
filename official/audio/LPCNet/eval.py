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

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import mindspore
import mindspore.numpy as mnp
from mindspore import context, load_checkpoint

from src import lpcnet
from src.ulaw import lin2ulaw, ulaw2lin
from cal_metrics import cal_mse

FRAME_SIZE = 160
NB_FEATURES = 36
NB_USED_FEATURES = 20
ORDER = 16

def eval_file(feature_file, out_file, model, enc, dec):
    features = np.fromfile(feature_file, dtype='float32')
    features = np.reshape(features, (-1, NB_FEATURES))
    nb_frames = 1
    feature_chunk_size = features.shape[0]
    pcm_chunk_size = FRAME_SIZE * feature_chunk_size

    features = np.reshape(features, (nb_frames, feature_chunk_size, NB_FEATURES))
    periods = (.1 + 50*features[:, :, 18:19]+100).astype('int32')

    pcm = np.zeros((nb_frames * pcm_chunk_size,))
    fexc = np.zeros((1, 1, 3), dtype='int32')+128
    state1 = mnp.zeros((1, 1, model.rnn_units1))
    state2 = mnp.zeros((1, 1, model.rnn_units2))

    mem = 0.
    coef = 0.85

    with open(out_file, 'wb') as fout:
        skip = ORDER + 1
        for c in range(0, nb_frames):
            cfeat = enc(mindspore.Tensor(features[c:c+1, :, :NB_USED_FEATURES]),
                        mindspore.Tensor(periods[c:c + 1, :, :]))
            for fr in range(0, feature_chunk_size):
                f = c*feature_chunk_size + fr
                a = features[c, fr, NB_FEATURES - ORDER:]
                for i in range(skip, FRAME_SIZE):
                    pred = -sum(a * pcm[f * FRAME_SIZE + i - 1:f * FRAME_SIZE + i - ORDER - 1:-1])
                    fexc[0, 0, 1] = lin2ulaw(pred)

                    p, state1, state2 = dec(mindspore.Tensor(fexc), cfeat[:, fr:fr+1, :], state1, state2)
                    p = p.asnumpy().astype('float64')
                    # lower the temperature for voiced frames to reduce noisiness
                    p *= np.power(p, np.maximum(0, 1.5*features[c, fr, 19] - .5))
                    p = p/(1e-18 + np.sum(p))
                    # cut off the tail of the remaining distribution
                    p = np.maximum(p-0.002, 0).astype('float64')
                    p = p/(1e-8 + np.sum(p))

                    fexc[0, 0, 2] = np.argmax(np.random.multinomial(1, p[0, 0, :], 1))

                    pcm[f * FRAME_SIZE + i] = pred + ulaw2lin(fexc[0, 0, 2])
                    fexc[0, 0, 0] = lin2ulaw(pcm[f * FRAME_SIZE + i])
                    mem = coef*mem + pcm[f * FRAME_SIZE + i]

                    np.array([np.round(mem)], dtype='int16').tofile(fout)
                skip = 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('test_data_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('model_file', type=Path)
    parser.add_argument('--device_id', default=0, type=int)

    args = parser.parse_args()
    tst_dir = args.test_data_path
    out_dir = args.output_path
    model_file = args.model_file
    device_id = args.device_id

    # NOTE: fails without max_call_depth due to RNN
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        max_call_depth=5000, device_id=device_id)

    _model = lpcnet.WithLossLPCNet()
    _model.backbone.to_float(mindspore.float16)
    load_checkpoint(str(model_file), net=_model)
    _model.set_train(False)

    _enc = _model.backbone.encoder
    _dec = _model.backbone.decoder

    for _f in tst_dir.glob('*.f32'):
        _feature_file = tst_dir / (_f.stem + '.f32')
        _out_file = out_dir / (_f.stem + '.pcm')
        eval_file(_feature_file, _out_file, _model, _enc, _dec)

    # Calculate MSE
    cal_mse(tst_dir, out_dir)
