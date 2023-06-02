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
import sys
from collections import defaultdict

import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import nn
from tqdm import tqdm

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
from mir_eval.melody import raw_chroma_accuracy, raw_pitch_accuracy, to_cent_voicing
from scipy.stats import hmean

from model import extract_notes, notes_to_frames, save_pianoroll, save_midi

eps = sys.float_info.epsilon


def evaluate(data, model, hop_length, alpha, gamma, sample_rate=16000, min_midi=21,
             onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    metrics = defaultdict(list)

    for label in tqdm(data):
        audio_label = label['audio']
        onset_label = label['onset']
        frame_label = label['frame']
        offset_label = label['offset']
        onset_pred = ms.Tensor(np.zeros(onset_label.shape, dtype=np.float32))
        offset_pred = ms.Tensor(np.zeros(offset_label.shape, dtype=np.float32))
        frame_pred = ms.Tensor(np.zeros(frame_label.shape, dtype=np.float32))
        n_steps = 0
        for start_time in range(0, len(audio_label), 204800):
            audio_t = audio_label[start_time: start_time+204800]
            onset_pred_t, offset_pred_t, frame_pred_t = model(audio_t)
            onset_pred[n_steps: n_steps+400] = onset_pred_t[0]
            offset_pred[n_steps: n_steps+400] = offset_pred_t[0]
            frame_pred[n_steps: n_steps+400] = frame_pred_t[0]
            n_steps += 400

        pred = {
            'onset': ops.reshape(onset_pred, onset_label.shape),
            'frame': ops.reshape(frame_pred, frame_label.shape),
            'offset': ops.reshape(offset_pred, offset_label.shape)
        }

        loss_fn = nn.BCELoss(weight=None, reduction='mean')
        try:
            losses = {
                'loss/onset': loss_fn(pred['onset'], onset_label),
                'loss/frame': loss_fn(pred['frame'], frame_label),
                'loss/offset': loss_fn(pred['offset'], offset_label),
            }
        except KeyError:
            losses = {
                'loss/onset': 0,
                'loss/frame': 0,
                'loss/offset': 0,
            }

        for key, loss in losses.items():
            metrics[key].append(loss)

        try:
            p_ref, i_ref = extract_notes(label['onset'], label['frame'])
            p_est, i_est = extract_notes(pred['onset'], pred['frame'], onset_threshold, frame_threshold)

            t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
            t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)
        except KeyError:
            p_ref, i_ref = 0, 0
            p_est, i_est = 0, 0

            t_ref, f_ref = 0, 0
            t_est, f_est = 0, 0

        scaling = hop_length / sample_rate

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(min_midi + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(min_midi + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(min_midi + midi) for midi in freqs]) for freqs in f_ref]

        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(min_midi + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps,
                                                 frame_metrics['Recall'] + eps]) - eps)

        f_ref = np.array([freqs[0] if freqs else 0 for freqs in f_ref])

        f_est = np.array([freqs[0] if freqs else 0 for freqs in f_est])

        v_ref, c_ref, v_est, c_est = to_cent_voicing(t_ref, f_ref, t_est, f_est)

        rca = raw_chroma_accuracy(v_ref, c_ref, v_est, c_est)
        rpa = raw_pitch_accuracy(v_ref, c_ref, v_est, c_est)
        metrics['metric/frame/RCA'].append(rca)
        metrics['metric/frame/RPA'].append(rpa)

        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            try:
                label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
                save_pianoroll(label_path, label['onset'], label['frame'])
                pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
                save_pianoroll(pred_path, pred['onset'], pred['frame'])
                midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
                save_midi(midi_path, p_est, i_est, v_est)
            except KeyError:
                print('0')
    return metrics
