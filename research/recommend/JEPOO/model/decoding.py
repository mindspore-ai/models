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
import numpy as np
import mindspore as ms


def extract_notes(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    onsets = (onsets > onset_threshold).astype(ms.uint8).asnumpy()
    frames = (frames > frame_threshold).astype(ms.uint8).asnumpy()
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1

    pitches = []
    intervals = []
    t_frame, t_pitch = onset_diff.nonzero()
    for nonzero in zip(t_frame, t_pitch):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)


def notes_to_frames(pitches, intervals, shape):
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
