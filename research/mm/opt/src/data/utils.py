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
"""utils"""
import numpy as np


def pad_attention_mask(input_ids, img_feat, audio_feat, max_lens=30):
    """pad_attention_mask"""
    txt_len = input_ids[-1]
    img_len = img_feat[1]
    audio_len = audio_feat[1]
    attn_masks = np.zeros(max_lens * 3, np.int64)
    attn_masks[:txt_len] = 1
    attn_masks[max_lens:max_lens + img_len] = 1
    attn_masks[max_lens * 2:max_lens * 2 + audio_len] = 1
    return attn_masks


def pad_sequence(sequences, batch_first=True, padding_value=0.0, max_lens=30):
    """pad_sequence"""
    lens = [len(x) for x in sequences]
    if max_lens == -1:
        max_lens = max(lens)

    padded_seq = []
    for x in sequences:
        pad_width = [(0, max_lens - len(x))]
        padded_seq.append(np.pad(x, pad_width, constant_values=(padding_value, padding_value)))

    sequences = np.stack(padded_seq, axis=0 if batch_first else 1)
    return sequences


def pad_sequence_(sequences, batch_first=False, padding_value=0.0, max_lens=30):
    """pad_sequence"""
    if sequences[0] is None:
        return None
    return pad_sequence(sequences, batch_first, padding_value, max_lens)


def masked_fill(x, mask, value):
    """masked_fill"""
    mask = np.broadcast_to(mask, x.shape)
    if mask.dtype != np.bool_:
        mask = mask == 1.0
    y = x.copy()
    y[mask] = value
    return y


if __name__ == "__main__":
    mask_test = np.array([[True], [False]])
    print(mask_test)
    x_test = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(masked_fill(x_test, mask_test, 0))
    print(x_test)
