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
import mindspore
from mindspore.communication import get_group_size, get_rank
from mindspore.dataset import GeneratorDataset, Schema

FEATURE_CHUNK_SIZE = 15

class LPCNetLoader:
    """ Class for loading data into model during training """
    def __init__(self, data, features, periods, batch_size):
        self.batch_size = batch_size
        self.nb_batches = np.minimum(np.minimum(data.shape[0], features.shape[0]), periods.shape[0])//self.batch_size
        self.data = data[:self.nb_batches * self.batch_size, :]
        self.features = features[:self.nb_batches * self.batch_size, :]
        self.periods = periods[:self.nb_batches * self.batch_size, :]
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches * self.batch_size)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        data = self.data[self.indices[index * self.batch_size:(index + 1) * self.batch_size], :, :]
        in_data = data[:, :, :3]
        out_data = data[:, :, 3:4]
        features = self.features[self.indices[index * self.batch_size:(index + 1) * self.batch_size], :, :]
        periods = self.periods[self.indices[index * self.batch_size:(index + 1) * self.batch_size], :, :]

        in_data = in_data[:, :FEATURE_CHUNK_SIZE * 160]
        features = features[:, :FEATURE_CHUNK_SIZE + 4]
        periods = periods[:, :FEATURE_CHUNK_SIZE + 4]
        out_data = out_data[:, :FEATURE_CHUNK_SIZE * 160]

        noise = np.random.uniform(0., 0.005, size=(self.batch_size, 160 * FEATURE_CHUNK_SIZE, 384))

        return in_data.astype('int32'), features, periods.astype('int32'), \
               out_data.astype('int32'), noise.astype('float16')

    def __len__(self):
        return self.nb_batches

def make_dataloader(pcm_file, feature_file, frame_size=160, nb_features=36,
                    nb_used_features=20, feature_chunk_size=15, batch_size=64,
                    parallel=False):
    """" Creates LPCNetLoader """
    pcm_chunk_size = frame_size * feature_chunk_size
    data = np.memmap(pcm_file, dtype='uint8', mode='r')
    nb_frames = (len(data)//(4*pcm_chunk_size)-1)//batch_size*batch_size

    features = np.memmap(feature_file, dtype='float32', mode='r')

    data = data[4*2*frame_size:]
    data = data[:nb_frames*4*pcm_chunk_size]

    data = np.reshape(data, (nb_frames, pcm_chunk_size, 4))

    sizeof = features.strides[-1]
    features = np.lib.stride_tricks.as_strided(features, shape=(nb_frames, feature_chunk_size+4, nb_features),
                                               strides=(feature_chunk_size*nb_features*sizeof, nb_features*sizeof,
                                                        sizeof))
    features = features[:, :, :nb_used_features]

    periods = (.1 + 50 * features[:, :, 18:19]+100).astype('int16')

    loader = LPCNetLoader(data, features, periods, batch_size)

    schema = Schema()
    schema.add_column(name='in_data', de_type=mindspore.int32, shape=[batch_size, frame_size * FEATURE_CHUNK_SIZE, 3])
    schema.add_column(name='features', de_type=mindspore.float32, shape=[batch_size, FEATURE_CHUNK_SIZE + 4,
                                                                         nb_used_features])
    schema.add_column(name='periods', de_type=mindspore.int32, shape=[batch_size, FEATURE_CHUNK_SIZE + 4, 1])
    schema.add_column(name='output', de_type=mindspore.int32, shape=[batch_size, frame_size * FEATURE_CHUNK_SIZE, 1])
    schema.add_column(name='noise', de_type=mindspore.float16, shape=[batch_size, frame_size * FEATURE_CHUNK_SIZE, 384])

    if parallel:
        rank_id = get_rank()
        rank_size = get_group_size()

    if parallel:
        dataset = GeneratorDataset(loader, schema=schema, shuffle=True, num_parallel_workers=8,
                                   num_shards=rank_size, shard_id=rank_id)
    else:
        dataset = GeneratorDataset(loader, schema=schema, shuffle=True, num_parallel_workers=8)
    return dataset, len(loader)
