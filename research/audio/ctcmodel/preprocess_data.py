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

"""preprocess data and convert to mindrecord"""

import os
import string
import logging
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
from mindspore.mindrecord import FileWriter
from src.model_utils.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CHARSET = set(string.ascii_lowercase + ' ')
PHONEME_LIST = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
    'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
PHONEME_DIC = {v: k for k, v in enumerate(PHONEME_LIST)}
WORD_DIC = {v: k for k, v in enumerate(string.ascii_lowercase + ' ')}


def read_timit_txt(f):
    '''read text label'''
    f = open(f)
    line = f.readlines()[0].strip().split(' ')
    line = line[2:]
    line = ' '.join(line)
    line = line.replace('.', '').lower()
    line = filter(lambda c: c in CHARSET, line)
    f.close()
    ret = []
    for c in line:
        ret.append(WORD_DIC[c])
    return np.asarray(ret)


def read_timit_phoneme(f):
    '''read phoneme label'''
    f = open(f)
    pho = []
    for line in f:
        line = line.strip().split(' ')[-1]
        pho.append(PHONEME_DIC[line])
    f.close()
    return np.asarray(pho)


def diff_feature(feat, nd=1):
    '''differentiate feature'''
    diff = feat[1:] - feat[:-1]
    feat = feat[1:]
    if nd == 1:
        return np.concatenate((feat, diff), axis=1)
    d2 = diff[1:] - diff[:-1]
    return np.concatenate((feat[1:], diff[1:], d2), axis=1)


def read_files(root_path):
    '''read files'''
    files = os.walk(root_path)
    filelists = []
    for filepath, _, filenames in files:
        for filename in filenames:
            filelists.append(os.path.join(filepath, filename))
    return filelists


def get_feature(f):
    '''extract feature'''
    fs, signal = wavfile.read(f)
    signal = signal.astype('float32')
    feat = mfcc(signal=signal, samplerate=fs, winlen=0.01, winstep=0.005, numcep=13, nfilt=26, lowfreq=0, highfreq=6000,
                preemph=0.95, appendEnergy=False)
    feat = diff_feature(feat, nd=2)
    return feat


class TIMIT_PARSER():
    """
    Parse the dataset,extract the feature by mfcc,convert to mindrecord
    """

    def __init__(self, dirname, output_path, label_type='phoneme'):
        self.dirname = dirname
        assert os.path.isdir(dirname), dirname
        self.filelists = [k for k in read_files(self.dirname)
                          if k.endswith('.wav')]
        assert label_type in ['phoneme', 'letter'], label_type
        self.label_type = label_type
        self.output_path = output_path

    def getdatas(self):
        '''get data'''
        data = []
        for f in self.filelists:
            feat = get_feature(f)
            if self.label_type == 'phoneme':
                label = read_timit_phoneme(f[:-5] + '.PHN')
            elif self.label_type == 'letter':
                label = read_timit_txt(f[:-5] + '.TXT')
            data.append([feat, label])
        return data

    def convert_to_mindrecord(self):
        '''convert to mindrecord'''
        schema_json = {"id": {"type": "int32"},
                       "feature": {"type": "float32", "shape": [-1, 39]},
                       "masks": {"type": "float32", "shape": [-1, 256]},
                       "label": {"type": "int32", "shape": [-1]},
                       "seq_len": {"type": "int32"},
                       }
        data_list = []
        logger.info("write into mindrecord,plaese wait")
        pair = self.getdatas()
        for i, data in enumerate(pair):
            feature = data[0]
            label = data[1]
            feature_padding = np.zeros((config.max_sequence_length, feature.shape[1]), dtype=np.float32)
            feature_padding[:feature.shape[0], :] = feature
            masks = np.zeros((config.max_sequence_length, 2 * config.hidden_size), dtype=np.float32)
            masks[:feature.shape[0], :] = 1
            label_padding = np.full(config.max_label_length, 61, dtype=np.int32)
            label_padding[:label.shape[0]] = label
            data_json = {"id": i,
                         "feature": feature_padding.reshape(-1, config.feature_dim),
                         "masks": masks.reshape(-1, 2 * config.hidden_size),
                         "label": label_padding.reshape(-1),
                         "seq_len": feature.shape[0],
                         }
            data_list.append(data_json)
        writer = FileWriter(self.output_path, shard_num=4)
        writer.add_schema(schema_json, "nlp_schema")
        writer.add_index(["id"])
        writer.write_raw_data(data_list)
        writer.commit()
        logger.info("writing into record suceesfully")


if __name__ == '__main__':
    if not os.path.exists(config.dataset_dir):
        os.makedirs(config.dataset_dir)
    logger.info("Preparing train dataset:")
    train_path = os.path.join(config.dataset_dir, config.train_name)
    parser = TIMIT_PARSER(config.train_dir, train_path)
    parser.convert_to_mindrecord()
    logger.info("Preparing test dataset:")
    test_path = os.path.join(config.dataset_dir, config.test_name)
    parser = TIMIT_PARSER(config.test_dir, test_path)
    parser.convert_to_mindrecord()
