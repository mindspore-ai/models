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
"""config file"""
import os


class Config:
    """hyperparameter configuration"""

    def __init__(self, datasetdir, outputdir, device):
        self.model_name = 'HyperText'
        if datasetdir:
            self.train_path = os.path.join(datasetdir, 'train.txt')
            self.dev_path = os.path.join(datasetdir, 'dev.txt')
            self.test_path = os.path.join(datasetdir, 'test.txt')
            self.vocab_path = os.path.join(datasetdir, 'vocab.txt')
            self.labels_path = os.path.join(datasetdir, 'labels.txt')
        self.class_list = []
        if outputdir:
            self.save_path = os.path.join(outputdir, self.model_name + '.ckpt')
            self.log_path = os.path.join(outputdir, self.model_name + '.log')
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
        self.device = device
        self.dropout = 0.5
        self.outputdir = outputdir
        self.num_classes = len(self.class_list)  # label number
        self.n_vocab = 0
        self.num_epochs = 30
        self.wordNgrams = 2

        self.batch_size = 32
        self.max_length = 1000
        self.learning_rate = 1e-2
        self.bucket = 20000  # word and ngram vocab size
        self.lr_decay_rate = 0.96

    def useTnews(self):
        """use tnew config"""
        self.dropout = 0.0
        self.num_classes = len(self.class_list)  # label number
        self.n_vocab = 0
        self.wordNgrams = 2
        self.datasetType = 'tnews'
        self.max_length = 40
        self.embed = 20
        self.eval_step = 100
        self.min_freq = 1
        self.learning_rate = 0.011
        self.bucket = 1500000  # word and ngram vocab size
        self.lr_decay_rate = 0.96

    def useIflyek(self):
        """use iflytek config"""
        self.dropout = 0.0
        self.datasetType = 'iflytek'
        self.num_classes = len(self.class_list)  # label number
        self.n_vocab = 0
        self.wordNgrams = 2
        self.max_length = 1000
        self.embed = 80
        self.eval_step = 50
        self.min_freq = 1
        self.learning_rate = 0.013
        self.bucket = 2000000  # word and ngram vocab size
        self.lr_decay_rate = 0.94
