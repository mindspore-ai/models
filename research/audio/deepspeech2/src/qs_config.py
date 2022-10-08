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
# ===========================================================================

from easydict import EasyDict as ed


quickstart_config = ed({
    "DataConfig": {
        "test_manifest": './quickstart/qs.csv',
        "batch_size": 20,
        "labels_path": "labels.json",

        "SpectConfig": {
            "sample_rate": 16000,
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hanning"
        },
    },

    "ModelConfig": {
        "rnn_type": "LSTM",
        "hidden_size": 1024,
        "hidden_layers": 5,
        "lookahead_context": 20,
    },

    "LMConfig": {
        "decoder_type": "greedy",
        "lm_path": './3-gram.pruned.3e-7.arpa',
        "top_paths": 1,
        "alpha": 1.818182,
        "beta": 0,
        "cutoff_top_n": 40,
        "cutoff_prob": 1.0,
        "beam_width": 1024,
        "lm_workers": 4
    },

})
