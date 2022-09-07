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
"""
network config setting, will be used in train.py and eval.py
"""
import inspect

import yaml
from easydict import EasyDict as ed

from src.model import JasperBlock, JasperDecoderForCTC, JasperEncoder

train_config = ed({


    "TrainingConfig": {
        "epochs": 440,
        "loss_scale": 128.0,
    },

    "DataConfig": {
        "Data_dir": '/data/train_datasets',
        "train_manifest": ['/data/train_datasets/librispeech-train-clean-100-wav.json',
                           '/data/train_datasets/librispeech-train-clean-360-wav.json',
                           '/data/train_datasets/librispeech-train-other-500-wav.json'],
        "mindrecord_format": "/data/jasper_tr{}.md",
        "mindrecord_files": [f"/data/jasper_tr{i}.md" for i in range(8)],
        "batch_size": 64,
        "accumulation_step": 2,
        "labels_path": "labels.json",

        "SpectConfig": {
            "sample_rate": 16000,
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hamming"
        },

        "AugmentationConfig": {
            "speed_volume_perturb": False,
            "spec_augment": False,
            "noise_dir": '',
            "noise_prob": 0.4,
            "noise_min": 0.0,
            "noise_max": 0.5,
        }
    },

    "OptimConfig": {
        "learning_rate": 0.01,
        "learning_anneal": 1.1,
        "weight_decay": 1e-5,
        "momentum": 0.9,
        "eps": 1e-8,
        "betas": (0.9, 0.999),
        "loss_scale": 1024,
        "epsilon": 0.00001
    },

    "CheckpointConfig": {
        "ckpt_file_name_prefix": 'Jasper',
        "ckpt_path": './checkpoint',
        "keep_checkpoint_max": 10
    }
})

eval_config = ed({

    "save_output": 'librispeech_val_output',
    "verbose": True,

    "DataConfig": {

        "Data_dir": '/data/inference_datasets',

        "test_manifest": ['/data/inference_datasets/librispeech-dev-clean-wav.json'],


        "batch_size": 32,
        "labels_path": "labels.json",

        "SpectConfig": {
            "sample_rate": 16000,
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hanning"
        },
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

infer_config = ed({
    "DataConfig": {
        "Data_dir":
        '/home/dataset/LibriSpeech',
        "test_manifest":
        ['/home/dataset/LibriSpeech/librispeech-test-clean-wav.json'],
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
    "batch_size_infer": 1,
    # for preprocess
    "result_path": "./preprocess_Result",
    # for postprocess
    "result_dir": "./result_Files",
    "post_out": "./infer_output.txt"
})


def default_args(klass):
    sig = inspect.signature(klass.__init__)
    return {k: v.default for k, v in sig.parameters.items() if k != 'self'}


def load(fpath):
    if fpath.endswith('.toml'):
        raise ValueError('.toml config format has been changed to .yaml')

    cfg = yaml.safe_load(open(fpath, 'r'))

    # Reload to deep copy shallow copies, which were made with yaml anchors
    yaml.Dumper.ignore_aliases = lambda *args: True
    cfg = yaml.dump(cfg)
    cfg = yaml.safe_load(cfg)
    return cfg


def validate_and_fill(klass, user_conf, ignore_unk=None, optional=None):
    conf = default_args(klass)
    if ignore_unk is None:
        ignore_unk = []
    if optional is None:
        optional = []
    for k, v in user_conf.items():
        conf[k] = v

    # Keep only mandatory or optional-nonempty
    conf = {k: v for k, v in conf.items()
            if k not in optional or v is not inspect.Parameter.empty}

    # Validate
    for k, v in conf.items():
        assert v is not inspect.Parameter.empty, \
            f'Value for {k} not specified for {klass}'
    return conf


def encoder(conf):
    """Validate config for JasperEncoder and subsequent JasperBlocks"""

    # Validate, but don't overwrite with defaults
    for blk in conf['jasper']['encoder']['blocks']:
        validate_and_fill(JasperBlock, blk, optional=['infilters'],
                          ignore_unk=['residual_dense'])

    return validate_and_fill(JasperEncoder, conf['jasper']['encoder'])


def decoder(conf, n_classes):
    deco_kw = {'n_classes': n_classes, **conf['jasper']['decoder']}
    return validate_and_fill(JasperDecoderForCTC, deco_kw)


def add_ctc_blank(sym):
    return sym + ['_']


cfgs = load('./src/jasper10x5dr_speca.yaml')

symbols = add_ctc_blank(cfgs['labels'])
encoder_kw = encoder(cfgs)
decoder_kw = decoder(cfgs, n_classes=len(symbols))
