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

import os.path as osp
import sys
import time
from argparse import ArgumentParser
from collections import Iterable
from importlib import import_module

import numpy as np
from addict import Dict
import mindspore

from src.nms.cpu_nms import cpu_soft_nms


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, k + '.')
        elif isinstance(v, Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print('connot parse key {} of type {}'.format(prefix + k, type(v)))
    return parser


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


class Config:
    """A facility for config and config files.
    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.
    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def fromfile(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if filename.endswith('.py'):
            module_name = osp.basename(filename)[:-3]
            if '.' in module_name:
                raise ValueError('Dots are not allowed in config file path.')
            config_dir = osp.dirname(filename)
            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
        elif filename.endswith(('.yaml', '.json')):
            import mmcv
            cfg_dict = mmcv.load(filename)
        else:
            raise IOError('Only py/yaml/json type are supported now!')
        return Config(cfg_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)
        """
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.from_file(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if filename:
            with open(filename, 'r') as f:
                super(Config, self).__setattr__('_text', f.read())
        else:
            super(Config, self).__setattr__('_text', '')

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)


def image_forward(img, net, priors, detector, transform):
    w, h = img.shape[1], img.shape[0]
    scale = mindspore.Tensor([w, h, w, h], dtype=mindspore.float32)
    x = mindspore.ops.ExpandDims()(transform(img), 0)
    out = net(x)
    boxes, scores = detector.construct(out, priors)
    boxes = (boxes[0] * scale).asnumpy()
    scores = scores[0].asnumpy()
    return boxes, scores


def nms_process(num_classes, image_index, scores, boxes, cfg, min_thresh, all_boxes, max_per_image):
    for cls_index in range(1, num_classes):  # ignore the background (category_id=0)
        inds = np.where(scores[:, cls_index] > min_thresh)[0]
        if inds.size == 0:
            all_boxes[cls_index][image_index] = np.empty([0, 5], dtype=np.float32)
            continue
        cls_bboxes = boxes[inds]
        cls_scores = scores[inds, cls_index]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, None])).astype(np.float32, copy=False)

        keep = cpu_soft_nms(cls_dets, cfg.test_cfg['iou'], method=1)
        keep = keep[:cfg.test_cfg['keep_per_class']]  # keep only the highest boxes
        cls_dets = cls_dets[keep, :]
        all_boxes[cls_index][image_index] = cls_dets

    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[cls_index][image_index][:, -1] for cls_index in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for cls_index in range(1, num_classes):
                keep = np.where(all_boxes[cls_index][image_index][:, -1] >= image_thresh)[0]
                all_boxes[cls_index][image_index] = all_boxes[cls_index][image_index][keep, :]


class Timer:
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multi-threading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            out = self.average_time
        else:
            out = self.diff
        return out

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
