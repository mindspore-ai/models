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

# Part of the file was copied from project taesungp NVlabs/SPADE https://github.com/NVlabs/SPADE
""" base options """

import ast
import sys
import argparse
import os
import pickle
from src.util import util
import src.data as data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='ade20k_pretrained', help='')
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--distribute', default=True, type=ast.literal_eval)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        # we use batchnorm instead of syncbatchnorm
        parser.add_argument('--norm_G', type=str, default='spectralspadebatch3x3', help='')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="")
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='resize_and_crop', \
                            help='scaling and cropping of images at load time.', \
                            choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", \
                                     "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=286, \
                            help='Scale images to this size. \
                            The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256, \
                            help='Crop to the width of crop_size (after initially scaling \
                            the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, \
                            help='The ratio width/height. The final height of \
                            the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=150, \
                            help='# of input label classes without unknown class. \
                            If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', default=True, \
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='./datasets/ADEChallengeData2016/')
        parser.add_argument('--dataset_mode', type=str, default='ade20k')
        parser.add_argument('--serial_batches', action='store_true', help='if true, \
        takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, \
        do not flip the images for data argumentation')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, \
                            help='Maximum number of samples allowed per dataset. \
                            If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', \
                            help='load the options from checkpoints and use that as default')
        parser.add_argument('--display_winsize', type=int, default=256, \
                            help='display window size')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--no_instance', action='store_true', default=True, \
                            help='if specified, do *not* add instance map as input')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)

        self.opt = opt
        return self.opt
