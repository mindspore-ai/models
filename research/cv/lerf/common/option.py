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

import argparse
import os
import pickle
import shutil
from pathlib import Path


class BaseOptions:
    def __init__(self, debug=False):
        self.initialized = False
        self.debug = debug
        self.is_train = False
        self.opt = None
        self.parser = None

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument(
            "--name",
            type=str,
            default="LeRF",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument("--model", type=str, default="LeRFNet")
        parser.add_argument("--scale", "-r", type=int, default=4)
        parser.add_argument("--nf", type=int, default=64)
        parser.add_argument("--modes", type=str, default="sct")
        parser.add_argument("--modes2", type=str, default="sct")
        parser.add_argument(
            "--stages",
            type=int,
            default=2,
            help="repeat block number in feature extraction stage",
        )
        parser.add_argument(
            "--suppSize",
            type=int,
            default=2,
            help="support patch size for interpolation",
        )
        parser.add_argument(
            "--norm", type=int, default=255, help="max quantization range"
        )
        parser.add_argument(
            "--interval", type=int, default=4, help="N bit uniform sampling"
        )
        parser.add_argument(
            "--sigma", type=int, default=10, help="max sigma for steerable kernel"
        )
        parser.add_argument("--modelRoot", type=str, default="./models/")
        parser.add_argument("--expDir", "-e", type=str, default="")
        parser.add_argument("--load_from_opt_file", action="store_true", default=False)
        parser.add_argument("--debug", default=False, action="store_true")

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        if self.debug:
            opt = parser.parse_args("")
        else:
            opt = parser.parse_args()

        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

    def save_options(self, opt):
        file_name = os.path.join(opt.expDir, "opt")
        with open(file_name + ".txt", "wt") as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ""
                default = self.parser.get_default(k)
                if v != default:
                    comment = "\t[default: %s]" % str(default)
                opt_file.write("{:>25}: {:<30}{}\n".format(str(k), str(v), comment))

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + ".pkl", "rb"))
        return new_opt

    def save_code(self):
        src_dir = "./"
        trg_dir = os.path.join(self.opt.expDir, "code")
        for f in Path(src_dir).rglob("*.py"):
            trg_path = os.path.join(trg_dir, f)
            if ".ipynb" not in os.path.dirname(trg_path):
                # skipping ipynb cache files
                os.makedirs(os.path.dirname(trg_path))
                shutil.copy(os.path.join(src_dir, f), trg_path)

    def parse(self):
        opt = self.gather_options()

        opt.is_train = self.is_train  # train or test

        if opt.expDir == "":
            if opt.is_train and opt.debug:
                opt.name = "debug"
            opt.modelDir = os.path.join(opt.modelRoot, opt.name)

            if not os.path.isdir(opt.modelDir):
                os.mkdir(opt.modelDir)

            count = 1
            while True:
                if os.path.isdir(os.path.join(opt.modelDir, "expr_{}".format(count))):
                    count += 1
                else:
                    break
            opt.expDir = os.path.join(opt.modelDir, "expr_{}".format(count))
            os.mkdir(opt.expDir)
        else:
            if not os.path.isdir(opt.expDir):
                os.makedirs(opt.expDir)
            opt.name = opt.expDir.split("/")[-1]

        if "LUT" in opt.model:
            opt.name += "_lutft"
            opt.lutft = True
        else:
            opt.lutft = False

        opt.modelPath = os.path.join(opt.expDir, "Model.ckpt")

        opt.valoutDir = os.path.join(opt.expDir, "val")
        if not os.path.isdir(opt.valoutDir):
            os.mkdir(opt.valoutDir)
        self.opt = opt

        if opt.is_train:
            self.save_options(opt)
            self.save_code()
            if opt.debug:
                opt.displayStep = 10
                opt.valStep = 50

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--batchSize", type=int, default=32)
        parser.add_argument(
            "--cropSize", type=int, default=48, help="input LR training patch size"
        )
        parser.add_argument("--trainDir", type=str, default="./datasets/DIV2K/")
        parser.add_argument("--valDir", type=str, default="./datasets/Benchmark/")
        parser.add_argument(
            "--startIter",
            type=int,
            default=0,
            help="Set 0 for from scratch, else will load saved params and trains further",
        )
        parser.add_argument(
            "--totalIter",
            type=int,
            default=50000,
            help="Total number of training iterations",
        )
        parser.add_argument(
            "--displayStep",
            type=int,
            default=100,
            help="display info every N iteration",
        )
        parser.add_argument(
            "--valStep", type=int, default=2000, help="validate every N iteration"
        )
        parser.add_argument(
            "--saveStep", type=int, default=2000, help="save models every N iteration"
        )
        parser.add_argument("--lr0", type=float, default=1e-3)
        parser.add_argument("--lr1", type=float, default=1e-4)
        parser.add_argument("--lambda_pixel", type=float, default=1)
        parser.add_argument("--weightDecay", type=float, default=0)
        parser.add_argument("--gpuNum", "-g", type=int, default=1)
        parser.add_argument("--workerNum", "-n", type=int, default=8)
        parser.add_argument("--scaleChange", type=int, default=1000)
        parser.add_argument("--save", default=False, action="store_true")
        self.is_train = True
        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument("--testDir", type=str, default="./datasets/Benchmark")
        parser.add_argument("--resultRoot", type=str, default="../results")
        parser.add_argument(
            "--loadIter", type=int, default=50000, help="validate every N iteration"
        )
        parser.add_argument("--lutName", type=str, default="LUT_ft")

        self.is_train = False
        return parser
