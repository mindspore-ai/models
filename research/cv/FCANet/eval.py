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
""" eval fcanet """
import argparse
from src.config import config
from src.trainer import Trainer

if __name__ == "__main__":
    # set resume path
    parser = argparse.ArgumentParser(description="Argparse for FCANet-Eval")
    parser.add_argument("-r", "--resume", type=str, default="./fcanet_pretrained.pth")
    args = parser.parse_args()

    # set config
    p = config
    p["resume"] = args.resume

    # set trainer
    mine = Trainer(p)

    # eval
    mine.validation_robot(0, tsh=p["pred_tsh"], resize=p["size"][0])
