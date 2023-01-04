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
# =======================================================================================
""" argparse entrance module """
import argparse


def creater_parser():
    parser = argparse.ArgumentParser(description="servering yolox", add_help=False)
    parser.add_argument("--servable_dir", type=str, default="./", help="servable config dir")
    parser.add_argument("--servable_name", type=str, default="yolox", help="servable name")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="Serving Port")
    parser.add_argument("--port", type=str, default="8000", help="Serving Client")
    parser.add_argument("--infer_img", type=str, default="", help="Client infer image path")
    parser.add_argument("--device_id", type=int, default=0, help="npu device id")
    parser.add_argument("--nms_thre", type=float, default=0.6, help="nms thre")
    parser.add_argument("--conf_thre", type=float, default=0.3, help="score thre")
    parser.add_argument("--num_classes", type=int, default=80, help="num_classes")
    parser.add_argument("--input_size", type=list, default=[640, 640], help="input size")
    args, _ = parser.parse_known_args()
    return args


config = creater_parser()
