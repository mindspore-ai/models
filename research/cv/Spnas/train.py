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
"""train"""
import argparse
import vega


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Spnas network")
    parser.add_argument("--config_path", type=str, required=True, help="spnas config path.")
    args = parser.parse_args()

    config_path = args.config_path
    vega.set_backend('mindspore', 'NPU')
    vega.run(config_path)
