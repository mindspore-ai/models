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
"""Calculate cpp infer result."""
from src.model_utils.config import config
from src.utils.eval_utils import Evaluate310


def main():
    evaluater = Evaluate310(config=config)
    metrics, fps = evaluater.eval(show_imgs=config.eval.show_images)
    print(f"FPS: {fps}\n"
          f"Recall: {metrics['recall'].avg}\n"
          f"Precision: {metrics['precision'].avg}\n"
          f"Fmeasure: {metrics['fmeasure'].avg}\n")

    print("finished")

if __name__ == '__main__':
    main()
