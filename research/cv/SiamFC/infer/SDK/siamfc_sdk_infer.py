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
"""set sdk infer tracker"""
from __future__ import absolute_import
import argparse
import os


from got10k.experiments import ExperimentOTB
from siamfc_tracker import SiamFCSDKTracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='siamfc tracking')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of GPU or Ascend')
    parser.add_argument(
        '--pipeline1', default='./SDK/pipeline/siamfc_get_exemplar.pipeline',
        type=str, help='get exemplar pipeline')
    parser.add_argument(
        '--pipeline2', default='./SDK/pipeline/siamfc_infer.pipeline',
        type=str, help='infer pipeline')
    parser.add_argument(
        '--dataset_path', default='OTB2013', type=str)
    args = parser.parse_args()
    tracker = SiamFCSDKTracker(
        pipeline1=args.pipeline1, pipeline2=args.pipeline2)
    root_dir = os.path.abspath(args.dataset_path)
    e = ExperimentOTB(root_dir, version=2013)
    e.run(tracker, visualize=False)
    prec_score = e.report(['SiamFC'])['SiamFC']['overall']
    score = ['success_score', 'precision_score', 'success_rate']
    mydic = []
    for key in score:
        mydic.append(prec_score[key])
    ss = '-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(mydic[1]),
                                                                float(
                                                                    mydic[0]),
                                                                float(mydic[2]))
    print(ss)
