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
''' entrance '''
import ast
import argparse
from utils.inference import infer


parser = argparse.ArgumentParser(description='start infer')
parser.add_argument('--img_path', type=str, default='../data_test/')
parser.add_argument('--result_path', type=str, default='result.txt')
parser.add_argument('--pipline_path', type=str, default='pipline/pamtri.pipline')
parser.add_argument('--segmentaware', type=ast.literal_eval, default=True)
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False)
args = parser.parse_args()
if __name__ == '__main__':
    infer(args.img_path, args.result_path, args.pipline_path,
          segmentaware=args.segmentaware, heatmapaware=args.heatmapaware)
