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

"""ctc evaluation"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from src.dataset import MsAudioDataset
import jiwer

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument('--chars_dict_path', type=str, default='', help='your/path/dataset/lang_1char/train_chars.txt')
args = parser.parse_args()

def run_eval():
    '''eval_function'''
    path1 = "./result_Files"
    path2 = "./preprocess_Result/04_data"
    chars_dict_path = args.chars_dict_path
    char_list, _, _ = MsAudioDataset.process_dict(chars_dict_path)
    file_name1 = os.listdir(path1)
    file_name2 = os.listdir(path2)
    size = len(file_name2)
    label_dict = dict()
    sample_num = 0
    for i in range(size):
        out_f_name = os.path.join(path1, file_name1[i])
        out = np.fromfile(out_f_name, np.int32)
        out_tokens = []
        for x in out.tolist():
            if x == 0:
                break
            out_tokens.append(char_list[x])
        out_tokens.append(char_list[2])
        out = " ".join(out_tokens)
        str1 = file_name1[i]
        file_name2_temp = str1[:len(str1)-6]+".bin"
        label_f_name = os.path.join(path2, file_name2_temp)
        labels = np.fromfile(label_f_name, np.int32)
        gt_tokens = [char_list[x] for x in labels.tolist() if x != -1]
        gt = " ".join(gt_tokens)
        label_dict[sample_num] = {'output': out, 'gt': gt,}
        sample_num += 1
    with open('./preprocess_Result/labels_dict.json', 'w') as file:
        json.dump(label_dict, file, indent=2)

    remove_non_words = jiwer.RemoveKaldiNonWords()
    remove_space = jiwer.RemoveWhiteSpace()
    preprocessing = jiwer.Compose([remove_non_words, remove_space])

    with Path('./preprocess_Result/labels_dict.json').open('r') as file:
        output_data = json.load(file)

    total_cer = 0
    for sample in output_data.values():
        res_text = preprocessing(sample['output'])
        res_text = ' '.join(res_text)
        gt_text = preprocessing(sample['gt'])
        gt_text = ' '.join(gt_text)
        cer = jiwer.wer(gt_text, res_text)
        total_cer += cer

    print('Resulting cer is ', (total_cer / len(output_data.values())) * 100)


if __name__ == "__main__":
    run_eval()
