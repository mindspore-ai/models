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

"""
preprocess.
"""
import argparse
import os
import subprocess

import numpy as np
from src.dataset import create_gru_dataset
from model_utils.device_adapter import get_device_num


def _create_tokenized_sentences(input_path, output_path, file, language):
    """
    Create tokenized sentences files.

    Args:
        input_path: input path
        output_path: output path
        file: file name.
        language: text language
    """
    from nltk.tokenize import word_tokenize
    sentence = []
    total_lines = open(os.path.join(input_path, file), "r").read().splitlines()
    for line in total_lines:
        line = line.strip('\r\n ')
        line = line.lower()
        tokenize_sentence = word_tokenize(line, language)
        str_sentence = " ".join(tokenize_sentence)
        sentence.append(str_sentence)
    tokenize_file = os.path.join(output_path, file + ".tok")
    f = open(tokenize_file, "w")
    for line in sentence:
        f.write(line)
        f.write("\n")
    f.close()


def _merge_text(input_path, output_path, file_list, output_file):
    """
    Merge text files together.

    Args:
        input_path: input path
        output_path: output path
        file_list: dataset files list.
        output_file: output file after merge
    """
    output_file = os.path.join(output_path, output_file)
    f_output = open(output_file, "w")
    for file_name in file_list:
        text_path = os.path.join(input_path, file_name) + ".tok"
        f = open(text_path)
        f_output.write(f.read() + "\n")
    f_output.close()


def data_preprocess(config):
    """Preprare data for later generation."""
    from src.preprocess import get_dataset_vocab
    config.mr_data_save_path = os.path.join(config.train_url, 'mr_data')
    config.processed_data_save_path = os.path.join(config.train_url, 'data')
    if os.path.exists(os.path.join(config.mr_data_save_path, config.dataset)):
        return
    if not os.path.exists(config.processed_data_save_path):
        os.mkdir(config.processed_data_save_path)
    if not os.path.exists(config.mr_data_save_path):
        os.mkdir(config.mr_data_save_path)

    # install tokenizer
    # nltk.download('punkt')
    def unzip(zip_file, save_dir):
        import zipfile
        zip_isexist = zipfile.is_zipfile(zip_file)
        if zip_isexist:
            fz = zipfile.ZipFile(zip_file, 'r')
            data_num = len(fz.namelist())
            print("Extract Start...")
            print("unzip file num: {}".format(data_num))
            data_print = int(data_num / 100) if data_num > 100 else 1
            i = 0
            for file in fz.namelist():
                if i % data_print == 0:
                    print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                i += 1
                fz.extract(file, save_dir)
            print("Extract Done.")
        else:
            print("This is not zip.")

    if not os.path.exists(os.path.join(os.environ['HOME'], 'nltk_data')):
        unzip_path = os.path.join(os.environ['HOME'], 'nltk_data', 'tokenizers')
        os.makedirs(unzip_path)
        unzip(os.path.join(config.data_url, 'punkt.zip'), unzip_path)

    # preprocess
    src_file_list = ["train.de", "test.de", "val.de"]
    dst_file_list = ["train.en", "test.en", "val.en"]
    for file in src_file_list:
        _create_tokenized_sentences(config.data_url, config.processed_data_save_path, file, "english")
    for file in dst_file_list:
        _create_tokenized_sentences(config.data_url, config.processed_data_save_path, file, "german")
    src_all_file = "all.de.tok"
    dst_all_file = "all.en.tok"
    _merge_text(config.processed_data_save_path, config.processed_data_save_path, src_file_list, src_all_file)
    _merge_text(config.processed_data_save_path, config.processed_data_save_path, dst_file_list, dst_all_file)
    src_vocab = os.path.join(config.processed_data_save_path, "vocab.de")
    dst_vocab = os.path.join(config.processed_data_save_path, "vocab.en")
    get_dataset_vocab(os.path.join(config.processed_data_save_path, src_all_file), src_vocab)
    get_dataset_vocab(os.path.join(config.processed_data_save_path, dst_all_file), dst_vocab)

    # paste
    cmd = f'paste {config.processed_data_save_path}/train.de.tok \
        {config.processed_data_save_path}/train.en.tok > \
        {config.processed_data_save_path}/train.all'
    os.system(cmd)
    cmd = f'paste {config.processed_data_save_path}/test.de.tok \
        {config.processed_data_save_path}/test.en.tok > \
        {config.processed_data_save_path}/test.all'
    os.system(cmd)

    # create data
    create_data_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "src/create_data.py")
    cmd = ["python", create_data_file, '--num_splits=1',
           f"--input_file={config.processed_data_save_path}/test.all",
           f"--src_vocab_file={config.processed_data_save_path}/vocab.de",
           f"--trg_vocab_file={config.processed_data_save_path}/vocab.en",
           f"--output_file={config.mr_data_save_path}/multi30k_test_mindrecord",
           '--max_seq_length=32', '--bucket=[32]']
    print(f"Start preprocess, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def generate_txt():
    """Generate txt files."""
    def w2txt(file, data):
        with open(file, "w") as f:
            for i in range(data.shape[0]):
                s = ' '.join(str(num) for num in data[i, 0])
                f.write(s+"\n")
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--dataset', type=str, default='multi30k_test_mindrecord_32', help='Dataset name')
    parser.add_argument('--data_url', type=str, default='./data', help='Raw data directory')
    parser.add_argument('--train_url', type=str, default='./cache', help='Output directory')
    parser.add_argument('--result_path', type=str, default='./cache/data', help='Result path')
    args_opt = parser.parse_args()

    # generate database if not exist
    data_preprocess(args_opt)

    # create dataset
    mindrecord_file = os.path.join(args_opt.mr_data_save_path, args_opt.dataset)
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=1, batch_size=1,
                                 dataset_path=mindrecord_file, rank_size=get_device_num(), rank_id=0,
                                 do_shuffle=False, is_training=False)

    if not os.path.exists(os.path.join(args_opt.result_path, args_opt.dataset)):
        os.makedirs(os.path.join(args_opt.result_path, args_opt.dataset))

    source_sents = []
    target_sents = []
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["source_ids"])
        target_sents.append(batch["target_ids"])

    source_sents = np.array(source_sents).astype(np.int32)
    target_sents = np.array(target_sents).astype(np.int32)

    w2txt(os.path.join(args_opt.result_path, args_opt.dataset, "source_ids.txt"), source_sents)
    w2txt(os.path.join(args_opt.result_path, args_opt.dataset, "target_ids.txt"), target_sents)


if __name__ == '__main__':
    generate_txt()
