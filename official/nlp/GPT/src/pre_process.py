# Copyright 2020 Huawei Technologies Co., Ltd
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
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import glob
import json
import os
import re
import time
from multiprocessing import current_process, Process
import numpy as np

try:
    from transformers import GPT2Tokenizer
except ModuleNotFoundError:
    print("module 'transformers' not installed.")

from mindspore.mindrecord import FileWriter


EOT = 50256  # id of endoftext
SEQ_LEN = 1025  # the length of sample
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def package_file(it, n):
    """ package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch


def clean_wikitext(string):
    """ cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" "+chr(176)+" ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def tokenize_openwebtext(iterator):
    """ tokenize openwebtext dataset"""
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for para in f.read().split("\n\n"):
                if para:
                    tokenized_text = tokenizer.tokenize(para)
                    content += tokenizer.convert_tokens_to_ids(tokenized_text) + [
                        EOT]
        for chunk in chunks(content, SEQ_LEN):
            sample = {}
            if len(chunk) == SEQ_LEN:
                sample['input_ids'] = np.array(chunk, dtype=np.int32)
                yield sample


def tokenize_wiki(file_path):
    """tokenize wikitext-2/wikitext-103 dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for para in clean_wikitext(f.read()).split("\n\n"):
            if para and para.strip().startswith('=') is False:
                tokenized_text = tokenizer.tokenize(para)
                content += tokenizer.convert_tokens_to_ids(tokenized_text) + [
                    EOT]
    for chunk in chunks(content, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def tokenize_lambada(file_path):
    """tokenize lambada dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            para = json.loads(line)['text'].replace(
                "“", '"').replace("”", '"').strip().strip(".")
            tokenized_text = tokenizer.tokenize(para)
            content += tokenizer.convert_tokens_to_ids(tokenized_text) + [EOT]
    for chunk in chunks(content, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def task_unit(iterator, mindrecord_filename, schema):
    writer = FileWriter(file_name=mindrecord_filename, shard_num=1)
    writer.add_schema(schema, args.dataset_type)

    p = current_process()
    index = p.pid if p.pid else 0

    item_iter = tokenize_openwebtext(iterator)
    batch_size = 1024  # size of write batch
    count = 0
    while True:
        data_batch = []
        try:
            for _ in range(batch_size):
                data_batch.append(next(item_iter))
                count += 1
            writer.write_raw_data(data_batch)
            print("Process {} transformed {} records.".format(
                index, count))
        except StopIteration:
            if data_batch:
                writer.write_raw_data(data_batch)
                print("Process {} transformed {} records.".format(
                    index, count))
            break
    writer.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='openwebtext')
    parser.add_argument('--input_glob', type=str, default='*.txt')
    parser.add_argument('--output_file', type=str,
                        default='./output/openweb_mindrecord')
    parser.add_argument('--file_partition', type=int, default=1)
    parser.add_argument('--file_batch_size', type=int, default=1024)
    parser.add_argument('--num_process', type=int, default=64)

    args = parser.parse_args()
    ###
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mindrecord_schema = {"input_ids": {"type": "int32", "shape": [-1]},}
    ###
    transforms_count = 0
    if args.dataset_type == 'wiki':
        wiki_writer = FileWriter(file_name=args.output_file, shard_num=args.file_partition)
        wiki_writer.add_schema(mindrecord_schema, args.dataset_type)
        for x in tokenize_wiki(args.input_glob):
            transforms_count += 1
            wiki_writer.write_raw_data([x])
        wiki_writer.commit()
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'lambada':
        lambada_writer = FileWriter(file_name=args.output_file, shard_num=args.file_partition)
        lambada_writer.add_schema(mindrecord_schema, args.dataset_type)
        for x in tokenize_lambada(args.input_glob):
            transforms_count += 1
            lambada_writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
        lambada_writer.commit()
    elif args.dataset_type == 'openwebtext':
        SUFFIX = len(str(args.file_partition - 1))
        file_names = ["{}{}".format(args.output_file, str(x).rjust(SUFFIX, '0'))
                      for x in range(args.file_partition)]
        file_iter = glob.iglob(args.input_glob)
        process_list = {}
        for file in file_names:
            p1 = Process(target=task_unit, args=(file_iter, file, mindrecord_schema))
            p1.start()
            process_list[file] = p1
        for process in process_list.values():
            while process.is_alive(): # wait child process exit
                time.sleep(0.01)
    else:
        raise ValueError(
            "Not support dataset type: {}".format(args.dataset_type))

    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
