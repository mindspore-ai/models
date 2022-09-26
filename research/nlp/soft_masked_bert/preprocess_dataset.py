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
import json
import gc
import os
import random
import opencc
from lxml import etree
from tqdm import tqdm


def get_main_dir():
    return os.path.join(os.path.dirname(__file__))

def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(get_main_dir(), fn))

def dump_json(obj, fp):
    fp = os.path.abspath(fp)
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))
    with open(fp, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, separators=(',', ':'))
    print(f'file is saved successfully, {fp}')
    return True


def proc_item(item, converter):
    root = etree.XML(item)
    passages = dict()
    mistakes = []
    for passage in root.xpath('/ESSAY/TEXT/PASSAGE'):
        passages[passage.get('id')] = converter.convert(passage.text)
    for mistake in root.xpath('/ESSAY/MISTAKE'):
        mistakes.append({'id': mistake.get('id'),
                         'location': int(mistake.get('location')) - 1,
                         'wrong': converter.convert(mistake.xpath('./WRONG/text()')[0].strip()),
                         'correction': converter.convert(mistake.xpath('./CORRECTION/text()')[0].strip())})

    rst_items = dict()

    def get_passages_by_id(pgs, _id):
        p = pgs.get(_id)
        if p:
            return p
        _id = _id[:-1] + str(int(_id[-1]) + 1)
        p = pgs.get(_id)
        if p:
            return p
        raise ValueError(f'passage not found by {_id}')

    for mistake in mistakes:
        if mistake['id'] not in rst_items.keys():
            rst_items[mistake['id']] = {'original_text': get_passages_by_id(passages, mistake['id']),
                                        'wrong_ids': [],
                                        'correct_text': get_passages_by_id(passages, mistake['id'])}
        ori_text = rst_items[mistake['id']]['original_text']
        cor_text = rst_items[mistake['id']]['correct_text']
        if len(ori_text) == len(cor_text):
            if ori_text[mistake['location']] in mistake['wrong']:
                rst_items[mistake['id']]['wrong_ids'].append(mistake['location'])
                wrong_char_idx = mistake['wrong'].index(ori_text[mistake['location']])
                start = mistake['location'] - wrong_char_idx
                end = start + len(mistake['wrong'])
                rst_items[mistake['id']][
                    'correct_text'] = f'{cor_text[:start]}{mistake["correction"]}{cor_text[end:]}'
        else:
            print(f'{mistake["id"]}\n{ori_text}\n{cor_text}')
    rst = []
    for k in rst_items:
        if len(rst_items[k]['correct_text']) == len(rst_items[k]['original_text']):
            rst.append({'id': k, **rst_items[k]})
        else:
            text = rst_items[k]['correct_text']
            rst.append({'id': k, 'correct_text': text, 'original_text': text, 'wrong_ids': []})
    return rst

def proc_test_set(fp, converter):
    """
    Generate the SIGHAN15 test set
    Args:
        fp:
        converter:
    Returns:
    """
    inputs = dict()
    with open(os.path.join(fp, 'SIGHAN15_CSC_TestInput.txt'), 'r', encoding='utf8') as f:
        for line in f:
            pid = line[5:14]
            text = line[16:].strip()
            inputs[pid] = text
    rst = []
    with open(os.path.join(fp, 'SIGHAN15_CSC_TestTruth.txt'), 'r', encoding='utf8') as f:
        for line in f:
            pid = line[0:9]
            mistakes = line[11:].strip().split(', ')
            if len(mistakes) <= 1:
                text = converter.convert(inputs[pid])
                rst.append({'id': pid,
                            'original_text': text,
                            'wrong_ids': [],
                            'correct_text': text})
            else:
                wrong_ids = []
                original_text = inputs[pid]
                cor_text = inputs[pid]
                for i in range(len(mistakes) // 2):
                    idx = int(mistakes[2 * i]) - 1
                    cor_char = mistakes[2 * i + 1]
                    wrong_ids.append(idx)
                    cor_text = f'{cor_text[:idx]}{cor_char}{cor_text[idx + 1:]}'
                original_text = converter.convert(original_text)
                cor_text = converter.convert(cor_text)
                if len(original_text) != len(cor_text):
                    print(pid)
                    print(original_text)
                    print(cor_text)
                    continue
                rst.append({'id': pid,
                            'original_text': original_text,
                            'wrong_ids': wrong_ids,
                            'correct_text': cor_text})
    return rst

def read_data(fp):
    for fn in os.listdir(fp):
        if fn.endswith('ing.sgml'):
            with open(os.path.join(fp, fn), 'r', encoding='utf-8', errors='ignore') as f:
                item = []
                for line in f:
                    if line.strip().startswith('<ESSAY') and item:
                        yield ''.join(item)
                        item = [line.strip()]
                    elif line.strip().startswith('<'):
                        item.append(line.strip())


def read_confusion_data(fp):
    fn = os.path.join(fp, 'train.sgml')
    with open(fn, 'r', encoding='utf8') as f:
        item = []
        for line in tqdm(f):
            if line.strip().startswith('<SENT') and item:
                yield ''.join(item)
                item = [line.strip()]
            elif line.strip().startswith('<'):
                item.append(line.strip())


def proc_confusion_item(item):
    """
    Process the Confusionset dataset
    Args:
        item:
    Returns:
    """
    root = etree.XML(item)
    text = root.xpath('/SENTENCE/TEXT/text()')[0]
    mistakes = []
    for mistake in root.xpath('/SENTENCE/MISTAKE'):
        mistakes.append({'location': int(mistake.xpath('./LOCATION/text()')[0]) - 1,
                         'wrong': mistake.xpath('./WRONG/text()')[0].strip(),
                         'correction': mistake.xpath('./CORRECTION/text()')[0].strip()})

    cor_text = text
    wrong_ids = []
    for mis in mistakes:
        cor_text = f'{cor_text[:mis["location"]]}{mis["correction"]}{cor_text[mis["location"] + 1:]}'
        wrong_ids.append(mis['location'])
    rst = [{
        'id': '-',
        'original_text': text,
        'wrong_ids': wrong_ids,
        'correct_text': cor_text
    }]
    if len(text) != len(cor_text):
        return [{'id': '--',
                 'original_text': cor_text,
                 'wrong_ids': [],
                 'correct_text': cor_text}]
    if random.random() < 0.01:
        rst.append({'id': '--',
                    'original_text': cor_text,
                    'wrong_ids': [],
                    'correct_text': cor_text})
    return rst

def preproc():
    rst_items = []
    converter = opencc.OpenCC('tw2sp.json')
    for item in read_data(get_abs_path('datasets', 'csc')):
        rst_items += proc_item(item, converter)
    for item in read_confusion_data(get_abs_path('datasets', 'csc')):
        rst_items += proc_confusion_item(item)
    # Split train and test
    dev_set_len = len(rst_items) // 10
    print(len(rst_items))
    random.seed(666)
    random.shuffle(rst_items)
    dump_json(rst_items[:dev_set_len], get_abs_path('datasets', 'csc', 'dev.json'))
    dump_json(rst_items[dev_set_len:], get_abs_path('datasets', 'csc', 'train.json'))
    gc.collect()

if __name__ == '__main__':
    preproc()
