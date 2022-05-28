#!/usr/bin/env python
# coding=utf-8
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
import os
import sys
import zipfile

from io import open

if os.path.exists('train.txt'):
    print('Tokenized text8 already exists - skipping processing')
    sys.exit()

zipfile.ZipFile('text8.zip').extractall()
data = open('text8', 'r', encoding='utf-8').read()

print('Length of text8: {}'.format(len(data)))

# Segment the text8 dataset according to the specification
num_test_chars = 5000000

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    # Change space ' ' to underscore '_'
    part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])
    print('- Writing...')
    f = open(fn, 'w').write(part_str)
    f = open(fn + '.raw', 'w', encoding='utf-8').write(part)
