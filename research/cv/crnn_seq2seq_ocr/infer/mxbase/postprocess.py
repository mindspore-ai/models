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
postprocess.

"""

import codecs
import os

def initialize_vocabulary(vocabulary_path):
    """
    initialize vocabulary from file.
    assume the vocabulary is stored one-item-per-line
    """
    characters_class = 9999

    if os.path.exists(vocabulary_path):
        with codecs.open(vocabulary_path, 'r', encoding='utf-8') as voc_file:
            rev_vocab = [line.strip() for line in voc_file]

        vocab = {x: y for (y, x) in enumerate(rev_vocab)}

        reserved_char_size = characters_class - len(rev_vocab)
        if reserved_char_size < 0:
            raise ValueError("Number of characters in vocabulary is equal or larger than config.characters_class")

        for _ in range(reserved_char_size):
            rev_vocab.append('')

        # put space at the last position
        vocab[' '] = len(rev_vocab)
        rev_vocab.append(' ')
        return vocab, rev_vocab

    raise ValueError("Initializing vocabulary ends: %s" % vocabulary_path)


def text_standardization(text_in):
    """
    replace some particular characters
    """
    stand_text = text_in.strip()
    stand_text = ' '.join(stand_text.split())
    stand_text = stand_text.replace(u'(', u'（')
    stand_text = stand_text.replace(u')', u'）')
    stand_text = stand_text.replace(u':', u'：')
    return stand_text


def LCS_length(str1, str2):
    """
    calculate longest common sub-sequence between str1 and str2
    """
    if str1 is None or str2 is None:
        return 0

    len1 = len(str1)
    len2 = len(str2)
    if len1 == 0 or len2 == 0:
        return 0

    lcs = [[0 for _ in range(len2 + 1)] for _ in range(2)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                lcs[i % 2][j] = lcs[(i - 1) % 2][j - 1] + 1
            else:
                if lcs[i % 2][j - 1] >= lcs[(i - 1) % 2][j]:
                    lcs[i % 2][j] = lcs[i % 2][j - 1]
                else:
                    lcs[i % 2][j] = lcs[(i - 1) % 2][j]

    return lcs[len1 % 2][-1]


def get_acc():
    '''generate accuracy'''
    vocab_path = '../../general_chars.txt'
    _, rev_vocab = initialize_vocabulary(vocab_path)

    num_correct_char = 0
    num_total_char = 0
    num_correct_word = 0
    num_total_word = 0

    text_path = '../data/fsns/test-anno/image2text.txt'
    anno_text = {}
    anno_file = open(text_path, 'r').readlines()
    for line in anno_file:
        file_name = line.split('\t')[0]
        labels = line.split('\t')[1].split('\n')[0]
        anno_text[file_name] = labels

    predict_path = 'temp_infer_result.txt'
    correct_file = 'result_correct.txt'
    incorrect_file = 'result_incorrect.txt'
    with codecs.open(correct_file, 'w', encoding='utf-8') as fp_output_correct, \
            codecs.open(incorrect_file, 'w', encoding='utf-8') as fp_output_incorrect:

        pred_file = open(predict_path, 'r').readlines()
        for line in pred_file:
            blocks = line.split(' ')
            file_name = blocks[0]
            decoded_words = []
            for i in blocks[1:]:
                decoded_words.append(rev_vocab[int(i)])

            text = anno_text[file_name]
            text = text_standardization(text)
            predict = text_standardization("".join(decoded_words))

            if predict == text:
                num_correct_word += 1
                fp_output_correct.write('\t\t' + text + '\n')
                fp_output_correct.write('\t\t' + predict + '\n\n')

            else:
                fp_output_incorrect.write('\t\t' + text + '\n')
                fp_output_incorrect.write('\t\t' + predict + '\n\n')

            num_total_word += 1
            num_correct_char += 2 * LCS_length(text, predict)
            num_total_char += len(text) + len(predict)

    print('\nnum of correct characters = %d' % (num_correct_char))
    print('\nnum of total characters = %d' % (num_total_char))
    print('\nnum of correct words = %d' % (num_correct_word))
    print('\nnum of total words = %d' % (num_total_word))
    print('\ncharacter precision = %f' % (float(num_correct_char) / num_total_char))
    print('\nAnnotation precision precision = %f' % (float(num_correct_word) / num_total_word))

    with open("eval_sdk.log", 'w') as f:
        f.write('num of correct characters = {}\n'.format(num_correct_char))
        f.write('num of total characters = {}\n'.format(num_total_char))
        f.write('num of correct words = {}\n'.format(num_correct_word))
        f.write('num of total words = {}\n'.format(num_total_word))
        f.write('character precision = {}\n'.format(float(num_correct_char) / num_total_char))
        f.write('Annotation precision precision = {}\n'.format(float(num_correct_word) / num_total_word))


if __name__ == '__main__':
    get_acc()
