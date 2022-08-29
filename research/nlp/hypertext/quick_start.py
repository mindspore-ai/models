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

'''
Hypertext quick start script.
'''
import argparse
import os
import numpy as np
from mindspore import load_checkpoint, load_param_into_net, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import Squeeze, Argmax
from src.config import Config
from src.data_preprocessing import changeListToText
from src.dataset import hash_str, addWordNgrams, load_vocab
from src.hypertext import HModel

MAX_VOCAB_SIZE = 5000000
UNK, PAD = '<UNK>', '<PAD>'

parser = argparse.ArgumentParser(description='HyperText Text Classification Quick Start')
parser.add_argument('--modelPath', default='./output/hypertext_tnews.ckpt', type=str, help='save model path')
parser.add_argument('--datasetType', default='tnews', type=str, help='iflytek/tnews')
args = parser.parse_args()
config = Config(None, None, None)
if args.datasetType == 'tnews':
    config.useTnews()
    config.vocab_path = os.path.join('./data/tnews_public', 'vocab.txt')
    config.num_classes = 15
    examples = [
        ("news_military", "歼20座舱盖上的两条“花纹”是什么？"),
        ("news_entertainment", "谢娜地位不保！张杰晒出双胞胎女儿照片，黄磊连称：好漂亮"),
        ("news_house", "如果房价涨到买不动了，会是什么样的结果？"),
        ("news_edu", "关于“寒门再难出贵子”你怎么看？")
    ]
    real_label = [100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116]
    label_dict = {100: 'news_story', 101: 'news_culture', 102: 'news_entertainment', 103: 'news_tech',
                  104: 'news_finance', 106: 'news_house', 107: 'news_car', 108: 'news_edu', 109: 'news_tech',
                  110: 'news_military', 112: 'news_travel', 113: 'news_world', 114: 'stock',
                  115: 'news_agriculture', 116: 'news_game'}
else:
    config.useIflyek()
    config.vocab_path = os.path.join('./data/iflytek_public', 'vocab.txt')
    config.num_classes = 119
    examples = [
        (96, "这是一款全面且专业、简单且实用、适合备考注册会计师的小伙伴们的智能学习软件。讲义、视频、习题三位一体，"
             "360度全方位覆盖，内容全面且专业。内容包括1.视频有免费的视频哦，分科目、分章节、分年份、分班型面面俱到；"
             "2.讲义重点突出，考点明确，给你更明确的方向；3.习题章节练习题、模拟题、历年真题尽在其中。应用功能包括1.收藏夹"
             "可以收藏自己喜欢的题目；2.错题集里面容纳了你的知识盲点；3.笔记随时写下自己的学习体会；4.下载管理专门存放下载"
             "的视频，方便管理哦。快乐与学习同在，ministudy智能题库,您的随身考试助手更新内容优化体验，增强稳定性。"),
        (70, "虚拟来电、虚拟短信，模拟真实来电和短信，完美通话界面丰富的通话皮肤，小米、vivo、oppo、华为等皮肤供您选择使用"
             "本程序仅仅是伪造一条真实的短信，或模拟一个真实的来电，并不会实际发送或接收到真实电话或短信，仅供娱乐使用，"
             "无恶意插件，请放心使用。适用场景1、假装收到各种优惠短信，爱慕短信等在朋友面前秀一秀；2、在各种场合的应酬中，"
             "让你找到合适的理由离开；3、模拟来电，时间自定，来电皮肤自定适应不同品牌手机；4、通话结束后一条真实通话记录完美；"
             ",模拟真实来电，多品牌通话皮肤,优化来电皮肤；修复bug；"),
        (108, "扫码公交，随兴出行。智慧出行，给你不一样的体验。乘公交，就是快，只要手机点开APP，扫二维码轻松乘坐公交。解决乘公交"
              "时没有零钱而烦恼；解决公交卡容易丢失的烦恼。,扫码公交，随兴出行。,1.对初始页面进行了优化"),
        (70, "QX模块是一款qq的增强xposed模块。更新内容优化系统性能，体验更顺畅")
    ]
    labels = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110,
              111, 112, 113, 114, 115, 116, 117, 118, 12, 13, 14, 15, 16, 17, 18,
              19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33,
              34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
              5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64,
              65, 66, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8,
              80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 9, 90, 91, 92, 93, 94, 95,
              96, 97, 98, 99]

def changeText(sentence):
    """change tnews"""
    sentence_content = sentence.strip('\n').replace('\t', '')
    return changeListToText(sentence_content)

tokenizer = lambda x: x.split(' ')  # word-level
vocab = load_vocab(config.vocab_path, max_size=MAX_VOCAB_SIZE, min_freq=int(config.min_freq))
config.n_vocab = len(vocab)

def process_oneline(sentence):
    """ process """
    sentence = sentence.strip()
    if sentence == 0:
        sentence = "0"
    tokens = tokenizer(sentence.strip())
    seq_len = len(tokens)
    if seq_len > config.max_length:
        tokens = tokens[:config.max_length]

    token_hash_list = [hash_str(token) for token in tokens]
    ngram = addWordNgrams(token_hash_list, config.wordNgrams, config.bucket)
    ngram_pad_size = int((config.wordNgrams - 1) * (config.max_length - config.wordNgrams / 2))

    if len(ngram) > ngram_pad_size:
        ngram = ngram[:ngram_pad_size]
    tokens_to_id = [vocab.get(token, vocab.get(UNK)) for token in tokens]

    return tokens_to_id, ngram


model_path = args.modelPath
hmodel = HModel(config).to_float(mstype.float16)
param_dict = load_checkpoint(model_path)
load_param_into_net(hmodel, param_dict)
squ = Squeeze(-1)
argmax = Argmax(output_type=mstype.int32)

for label, text in examples:
    content = changeText(text)
    ids, ngrad_ids = process_oneline(content)
    ids = list(ids + [0] * (config.max_length - len(ids)))
    ngrad_ids = list(ngrad_ids + [0] * (config.max_length - len(ngrad_ids)))
    ids_np = np.array(ids).reshape(1, len(ids)).astype(np.int32)
    ngrad_np = np.array(ngrad_ids).reshape(1, len(ngrad_ids)).astype(np.int32)
    hmodel.set_train(False)
    out = hmodel(Tensor(ids_np), Tensor(ngrad_np))
    predict = argmax(out)
    if args.datasetType == 'tnews':
        index = predict[0]
        key = real_label[index]
        print("================================================")
        print("Text:")
        print(text)
        print("Label:")
        print(label)
        print("Predict")
        print(label_dict[key])
    elif args.datasetType == 'iflytek':
        print("================================================")
        print("Text:")
        print(text)
        print("Label:")
        print(labels.index(label))
        print("Predict")
        print(predict[0])
