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
Bert quick start script.
'''

import mindspore as ms
from mindspore.train.model import Model
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.finetune_eval_model import BertCLSModel
import src.generate_mindrecord.tokenization as tokenization
from src.model_utils.config import config as args_opt, bert_net_cfg


examples = [
    ("news_entertainment", "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"),
    ("news_military", "以色列大规模空袭开始！伊朗多个军事目标遭遇打击，誓言对等反击"),
    ("news_finance", "出栏一头猪亏损300元，究竟谁能笑到最后！"),
    ("news_culture", "走进荀子的世界 触摸二千年前的心灵温度"),
    ("news_finance", "区块链投资心得，能做到就不会亏钱"),]

label_map = ["news_story", "news_culture", "news_entertainment", "news_sports", "news_finance",
             "news_house", "news_car", "news_edu", "news_tech", "news_military", "news_travel",
             "news_world", "news_stock", "news_agriculture", "news_game"]

def convert_single_example(text, max_seq_length, tokenizer):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    all_tokens = []
    segment_ids = []
    all_tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens:
        all_tokens.append(token)
        segment_ids.append(0)
    all_tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenization.convert_tokens_to_ids(args_opt.vocab_file_path, all_tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    input_ids = ms.Tensor([input_ids,], dtype=ms.int32)
    input_mask = ms.Tensor([input_mask,], dtype=ms.int32)
    segment_ids = ms.Tensor([segment_ids,], dtype=ms.int32)

    return input_ids, input_mask, segment_ids

def main():
    network = BertCLSModel(bert_net_cfg, False, args_opt.num_class)
    network.set_train(False)
    param_dict = load_checkpoint(args_opt.load_finetune_checkpoint_path)
    load_param_into_net(network, param_dict)
    model = Model(network)

    argmax = P.Argmax()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args_opt.vocab_file_path, do_lower_case=True)
    for label, text in examples:
        input_ids, input_mask, segment_ids = convert_single_example(text, bert_net_cfg.seq_length, tokenizer)
        logit = model.predict(input_ids, input_mask, segment_ids)

        print("sentence: {}".format(text))
        print("label: {}".format(label))
        print("prediction: {}\n".format(label_map[argmax(logit)[0]]))


if __name__ == '__main__':
    main()
