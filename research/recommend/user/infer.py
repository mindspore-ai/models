# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import argparse
import mindspore
import mindspore.numpy as np
from mindspore import ops, context, Tensor
from mindspore import dtype as mstype

from model import NETE
from metrics.metrics import rouge_score, bleu_score, root_mean_square_error, mean_absolute_error
from utils import NewDataLoader, NewBatchify, now_time, ids2tokens, set_seed


parser = argparse.ArgumentParser(description='Inference (take NETE for example)')
# data params
parser.add_argument('--dataset', type=str, default='small',
                    help='dataset name')
# model params
parser.add_argument('--nlayers', type=int, default=4,
                    help='rating prediction layer number, default=4')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--emsize', type=int, default=32,
                    help='embedding dimension of users、 items and words, default=32')
parser.add_argument('--rnn_dim', type=int, default=256,
                    help='dimension of RNN hidden states, default=256')
parser.add_argument('--dropout_prob', type=float, default=0.8,
                    help='save ratio in dropout layer, default=0.8')
parser.add_argument('--seq_max_len', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--mean_rating', type=int, default=3,
                    help='distinguish the sentiment')
# running params
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--endure_times', type=int, default=2,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--model_path', type=str, default='nete_small.ckpt',
                    help='output file for generated text')
parser.add_argument('--checkpoint', type=str, default='./nete/',
                    help='directory to save the final model')
parser.add_argument('--generated_file_path', type=str, default='_nete_generated.txt',
                    help='output file for generated text')
parser.add_argument('--device', type=str, default='CPU',
                    help='CPU or GPU')
args = parser.parse_args()

data_path = 'dataset/' + args.dataset + '/reviews.pickle'
train_data_path = 'dataset/' + args.dataset + '/train.csv'
valid_data_path = 'dataset/' + args.dataset + '/valid.csv'
test_data_path = 'dataset/' + args.dataset + '/test.csv'
if data_path is None:
    parser.error('--data_path should be provided for loading data')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
set_seed(args.seed)
context.set_context(mode=context.GRAPH_MODE, device_target=args.device, save_graphs=False)

generated_file = args.dataset + args.generated_file_path
prediction_path = os.path.join(args.checkpoint, generated_file)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading dataset: {}'.format(args.dataset))
corpus = NewDataLoader(data_path, train_data_path, valid_data_path, test_data_path, args.vocab_size)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
test_data = NewBatchify(corpus.test, word2idx, args.seq_max_len, args.batch_size)

TOKEN_NUMBER = len(corpus.word_dict)
USER_NUMBER = len(corpus.user_dict)
ITEM_NUMBER = len(corpus.item_dict)
pad_idx = word2idx['<pad>']

# Build the model
model = NETE(USER_NUMBER, ITEM_NUMBER, TOKEN_NUMBER, args.emsize, args.rnn_dim, args.dropout_prob, args.hidden_size,
             args.nlayers)
print(now_time() + 'Load the pre-trained model: ' + args.model_path)
param_dict = mindspore.load_checkpoint(args.model_path)
param_not_load = mindspore.load_param_into_net(model, param_dict)


def inference(data):
    model.set_train(False)
    idss_predict = []
    rating_predict = []
    while True:
        batch_data = data.next_batch()
        user = batch_data.user
        item = batch_data.item
        rating = batch_data.rating
        seq = batch_data.seq
        feature = batch_data.feature
        rating_p = model.predict_rating(user, item)  # (batch_size,)
        rating_predict.extend(rating_p.asnumpy().tolist())
        one = np.ones_like(rating, dtype=mstype.int64)
        zero = np.zeros_like(rating, dtype=mstype.int64)
        sentiment_index = np.where(rating_p < args.mean_rating, zero, one)

        inputs = seq[:, :1]  # (batch_size, 1)
        hidden = None
        ids = inputs
        for idx in range(args.seq_max_len):
            if idx == 0:
                hidden = model.encoder(user, item, sentiment_index)
                log_word_prob, hidden = model.decoder(inputs, feature, hidden)  # (batch_size, 1, ntoken)
            else:
                log_word_prob, hidden = model.decoder(inputs, feature, hidden)  # (batch_size, 1, ntoken)
            word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
            inputs = ops.argmax(word_prob, axis=1, keepdims=True)
            ids = ops.concat([ids, Tensor(inputs, mstype.int32)], 1)  # (batch_size, len++)
        ids = ids[:, 1:].asnumpy().tolist()
        idss_predict.extend(ids)

        if data.step == data.total_step:
            break

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.asnumpy().tolist(), rating_predict)]
    rmse_score = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    mae_score = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(rmse_score))
    print(now_time() + 'MAE {:7.4f}'.format(mae_score))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.asnumpy().tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    # bleu
    bleu_score_1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(bleu_score_1))
    bleu_score_4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(bleu_score_4))
    # rouge
    text_test = [' '.join(tokens) for tokens in tokens_test]  # 32003
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    rouge = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in rouge.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    # generate
    text_out = ''
    for (real, fake) in zip(text_test, text_predict):  # format: ground_truth|context|explanation
        text_out += '{}\n{}\n\n'.format(real, fake)
    return text_out

# Inference on test data.
print(now_time() + 'Run on test set:')
TEST_O = inference(test_data)
print(now_time() + 'Running is OK!')
print('=' * 89)
