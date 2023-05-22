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

import argparse
import os
from operator import itemgetter
import mindspore
import mindspore.nn as nn
import mindspore.numpy as np
from mindspore import ops, context, Tensor
from mindspore import dtype as mstype


from model import NeteUser, MfUi, MfFui
from metrics.metrics import rouge_score, bleu_score, root_mean_square_error, mean_absolute_error
from utils import NewDataLoader, NewBatchify, now_time, ids2tokens, set_seed, get_local_time

parser = argparse.ArgumentParser(description='NETE_USER')
# data params
parser.add_argument('--dataset', type=str, default='trip',
                    help='dataset name')
# model params
parser.add_argument('--nlayers', type=int, default=4,
                    help='rating prediction layer number, default=4')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--emsize', type=int, default=32,
                    help='embedding dimension of users、 items and words, default=32')
parser.add_argument('--emsize_mf', type=int, default=32,
                    help='embedding dimension of mf model, default=32')
parser.add_argument('--rnn_dim', type=int, default=256,
                    help='dimension of RNN hidden states, default=256')
parser.add_argument('--rating_reg', type=float, default=1,
                    help='rating regularization rate, default=1')
parser.add_argument('--text_reg', type=float, default=1,
                    help='text regularization rate, default=1')
parser.add_argument('--treat_reg', type=float, default=1,
                    help='treatment regularization rate, default=1')
parser.add_argument('--pui_reg', type=float, default=1,
                    help='pui rate, default=1')
parser.add_argument('--dropout_prob', type=float, default=0.8,
                    help='save ratio in dropout layer, default=0.8')
parser.add_argument('--seq_max_len', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--mean_rating', type=int, default=3,
                    help='distinguish the sentiment')
# running params
parser.add_argument('--alternate_num', type=int, default=1,
                    help='max min alternate_num')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--t', type=float, default=1,
                    help='initial template param')
parser.add_argument('--protect_num', type=float, default=1e-7,
                    help='prevent the denominator from being zero')

parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--endure_times', type=int, default=2,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--device', type=str, default='CPU',
                    help='CPU or GPU')

parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict ')
parser.add_argument('--checkpoint', type=str, default='./nete_user/',
                    help='directory to save the final model')
parser.add_argument('--generated_file_path', type=str, default='_nete_d_generated.txt',
                    help='output file for generated text')
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
context.set_context(max_call_depth=50000)

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

MODEL_PATH = ''
generated_file = args.dataset + args.generated_file_path
prediction_path = os.path.join(args.checkpoint, generated_file)
MF_UI_PATH = './init_ips/mf_ui_small.ckpt'
MF_FUI_PATH = './init_ips/mf_fui_small.ckpt'

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading dataset: {}'.format(args.dataset))
corpus = NewDataLoader(data_path, train_data_path, valid_data_path, test_data_path, args.vocab_size)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
user_inter_count = Tensor.from_numpy(corpus.user_inter_count)

TOKEN_NUMBER = len(corpus.word_dict)
USER_NUMBER = len(corpus.user_dict)
ITEM_NUMBER = len(corpus.item_dict)
FEATURE_NUMBER = len(corpus.feature_set)
trainset_size = corpus.train_size
validset_size = corpus.valid_size
testset_size = corpus.test_size
pad_idx = word2idx['<pad>']

feature_set = corpus.feature_set
feaID_transform_dict = {}
for new_id, old_id in enumerate(feature_set):
    feaID_transform_dict[old_id] = new_id

print(now_time() + '{}: nuser:{} | nitem:{} | ntoken:{} | nfeature:{}'.format(args.dataset, USER_NUMBER, ITEM_NUMBER,
                                                                              TOKEN_NUMBER, FEATURE_NUMBER))
print(now_time() + 'trainset:{} | validset:{} | testset:{}'.format(trainset_size, validset_size, testset_size))

train_data = NewBatchify(corpus.train, word2idx, args.seq_max_len, args.batch_size, shuffle=True)
val_data = NewBatchify(corpus.valid, word2idx, args.seq_max_len, args.batch_size)
test_data = NewBatchify(corpus.test, word2idx, args.seq_max_len, args.batch_size)

###############################################################################
# Build the model
###############################################################################
print('=' * 89)
# Load the best pretrained model.
mf_ui = MfUi(USER_NUMBER, ITEM_NUMBER, args.emsize_mf)
param_dict_mf_ui = mindspore.load_checkpoint(MF_UI_PATH)
param_not_load_mf_ui = mindspore.load_param_into_net(mf_ui, param_dict_mf_ui)
for p_1 in mf_ui.trainable_params():
    p_1.requires_grad = False

mf_fui = MfFui(USER_NUMBER, ITEM_NUMBER, TOKEN_NUMBER, args.emsize_mf)
param_dict_mf_fui = mindspore.load_checkpoint(MF_FUI_PATH)
param_not_load_mf_fui = mindspore.load_param_into_net(mf_fui, param_dict_mf_fui)
for p_2 in mf_fui.trainable_params():
    p_2.requires_grad = False

model = NeteUser(USER_NUMBER, ITEM_NUMBER, TOKEN_NUMBER, FEATURE_NUMBER, args.emsize, args.rnn_dim, args.dropout_prob,
                 args.hidden_size, args.t, args.nlayers)
text_criterion = nn.NLLLoss(ignore_index=pad_idx, reduction='none')  # ignore the padding when computing loss
rating_criterion = nn.MSELoss(reduction='none')
CE_criterion = nn.CrossEntropyLoss()

# max_optimizer
for p_3 in model.trainable_params():
    p_3.requires_grad = False
for component_1 in [model.user_embeddings_mlp.embedding_table, model.item_embeddings_mlp.embedding_table,
                    model.feature_embeddings_mlp.embedding_table]:
    component_1.requires_grad = True
max_optimizer = nn.Adam(params=model.trainable_params(), learning_rate=args.lr)

# min_optimizer
for p_4 in model.untrainable_params():
    p_4.requires_grad = True
for component_2 in [model.user_embeddings_mlp.embedding_table, model.item_embeddings_mlp.embedding_table,
                    model.feature_embeddings_mlp.embedding_table]:
    component_2.requires_grad = False
min_optimizer = nn.Adam(params=model.trainable_params(), learning_rate=args.lr)

for p_5 in model.untrainable_params():
    p_5.requires_grad = True


def forward_fn_min(user, item, feature, sentiment_index, rating_reg, rating_p, rating, text_reg, seq, treat_reg,
                   r_weight_uiz1, repeat_r_weight_uiz3f, treat_item, treat_fea, pred_item1, pred_item2, pred_fea):
    log_word_prob = model(user, item, sentiment_index, seq[:, :-1], feature)  # (batch_size, tgt_len, ntoken)

    r_loss = ops.mean(r_weight_uiz1 * rating_criterion(rating_p, rating))
    t_loss = ops.mean(repeat_r_weight_uiz3f * text_criterion(log_word_prob.view((-1, TOKEN_NUMBER)),
                                                             seq[:, 1:].reshape((-1,))))
    treatment_loss = CE_criterion(pred_item1, treat_item) \
                     + CE_criterion(pred_item2, treat_item) \
                     + CE_criterion(pred_fea, treat_fea)
    loss = rating_reg * r_loss + text_reg * t_loss + treat_reg * treatment_loss
    return r_loss, t_loss, loss

grad_fn_min = mindspore.value_and_grad(forward_fn_min, None, min_optimizer.parameters, has_aux=True)


def train_step_min(user, item, feature, sentiment_index, rating_reg, rating_p, rating, text_reg, seq,
                   treat_reg, r_weight_uiz1, repeat_r_weight_uiz3f, treat_item, treat_fea,
                   pred_item1, pred_item2, pred_fea):
    (r_loss, t_loss, loss), grads = grad_fn_min(user, item, feature, sentiment_index, rating_reg, rating_p, rating,
                                                text_reg, seq, treat_reg, r_weight_uiz1, repeat_r_weight_uiz3f,
                                                treat_item, treat_fea, pred_item1, pred_item2, pred_fea)
    min_optimizer(grads)
    return r_loss, t_loss, loss


def forward_fn_max(user, item, feature, sentiment_index, rating_reg, rating_p, rating, text_reg, seq,
                   pui_reg, r_weight_uiz1, repeat_r_weight_uiz3f, weight_uiz1, weight_uiz2,
                   weight_uiz3_f, hat_weight_ui_f, hat_weight_ui):
    log_word_prob = model(user, item, sentiment_index, seq[:, :-1], feature)  # (batch_size, tgt_len, ntoken)
    r_loss = ops.mean(r_weight_uiz1 * rating_criterion(rating_p, rating))
    t_loss = ops.mean(repeat_r_weight_uiz3f * text_criterion(log_word_prob.view((-1, TOKEN_NUMBER)),
                                                             seq[:, 1:].reshape((-1,))))
    puiz1_loss = np.sum(ops.abs(weight_uiz1 - hat_weight_ui))
    puiz2_loss = np.sum(ops.abs(weight_uiz2 - hat_weight_ui))
    puiz3_f_loss = np.sum(ops.abs(weight_uiz3_f - hat_weight_ui_f))
    robust_loss = pui_reg * (puiz1_loss + puiz2_loss + puiz3_f_loss)
    loss = - (rating_reg * r_loss + text_reg * t_loss) + robust_loss
    return r_loss, t_loss, loss

grad_fn_max = mindspore.value_and_grad(forward_fn_max, None, max_optimizer.parameters, has_aux=True)


def train_step_max(user, item, feature, sentiment_index, rating_reg, rating_p, rating, text_reg, seq,
                   pui_reg, r_weight_uiz1, repeat_r_weight_uiz3f, weight_uiz1, weight_uiz2,
                   weight_uiz3_f, hat_weight_ui_f, hat_weight_ui):
    (r_loss, t_loss, loss), grads = grad_fn_max(user, item, feature, sentiment_index, rating_reg, rating_p, rating,
                                                text_reg, seq, pui_reg, r_weight_uiz1, repeat_r_weight_uiz3f,
                                                weight_uiz1, weight_uiz2, weight_uiz3_f, hat_weight_ui_f, hat_weight_ui)
    max_optimizer(grads)
    return r_loss, t_loss, loss


def train(data, weight_ui, weight_uif):  # train
    # Turn on training mode which enables dropout.
    model.set_train()
    train_rating_loss = 0.
    train_text_loss = 0.
    total_loss = 0.
    total_sample = 0
    while True:
        batch_data = data.next_batch()
        user = batch_data.user
        item = batch_data.item
        rating = batch_data.rating
        seq = batch_data.seq
        feature = batch_data.feature
        index = batch_data.index
        batch_size = user.size
        fea = ops.squeeze(feature)  # (batch_size,)
        fea_trans = itemgetter(*fea.numpy())(feaID_transform_dict)
        fea_trans = Tensor(fea_trans)

        treat_item = np.zeros((user.size, ITEM_NUMBER))
        treat_item[np.arange(user.size), item] = 1.0
        treat_fea = np.zeros((user.size, TOKEN_NUMBER))
        treat_fea[np.arange(user.size), feature] = 1.0

        # p_hat
        hat_weight_ui = weight_ui[index]
        hat_weight_ui_f = weight_uif[index]

        print(now_time() + 'Maximization')
        # MAX MIN training
        for _ in range(args.alternate_num):
            # p
            weight_uiz1, _, _ = model.predict_puiz1(user, item)
            weight_uiz2 = model.predict_puiz2(user, item)
            weight_uiz3_f = model.predict_puiz3_f(user, item, feature, fea_trans)
            weight_uiz1_under = weight_uiz1 * user_inter_count[user]
            weight_uiz2_under = weight_uiz2 * user_inter_count[user]
            weight_uiz3f_under = weight_uiz2_under * weight_uiz3_f * 1
            r_weight_uiz1 = np.reciprocal(weight_uiz1_under + args.protect_num)
            r_weight_uiz3f = np.reciprocal(weight_uiz3f_under + args.protect_num)
            repeat_r_weight_uiz3f = r_weight_uiz3f.repeat_interleave(16)

            rating_p = model.predict_rating(user, item)  # (batch_size,)
            one = np.ones_like(rating, dtype=mstype.int64)
            zero = np.zeros_like(rating, dtype=mstype.int64)
            sentiment_index = np.where(rating_p < args.mean_rating, zero, one)

            r_loss, t_loss, loss = train_step_max(user, item, feature, sentiment_index, args.rating_reg,
                                                  rating_p, rating, args.text_reg, seq, args.pui_reg,
                                                  r_weight_uiz1, repeat_r_weight_uiz3f, weight_uiz1, weight_uiz2,
                                                  weight_uiz3_f, hat_weight_ui_f, hat_weight_ui)

        print(now_time() + 'Minimization')
        # p
        weight_uiz1, _, _ = model.predict_puiz1(user, item)
        weight_uiz2 = model.predict_puiz2(user, item)
        weight_uiz3_f = model.predict_puiz3_f(user, item, feature, fea_trans)
        weight_uiz1_under = weight_uiz1 * user_inter_count[user]
        weight_uiz2_under = weight_uiz2 * user_inter_count[user]
        weight_uiz3f_under = weight_uiz2_under * weight_uiz3_f * 1
        r_weight_uiz1 = np.reciprocal(weight_uiz1_under + args.protect_num)
        r_weight_uiz3f = np.reciprocal(weight_uiz3f_under + args.protect_num)
        repeat_r_weight_uiz3f = r_weight_uiz3f.repeat_interleave(16)

        rating_p = model.predict_rating(user, item)  # (batch_size,)
        one = np.ones_like(rating, dtype=mstype.int64)
        zero = np.zeros_like(rating, dtype=mstype.int64)
        sentiment_index = np.where(rating_p < args.mean_rating, zero, one)

        pred_item1 = model.predict_treat1(user, item)
        pred_item2 = model.predict_treat2(user, item)
        pred_fea = model.predict_treat3(user, item, feature)

        r_loss, t_loss, loss = train_step_min(user, item, feature, sentiment_index, args.rating_reg, rating_p, rating,
                                              args.text_reg, seq, args.treat_reg, r_weight_uiz1, repeat_r_weight_uiz3f,
                                              treat_item, treat_fea, pred_item1, pred_item2, pred_fea)
        train_rating_loss += batch_size * r_loss
        train_text_loss += batch_size * t_loss
        total_loss += batch_size * loss
        total_sample += batch_size

        if data.step == data.total_step:
            break
    return train_rating_loss / total_sample, train_text_loss / total_sample, total_loss / total_sample


def evaluate(data):
    # Turn on training mode which enables dropout.
    model.set_train(False)
    eval_rating_loss = 0.
    eval_text_loss = 0.
    total_loss = 0.
    total_sample = 0
    rating_predict = []
    while True:
        batch_data = data.next_batch()
        user = batch_data.user
        item = batch_data.item
        rating = batch_data.rating
        seq = batch_data.seq
        feature = batch_data.feature
        batch_size = user.size

        rating_p = model.predict_rating(user, item)  # (batch_size,)
        rating_predict.extend(rating_p.asnumpy().tolist())
        one = np.ones_like(rating, dtype=mstype.int64)
        zero = np.zeros_like(rating, dtype=mstype.int64)
        sentiment_index = np.where(rating_p < args.mean_rating, zero, one)
        log_word_prob = model(user, item, sentiment_index, seq[:, :-1], feature)  # (batch_size, tgt_len, ntoken)
        r_loss = ops.mean(rating_criterion(rating_p, rating))
        t_loss = ops.mean(text_criterion(log_word_prob.view((-1, TOKEN_NUMBER)), seq[:, 1:].reshape((-1,))))
        loss = r_loss + t_loss

        eval_rating_loss += batch_size * r_loss
        eval_text_loss += batch_size * t_loss
        total_loss += batch_size * loss
        total_sample += batch_size

        if data.step == data.total_step:
            break
    return eval_rating_loss / total_sample, eval_text_loss / total_sample, total_loss / total_sample


def generate(data):  # generate explanation & evaluate on metrics
    # Turn on evaluation mode which disables dropout.
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
            # produce a word at each step
            if idx == 0:
                hidden = model.encoder(user, item, sentiment_index, feature)
                log_word_prob, hidden = model.decoder(inputs, feature, hidden)  # (batch_size, 1, ntoken)
            else:
                log_word_prob, hidden = model.decoder(inputs, feature, hidden)  # (batch_size, 1, ntoken)
            word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
            inputs = ops.argmax(word_prob, axis=1, keepdims=True)
            ids = ops.concat([ids, Tensor(inputs, mstype.int32)], 1)  # (batch_size, len++)
        ids = ids[:, 1:].asnumpy().tolist()  # remove bos
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
    for (key, value) in rouge.items():
        print(now_time() + '{} {:7.4f}'.format(key, value))
    # generate
    text_out = ''
    for (real, fake) in zip(text_test, text_predict):  # format: ground_truth|context|explanation
        text_out += '{}\n{}\n\n'.format(real, fake)
    return text_out


###############################################################################
# Loop over epochs.
###############################################################################
print(now_time() + 'NETE_USER learning')
best_val_loss = float('inf')
ENDURE = 0
print(now_time() + 'Initial IPS estimation')
train_hat_weight_ui = mf_ui(train_data.user, train_data.item)
feature_trans = itemgetter(*train_data.feature.squeeze().numpy())(feaID_transform_dict)
feature_trans = Tensor(feature_trans)
train_hat_weight_ui_f = mf_fui(train_data.user, train_data.item, feature_trans)

for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    rating_loss, text_loss, train_loss = train(train_data, train_hat_weight_ui, train_hat_weight_ui_f)
    print(now_time() + 'rating loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} on train'.format(
        float(rating_loss), float(text_loss), float(train_loss)))
    rating_loss, text_loss, val_loss = evaluate(val_data)
    print(now_time() + 'rating loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} on valid'.format(
        float(rating_loss), float(text_loss), float(val_loss)))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        SAVED_MODEL_FILE = '{}-{}.ckpt'.format('nete', get_local_time())
        MODEL_PATH = os.path.join(args.checkpoint, SAVED_MODEL_FILE)
        mindspore.save_checkpoint(model, MODEL_PATH)
        print(now_time() + 'Save the best model' + MODEL_PATH)
    else:
        ENDURE += 1
        print(now_time() + 'Endured {} time(s)'.format(ENDURE))
        if ENDURE == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

print(now_time() + 'Load the best model' + MODEL_PATH)
param_dict = mindspore.load_checkpoint(MODEL_PATH)
param_not_load = mindspore.load_param_into_net(model, param_dict)

# Run on test data.
TEST_O = generate(test_data)
print(now_time() + 'NETE_USER is OK!')
print('=' * 89)
