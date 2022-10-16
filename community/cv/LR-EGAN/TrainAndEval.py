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
""" Train and Eval Function of LR-EGAN """
from __future__ import division
from __future__ import print_function
import warnings
from time import time
from preprocess import preprocess, parameter
from sklearn.metrics import average_precision_score
from src.utils import load_weights
from src.LR_EGAN import CB_GAN
from src.pyod_utils import standardizer, AUC_and_Gmean
from mindspore import context, Tensor
from configs import parse_args
from selection import select_random
import pandas as pd
import numpy as np


# suppress warnings for clean output
warnings.filterwarnings("ignore")


# Define data file and result file
folder_name = 'data'
save_dir = 'results'

# define the number of iterations
n_ite = 1
n_classifiers = 1

df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc',
              'CB-GAN']

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
rec_df = pd.DataFrame(columns=df_columns)
gmean_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

args = parse_args()

if args.mindspore_mode == "GRAPH_MODE":
    context.set_context(mode=context.GRAPH_MODE)
else:
    context.set_context(mode=context.PYNATIVE_MODE)

context.set_context(device_target=args.device, device_id=args.device_id)
dataset_args = parameter()

if args.data_path[-1] != '/':
    args.data_path = args.data_path+'/'
dataset_args.data_name = args.data_name
dataset_args.data_path = args.data_path
dataset_args.data_format = args.data_format

X_train, y_train, X_val, y_val, X_test, y_test = preprocess(dataset_args)


X_train.astype(np.float32)
X_test.astype(np.float32)

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=None)

y = y.astype(np.int32)

outliers_fraction = np.count_nonzero(y) / len(y)
outliers_percentage = round(outliers_fraction * 100, ndigits=4)

# construct containers for saving results
roc_list = [dataset_args.data_name, X.shape[0],
            X.shape[1], outliers_percentage]
prn_list = [dataset_args.data_name, X.shape[0],
            X.shape[1], outliers_percentage]
rec_list = [dataset_args.data_name, X.shape[0],
            X.shape[1], outliers_percentage]
gmean_list = [dataset_args.data_name,
              X.shape[0], X.shape[1], outliers_percentage]
time_list = [dataset_args.data_name,
             X.shape[0], X.shape[1], outliers_percentage]

X_train_norm, X_test_norm = standardizer(X_train, X_test)
X_train_norm_not_used, X_val_norm = standardizer(X_train, X_val)

X_train_pandas = pd.DataFrame(X_train_norm)
X_test_pandas = pd.DataFrame(X_test_norm)
X_val_pandas = pd.DataFrame(X_val_norm)
X_train_pandas.fillna(X_train_pandas.mean(), inplace=True)
X_test_pandas.fillna(X_train_pandas.mean(), inplace=True)
X_val_pandas.fillna(X_val_pandas.mean(), inplace=True)
X_train_norm = X_train_pandas.values
X_test_norm = X_test_pandas.values
X_val_norm = X_val_pandas.values

roc_mat = np.zeros([n_ite, n_classifiers])
pr_mat = np.zeros([n_ite, n_classifiers])
prn_mat = np.zeros([n_ite, n_classifiers])
rec_mat = np.zeros([n_ite, n_classifiers])
gmean_mat = np.zeros([n_ite, n_classifiers])
time_mat = np.zeros([n_ite, n_classifiers])

result_train = pd.DataFrame([])
result_test = pd.DataFrame([])
result_val = pd.DataFrame([])

for i in range(n_ite):
    print("\n... Processing", dataset_args.data_name, '...', 'Iteration', i + 1)

    X_train_norm = X_train_norm.astype(np.float32)
    X_test_norm = X_test_norm.astype(np.float32)
    X_val_norm = X_val_norm.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_val = y_val.astype(np.int32)
    data_x = X_train_norm
    data_y = y_train
    X_test = X_test_norm
    X_val = X_val_norm

    if args.LR_flag:
        input_dim = data_x.shape[1]
        y_train_real = data_y.copy()
        print('Abnormal samples in train_set before add noise label:',
              y_train_real.sum())
        data_y = select_random(data_y, args.noiseLabelRate)
        print('Abnormal samples in train_set after add noise label:', data_y.sum())
        X_val = data_x
        y_val = y_train_real

    t0 = time()
    cb_gan = CB_GAN(args, data_x, data_y, X_test, y_test, X_val, y_val)
    if args.mode == "eval":
        print("reload trained parameters")
        if args.LR_flag:
            weights_root = f'./trueRate{args.noiseLabelRate}_weights_' + \
                args.data_name
        else:
            weights_root = 'weights_'+args.data_name

        cb_gan.netG, cb_gan.NetD_Ensemble = load_weights(
            cb_gan.netG, cb_gan.NetD_Ensemble, weights_root=weights_root, epoch=args.resume_epoch)
    else:
        start_time = time()
        cb_gan.fit()
        print(
            f'Time cost in training is {(time() - start_time):.2f} seconds\n')

    test_scores = cb_gan.predict(
        Tensor(X_test.astype(np.float32)), cb_gan.NetD_Ensemble)
    train_scores = cb_gan.predict(
        Tensor(data_x.astype(np.float32)), cb_gan.NetD_Ensemble)
    val_scores = cb_gan.predict(
        Tensor(X_val.astype(np.float32)), cb_gan.NetD_Ensemble)
    t1 = time()
    duration = round(t1 - t0, ndigits=4)

    test_scores = test_scores.asnumpy()
    train_scores = train_scores.asnumpy()
    val_scores = val_scores.asnumpy()

    roc, prn, rec, gmean = AUC_and_Gmean(y_test, test_scores)
    pr = average_precision_score(y_test, test_scores)
    roc_t, prn_t, rec_t, gmean_t = AUC_and_Gmean(y_train, train_scores)
    roc_v, prn_v, rec_v, gmean_v = AUC_and_Gmean(y_val, val_scores)

    test_label = 'golden_label'
    test_name = 'EALGAN_score'

    result_train_path = "./result_train_"+dataset_args.data_name+".csv"
    result_test_path = "./result_test_"+dataset_args.data_name+".csv"
    result_val_path = "./result_val_"+dataset_args.data_name+".csv"

    if i == 0:
        result_train[test_name] = train_scores
        result_train[test_label] = y_train
    else:
        result_train[test_name] += train_scores
        result_train[test_label] += y_train
    if i == n_ite - 1:
        result_train[test_name] /= n_ite
        result_train[test_label] /= n_ite

    if i == 0:
        result_test[test_name] = test_scores
        result_test[test_label] = y_test
    else:
        result_test[test_name] += test_scores
        result_test[test_label] += y_test
    if i == n_ite - 1:
        result_test[test_name] /= n_ite
        result_test[test_label] /= n_ite

    if i == 0:
        result_val[test_name] = val_scores
        result_val[test_label] = y_val
    else:
        result_val[test_name] += val_scores
        result_val[test_label] += y_val
    if i == n_ite - 1:
        result_val[test_name] /= n_ite
        result_val[test_label] /= n_ite

    print(('AUC:{roc}, precision @ rank n:{prn}, recall @ rank n:{rec}, Gmean:{gmean}, train_AUC:{train_auc},'+\
'train_precision:{train_prn}, train_recall:{train_rec}, train_gmean:{train_gmean}, val_AUC:{val_auc},'+\
'val_precision:{val_prn}, val_recall:{val_rec}, val_gmean:{val_gmean},  execution time: {duration}s').format(\
roc=roc, prn=prn, rec=rec, gmean=gmean, train_auc=roc_t, train_prn=prn_t, train_rec=rec_t, train_gmean=gmean_t,\
val_auc=roc_v, val_prn=prn_v, val_rec=rec_v, val_gmean=gmean_v, duration=duration))

    time_mat[i, 0] = duration
    roc_mat[i, 0] = roc
    pr_mat[i, 0] = pr
    prn_mat[i, 0] = prn
    rec_mat[i, 0] = rec
    gmean_mat[i, 0] = gmean

print("Average: ")
print('auc: ', np.mean(roc_mat, axis=0))
print('gmean: ', np.mean(gmean_mat, axis=0))
print('pr: ', np.mean(pr_mat, axis=0))
print('prn: ', np.mean(prn_mat, axis=0))
print('rec: ', np.mean(rec_mat, axis=0))
