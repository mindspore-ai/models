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
import math
import numpy as np
import mindspore as ms
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def evaluate_fewshot(
        encoder, loader, n_way=5, n_shots='1,5', n_query=15, n_tasks=3000, classifier='LR', power_norm=True):

    n_shots = [int(i) for i in n_shots.split(',')]

    # exect features
    encoder.set_train(False)
    features_temp = np.zeros(
        (loader.get_dataset_size()*loader.get_batch_size(), encoder.num_features), dtype=np.float32)
    labels_temp = np.zeros(loader.get_dataset_size()*loader.get_batch_size(), dtype=np.int64)
    start_idx = 0
    for _, batchs in enumerate(loader):
        images, labels = batchs[0], batchs[1]
        features = encoder(images)
        features /= ms.numpy.norm(features, axis=-1, keepdims=True)
        if power_norm: features = features ** 0.5
        bsz = features.shape[0]
        features_temp[start_idx:start_idx+bsz, :] = features.asnumpy()
        labels_temp[start_idx:start_idx+bsz] = labels.asnumpy()
        start_idx += bsz
    features_temp = features_temp[:(loader.get_dataset_size()-1)*loader.get_batch_size()+bsz]
    labels_temp = labels_temp[:(loader.get_dataset_size()-1)*loader.get_batch_size()+bsz]

    # few-shot evaluation
    catlocs = [np.argwhere(labels_temp == c).reshape(-1) for c in range(labels_temp.max() + 1)]

    def get_select_index(n_cls, n_samples):
        episode = []
        classes = np.random.choice(len(catlocs), n_cls, replace=False)
        episode = [np.random.choice(catlocs[c], n_samples, replace=False) for c in classes]
        return np.concatenate(episode).reshape((n_cls, n_samples))

    accs = {}
    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []
        for _ in range(n_tasks):
            select_idx = get_select_index(n_way, n_shot+n_query)
            sup_idx = select_idx[:, :n_shot].reshape(-1)
            qry_idx = select_idx[:, n_shot:].reshape(-1)
            sup_f, qry_f = features_temp[sup_idx].reshape(n_way*n_shot, -1), features_temp[qry_idx]
            sup_y = np.arange(n_way)[:, None].repeat(n_shot, 1).reshape(-1)
            qry_y = np.arange(n_way)[:, None].repeat(n_query, 1).reshape(-1)
            if classifier == 'LR':
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
            elif classifier == 'SVM':
                clf = LinearSVC(C=1.0)
            clf.fit(sup_f, sup_y)
            qry_pred = clf.predict(qry_f)
            acc = metrics.accuracy_score(qry_y, qry_pred)
            accs[f'{n_shot}-shot'].append(acc)
    for n_shot in n_shots:
        acc = np.array(accs[f'{n_shot}-shot'])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])
        print('classifier: {}, power_norm: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, power_norm, n_way, n_shot, mean*100, c95*100))
    return accs
