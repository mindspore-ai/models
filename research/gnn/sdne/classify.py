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
Classification Task
"""
import argparse
import numpy
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

from src import read_node_label
from src import cfg

parser = argparse.ArgumentParser(description='Classification Task')
parser.add_argument('--embeddings', type=str, required=True, help="embeddings data path")
parser.add_argument('--node_idx', type=str, required=True, help="embedding's node index data path")
parser.add_argument('--dataset', type=str, default='WIKI',
                    choices=['WIKI', 'BLOGCATALOG', 'FLICKR', 'YOUTUBE', 'GRQC', 'NEWSGROUP'])
parser.add_argument('--label', type=str, default='', help="node group label path")
args = parser.parse_args()

class TopKRanker(OneVsRestClassifier):
    """
    core classifier
    """
    # define the predict function
    def predict(self, data, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(data))
        all_labels = []
        for ind, k in enumerate(top_k_list):
            probs_ = probs[ind, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier:
    """
    main classifier
    """
    # define the operator required
    def __init__(self, all_embs, clfy):
        self.all_embs = all_embs
        self.clfy = TopKRanker(clfy)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    # define train task
    def train(self, X_data, Y_data, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.all_embs[x] for x in X_data]
        Y_data = self.binarizer.transform(Y_data)
        self.clfy.fit(X_train, Y_data)

    # define eval task
    def evaluate(self, X_data, Y_data):
        """
        evaluate
        """
        top_k_list = [len(l) for l in Y_data]
        Y_ = self.predict(X_data, top_k_list)
        Y_data = self.binarizer.transform(Y_data)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y_data, Y_, average=average)
        results['acc'] = accuracy_score(Y_data, Y_)
        print('-------------------Classifier Result-------------------')
        print(results)
        return results

    # define predict task
    def predict(self, X_data, top_k_list):
        X_ = numpy.asarray([self.all_embs[x] for x in X_data])
        Y_data = self.clfy.predict(X_, top_k_list=top_k_list)
        return Y_data

    # define a task which contain train and eval task.
    # The dataset of these tasks was derived from the same randomly divided dataset.
    def split_train_evaluate(self, X_data, Y_data, train_precent, seed=0):
        """
        train and evaluate
        """
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X_data))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X_data)))
        X_train = [X_data[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y_data[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X_data[shuffle_indices[i]] for i in range(training_size, len(X_data))]
        Y_test = [Y_data[shuffle_indices[i]] for i in range(training_size, len(X_data))]

        self.train(X_train, Y_train, Y_data)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)

if __name__ == "__main__":
    config = cfg[args.dataset]['classify']

    embeddings = numpy.load(args.embeddings)
    node_idx = numpy.load(args.node_idx)
    embs = {}
    for i, embedding in enumerate(embeddings):
        embs[str(node_idx[i])] = embedding
    embeddings = embs
    label = args.label
    if label == '':
        label = config['label']
    X, Y = read_node_label(label, config['skip_head'])

    tr_frac = config['tr_frac']
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    clf = Classifier(all_embs=embeddings, clfy=LogisticRegression(solver='liblinear'))
    clf.split_train_evaluate(X, Y, tr_frac)
