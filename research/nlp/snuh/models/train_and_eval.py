# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import time
import pickle
from mindspore import nn, Model
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.train.callback import Callback, TimeMonitor
from models.snuh import SNUH
from utils.data import LabeledDocuments
from utils.logger import Logger
from utils.evaluation import compute_retrieval_precision

class MyCallback(Callback):
    '''Callback base class'''
    def __init__(self, hparams, logger, model, val_loader) -> None:
        super(MyCallback, self).__init__()
        self.hparams = hparams
        self.logger = logger
        self.model = model
        self.val_loader = val_loader
        self.best_val_prec = float('-inf')
        self.bad_epochs = 0

    def begin(self, run_context):
        '''Called once before the network executing.'''
        self.logger.log('********************** Arguments **********************')
        for param in vars(self.hparams):
            self.logger.log('{} : {}'.format(param, getattr(self.hparams, param)))
        self.begin_time = time.time() * 1000

    def epoch_end(self, run_context):
        '''Called after each epoch finished.'''
        cb_params = run_context.original_args()
        result = self.model.eval(self.val_loader, dataset_sink_mode=True)
        val_prec = result['prec']
        self.logger.log('*' * 60)
        self.logger.log('End of epoch {:d}, val perf: {:.2f}'.format(cb_params.cur_epoch_num, val_prec))
        if val_prec > self.best_val_prec:
            self.best_val_prec = val_prec
            self.bad_epochs = 0
            self.logger.log('Best model so far, save model!')
            save_checkpoint(save_obj=cb_params.train_network,
                            ckpt_file_name='checkpoints/{}.ckpt'.format(self.hparams.model_path))
            pickle.dump(self.hparams, open('checkpoints/{}.hpar'.format(self.hparams.model_path), 'wb'))
        else:
            self.bad_epochs += 1
            self.logger.log('Bad epoch {:d}.'.format(self.bad_epochs))

        if self.bad_epochs > self.hparams.num_bad_epochs:
            run_context.request_stop()

    def end(self, run_context):
        '''Called once after network training.'''
        cb_params = run_context.original_args()
        training_time = time.time() * 1000 - self.begin_time
        self.logger.log('********************** Time Statistics **********************')
        self.logger.log('The training process took {:.3f} ms.'.format(training_time))
        self.logger.log('Each step took an average of {:.3f} ms' \
                            .format(training_time / cb_params.cur_step_num))
        self.logger.log('************************** End Training **************************')

class EvalCell(nn.Cell):
    def construct(self, data, label):
        return data, label

class EvalMetric(nn.Metric):
    def __init__(self, hparams, network, database_loader, val_loader):
        super(EvalMetric, self).__init__()
        self.hparams = hparams
        self.network = network
        self.database_loader = database_loader
        self.val_loader = val_loader
        self.clear()

    def clear(self):
        self.prec = 0

    def update(self, *inputs):
        self.prec = compute_retrieval_precision(self.database_loader, self.val_loader, self.network.encode_discrete,
                                                self.hparams.distance_metric, self.hparams.num_retrieve,
                                                self.hparams.num_features)

    def eval(self):
        return self.prec

class TrainWrapper:
    def __init__(self, hparams):
        self.hparams = hparams
        self._load_data()

        self.logger = Logger()
        self.network = SNUH(self.hparams, self.data.num_nodes, self.data.num_edges, self.data.vocab_size)
        self.opt = nn.Adam(params=self.network.trainable_params(), learning_rate=self.hparams.lr)

        self.eval_network = EvalCell()
        self.eval_metric = EvalMetric(hparams, self.network, self.database_loader, self.val_loader)
        self.model = Model(network=self.network, optimizer=self.opt,
                           metrics={'prec': self.eval_metric}, eval_network=self.eval_network)

        self.callback = MyCallback(hparams, self.logger, self.model, self.val_loader)
        self.time_monitor = TimeMonitor()

    def _load_data(self):
        self.data = LabeledDocuments(self.hparams.data_path, self.hparams.num_neighbors)
        self.train_loader, self.database_loader, self.val_loader, _ = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size)

    def run_training_session(self):
        self.model.train(epoch=self.hparams.epochs, train_dataset=self.train_loader,
                         callbacks=[self.callback, self.time_monitor], dataset_sink_mode=True)

class EvalWrapper:
    def __init__(self, hparams_path, ckpt_path):
        self.hparams = pickle.load(open(hparams_path, 'rb'))
        self._load_data()

        self.logger = Logger()
        self.network = SNUH(self.hparams, self.data.num_nodes, self.data.num_edges, self.data.vocab_size)
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(self.network, param_dict)

        self.eval_network = EvalCell()
        self.eval_metric = EvalMetric(self.hparams, self.network, self.database_loader, self.val_loader)
        self.test_metric = EvalMetric(self.hparams, self.network, self.database_loader, self.test_loader)
        self.model = Model(network=self.network, optimizer=None,
                           metrics={'val prec': self.eval_metric, 'test prec': self.test_metric},
                           eval_network=self.eval_network)

    def _load_data(self):
        self.data = LabeledDocuments(self.hparams.data_path, self.hparams.num_neighbors)
        _, self.database_loader, self.val_loader, self.test_loader = self.data.get_loaders(
            self.hparams.num_trees, self.hparams.alpha, self.hparams.batch_size)

    def run_eval(self):
        self.logger.log('************************** Evaluating **************************')
        result = self.model.eval(self.test_loader, dataset_sink_mode=True)
        val_prec = result['val prec']
        test_prec = result['test prec']
        self.logger.log('val precision: {:.2f}'.format(val_prec))
        self.logger.log('test precision: {:.2f}'.format(test_prec))
        self.logger.log('************************** End Evaluation **************************')
