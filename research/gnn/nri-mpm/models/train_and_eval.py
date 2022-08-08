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

import time
from mindspore import nn, Model, Tensor
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.communication import get_rank, get_group_size
from mindspore import dataset as ds
from models.nri import NRIModel
from utils.metrics import edge_accuracy, nll_gaussian, cross_entropy, kl_divergence
from utils.helper import transpose
from utils.logger import create_logger

class DatasetGenerator:
    def __init__(self, inputs):
        self.adj = inputs[0]
        self.states = inputs[1]

    def __getitem__(self, index):
        return self.adj[index], self.states[index]

    def __len__(self):
        return len(self.adj)


class MyLoss(nn.LossBase):
    def __init__(self, size: int, reg: int, reduction="none"):
        super(MyLoss, self).__init__(reduction)
        self.size = size
        self.reg = reg

    def construct(self, output: Tensor, prob: Tensor, states: Tensor):
        prob = prob.swapaxes(0, 1)
        # reconstruction loss and the KL-divergence
        loss_nll = nll_gaussian(output, states[:, 1:], 5e-5)
        loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
        loss = loss_nll + loss_kl
        # impose the soft symmetric constraint by adding a regularization term
        if self.reg > 0:
            # transpose the relation distribution
            prob_hat = transpose(prob, self.size)
            loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
            loss = loss + loss_sym * self.reg
        return self.get_loss(loss * 1e-2)


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, adj: Tensor, states: Tensor):
        output, prob = self.backbone(states, states)
        return self.loss_fn(output, prob, states)


class MyCallback(Callback):
    def __init__(self, args, eval_model, eval_ds, logger):
        super(MyCallback, self).__init__()
        self.args = args
        self.logger = logger
        self.eval_model = eval_model
        self.eval_ds = eval_ds
        self.eval_best = 0

    def begin(self, run_context):
        """Called once before the network executing."""
        if not self.args.parallel or get_rank() == 0:
            self.logger.info(str(self.args))
        self.begin_time = time.time() * 1000

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        self.train_sample_num = 0
        self.train_loss_total = 0

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        cb_params = run_context.original_args()
        result = self.eval_model.eval(self.eval_ds, dataset_sink_mode=True)
        mse, acc, eval_loss_total = result["eval"]
        if not self.args.parallel or get_rank() == 0:
            self.logger.info("******************** {} epoch {:03d} ********************" \
                             .format(self.args.dataset, cb_params.cur_epoch_num))
            self.logger.info("train_loss: {:.6f}".format((self.train_loss_total / self.train_sample_num).asnumpy()))
            self.logger.info("eval_step: 10, mse: {:.3e}, acc: {:.4f}, eval_loss: {:.6f}" \
                             .format(mse, acc, eval_loss_total))
        if acc >= self.eval_best:
            if not self.args.parallel or get_rank() == 0:
                save_checkpoint(save_obj=cb_params.train_network,
                                ckpt_file_name="checkpoints/{}.ckpt".format(self.args.dataset))
            self.eval_best = acc

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        self.train_sample_num += self.args.batch_size
        self.train_loss_total += cb_params.net_outputs

    def end(self, run_context):
        """Called once after network training."""
        cb_params = run_context.original_args()
        training_time = time.time() * 1000 - self.begin_time
        if not self.args.parallel or get_rank() == 0:
            self.logger.info("********************** Time Statistics **********************")
            self.logger.info("The training process took {:.3f} ms.".format(training_time))
            self.logger.info("Each step took an average of {:.3f} ms" \
                             .format(training_time / cb_params.cur_step_num))
            self.logger.info("************************** End Training **************************")

class EvalCell(nn.Cell):
    def __init__(self, args, network, loss):
        super(EvalCell, self).__init__(auto_prefix=False)
        self.step = args.train_step
        self.network = network
        self.loss = loss

    def construct(self, adj: Tensor, states: Tensor):
        states_enc = states[:, :self.step, :, :]
        states_dec = states[:, -self.step:, :, :]
        target = states_dec[:, 1:]
        output, prob = self.network(states_enc, states_dec, hard=True)
        eval_loss_val = self.loss(output, prob, states_dec)
        return output, target, prob, adj, eval_loss_val


class EvalMetric(nn.Metric):
    def __init__(self, args):
        super(EvalMetric, self).__init__()
        self.args = args
        self.clear()

    def clear(self):
        self.N = 0
        self.acc_l = []
        self.mse_l = []
        self.loss_l = []

    def update(self, *inputs):
        output, target, prob, adj, loss_val = inputs
        prob = prob.swapaxes(0, 1)
        scale = target.shape[0] / self.args.batch_size
        self.mse_l.append(scale * nn.MSELoss()(output, target))
        self.acc_l.append(scale * edge_accuracy(prob, adj))
        self.loss_l.append(scale * loss_val)
        self.N += scale

    def eval(self):
        mse = sum(self.mse_l) / self.N
        acc = sum(self.acc_l) / self.N
        acc = max(acc, 1 - acc)
        loss = sum(self.loss_l) / self.N
        return mse.asnumpy(), acc.asnumpy(), loss.asnumpy()


class TestCell(nn.Cell):
    def __init__(self, args, network):
        super(TestCell, self).__init__(auto_prefix=False)
        self.train_step = args.train_step
        self.test_step = args.test_step
        self.network = network

    def construct(self, adj: Tensor, states: Tensor):
        states_enc = states[:, :self.train_step, :, :]
        states_dec = states[:, -self.train_step:, :, :]
        target = states_dec[:, 1:]
        output, prob = self.network(states_enc, states_dec, hard=True, M=self.test_step)

        test_states_dec = states[:, self.train_step:self.train_step+self.test_step+1, :, :]
        test_target = test_states_dec[:, 1:]
        test_output, _ = self.network(states_enc, test_states_dec, hard=True, M=self.test_step)
        return output, target, prob, adj, test_output, test_target


class TestMetric(nn.Metric):
    def __init__(self, args):
        super(TestMetric, self).__init__()
        self.args = args
        self.clear()

    def clear(self):
        self.N = 0
        self.acc_l = []
        self.mse_l = []
        self.multi_mse_l = []

    def update(self, *inputs):
        output, target, prob, adj, test_output, test_target = inputs
        prob = prob.swapaxes(0, 1)
        scale = target.shape[0] / self.args.batch_size
        self.mse_l.append(scale * nn.MSELoss()(output, target))
        self.acc_l.append(scale * edge_accuracy(prob, adj))
        self.multi_mse_l.append(scale * ((test_output - test_target) ** 2).mean(axis=(0, 2, -1)))
        self.N += scale

    def eval(self):
        mse = sum(self.mse_l) / self.N
        acc = sum(self.acc_l) / self.N
        acc = max(acc, 1 - acc)
        multi_mse = sum(self.multi_mse_l) / self.N
        msteps = ", ".join(["{:.3e}".format(mse.asnumpy()) for mse in multi_mse])
        return mse.asnumpy(), acc.asnumpy(), msteps

class TrainWrapper:
    def __init__(self, args, dataset, es):
        self.args = args
        self.logger = create_logger()

        self.train_ds = self.load_data(dataset["train"], args.batch_size, args.parallel)
        self.eval_ds = self.load_data(dataset["val"], args.batch_size, args.parallel, shuffle=False)

        self.network = NRIModel(args.dim, args.hidden, args.edge_type, args.drop_out, args.skip, args.size, es)
        self.loss = MyLoss(args.size, args.reg)
        self.net_with_criterion = MyWithLossCell(self.network, self.loss)
        self.opt = nn.Adam(self.network.trainable_params(),
                           learning_rate=nn.ExponentialDecayLR(learning_rate=args.lr,
                                                               decay_rate=0.5,
                                                               decay_steps=self.train_ds.get_dataset_size() * 200,
                                                               is_stair=True))

        self.eval_cell = EvalCell(self.args, self.network, self.loss)
        self.eval_metric = EvalMetric(args)

        self.model = Model(network=self.net_with_criterion, optimizer=self.opt,
                           metrics={"eval": self.eval_metric}, eval_network=self.eval_cell)

        self.callback = MyCallback(self.args, self.model, self.eval_ds, self.logger)
        self.time_monitor = TimeMonitor()

    def load_data(self, inputs, batch_size: int, parallel, shuffle: bool = True):
        """
        Return a dataloader given the input and the batch size.
        """
        num_shards = get_group_size() if parallel else None
        shard_id = get_rank() if parallel else None

        dataset_generator = DatasetGenerator(inputs)
        dataset = ds.GeneratorDataset(dataset_generator, ["adj", "state"], shuffle=shuffle,
                                      num_shards=num_shards, shard_id=shard_id)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    def train(self):
        if not self.args.parallel or get_rank() == 0:
            callbacks = [self.callback, self.time_monitor]
        else:
            callbacks = [self.callback]
        self.model.train(epoch=self.args.epochs, train_dataset=self.train_ds,
                         callbacks=callbacks, dataset_sink_mode=True)


class EvalWrapper:
    def __init__(self, args, dataset, es):
        self.args = args
        self.logger = create_logger()
        self.ckpt_file = args.ckpt_file

        self.eval_ds = self.load_data(dataset["test"], args.batch_size)

        self.network = NRIModel(args.dim, args.hidden, args.edge_type, args.drop_out, args.skip, args.size, es)

        self.eval_cell = TestCell(self.args, self.network)
        self.eval_metric = TestMetric(args)

        self.model = Model(network=self.network, optimizer=None,
                           metrics={"eval": self.eval_metric}, eval_network=self.eval_cell)

    def load_data(self, inputs, batch_size: int, shuffle: bool = False):
        """
        Return a dataloader given the input and the batch size.
        """
        dataset_generator = DatasetGenerator(inputs)
        dataset = ds.GeneratorDataset(dataset_generator, ["adj", "state"], shuffle=shuffle)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    def eval(self):
        param_dict = load_checkpoint(self.ckpt_file)
        load_param_into_net(self.network, param_dict)

        self.logger.info("************************** Evaluating **************************")
        result = self.model.eval(self.eval_ds, dataset_sink_mode=True)
        mse, acc, multi_mse = result["eval"]
        self.logger.info("test_step: {}, mse: {:.3e}, acc: {:.4f}".format(self.args.test_step, mse, acc))
        self.logger.info("multi_mse: "+multi_mse)
        self.logger.info("************************** End Evaluation **************************")
