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
""" LR-EGAN model """

import random
import time
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
import mindspore.ops as ops
from mindspore.common import initializer as init
import mindspore.dataset as ds
from . import utils
#from .utils import active_sampling_V1, save_weights, prepare_z_y, load_weights
from . import losses
#from .losses import  loss_dis_fake_ave_weights, loss_dis_real_ave_weights, loss_dis_fake_ave_weights_LR, loss_dis_real_ave_weights_LR
from . import pyod_utils
#from pyod_utils import AUC_and_Gmean, AUC_and_PR



class DatasetGenerator:
    ''' dataset generator '''
    def __init__(self, data_x, data_y):
        self.data = data_x
        self.label = data_y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class MyLoss(nn.LossBase):
    ''' my loss '''
    def __init__(self, args, loss_fn, reduction="mean"):
        super(MyLoss, self).__init__(reduction)
        self.loss_fn = loss_fn
        self.args = vars(args)

    def construct(self, output, out_category, label, detector_index,
                  weights_rf=None, weights_cat=None, ref_weight=None):
        if self.args['LR_flag']:
            loss, loss_cat, weights_rf, weights_cat = self.loss_fn(
                output, out_category, label, detector_index, weights_rf, weights_cat, ref_weight)
        else:
            loss, loss_cat, weights_rf, weights_cat = self.loss_fn(
                output, out_category, label, detector_index, weights_rf, weights_cat)
        sum_loss = loss+loss_cat
        return self.get_loss(sum_loss), weights_rf, weights_cat


class MyWithLossCell(nn.Cell):
    ''' loss cell '''
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label, weights_rf=None, weights_cat=None):
        output, out_category = self.backbone(data, label)
        loss, weights_rf, weights_cat = self.loss_fn(
            output, out_category, label, weights_rf, weights_cat)
        self.weights_rf = weights_rf
        self.weights_cat = weights_cat
        return loss

    def backbone_network(self):
        return self.backbone


class MyDetectorWithLossCell(nn.Cell):
    ''' detector with loss cell '''
    def __init__(self, Dnet, loss_fn_real, loss_fn_fake, args):
        super(MyDetectorWithLossCell, self).__init__(auto_prefix=False)
        self.args = vars(args)
        self.Dnet = Dnet
        self.loss_fn_fake = loss_fn_fake
        self.loss_fn_real = loss_fn_real
        self.weights_rf_real = Parameter(
            ms.numpy.zeros((args.batch_size, 1), mstype.float32))
        self.weights_rf_fake = Parameter(
            ms.numpy.zeros((args.batch_size, 1), mstype.float32))
        self.weights_cat_real = Parameter(
            ms.numpy.zeros((args.batch_size, 1), mstype.float32))
        self.weights_cat_fake = Parameter(
            ms.numpy.zeros((args.batch_size, 1), mstype.float32))
        self.loss = Parameter(Tensor(0.0, mstype.float32))
        self.print = ops.Print()

    def construct(self, real_x, real_y, real_y_true, fake_x, fake_y,\
detector_index, weights_rf_real=None, weights_cat_real=None,\
weights_rf_fake=None, weights_cat_fake=None, out_real_category_list=None,\
out_fake_category_list=None):
        ''' forward '''
        # real_loss
        output, out_category = self.Dnet(real_x, real_y)

        if self.args['LR_flag']:
            out_real_category_list_2 = ms.numpy.tile(
                out_category, (1, self.args['ensemble_num']))
            out_real_category_ref_rate = ((ms.numpy.absolute((out_real_category_list - out_real_category_list_2)))
                                          <= 0.1).sum(1) * (self.args['refurbish_rate']/self.args['ensemble_num'])
            loss_real, weights_rf_real, weights_cat_real = self.loss_fn_real(\
output, out_category, real_y, detector_index, weights_rf_real, weights_cat_real, out_real_category_ref_rate)

        else:
            loss_real, weights_rf_real, weights_cat_real = self.loss_fn_real(
                output, out_category, real_y, detector_index, weights_rf_real, weights_cat_real)

        self.weights_rf_real = weights_rf_real
        self.weights_cat_real = weights_cat_real

        # fake loss
        output, out_category = self.Dnet(fake_x, fake_y)

        if self.args['LR_flag']:
            out_fake_category_list_2 = ms.numpy.tile(out_category, (1, self.args['ensemble_num']))
            out_fake_category_ref_rate = ((ms.numpy.absolute((out_fake_category_list - out_fake_category_list_2)))\
<= 0.1).sum(1) * (self.args['refurbish_rate']/self.args['ensemble_num'])
            loss_fake, weights_rf_fake, weights_cat_fake = self.loss_fn_fake(\
output, out_category, fake_y, detector_index, weights_rf_fake, weights_cat_fake, out_fake_category_ref_rate)

        else:
            loss_fake, weights_rf_fake, weights_cat_fake = self.loss_fn_fake(\
                output, out_category, fake_y, detector_index, weights_rf_fake, weights_cat_fake)

        self.weights_rf_fake = weights_rf_fake
        self.weights_cat_fake = weights_cat_fake

        loss = loss_real+loss_fake
        self.loss = loss

        return loss

    def backbone_network(self):
        return self.Dnet

class MyGeneratorWithLossCell(nn.Cell):
    '''generator with loss cell '''
    def __init__(self, args, Gnet, Dnet_list, loss_fn, ensemble_num,):
        super(MyGeneratorWithLossCell, self).__init__(auto_prefix=False)
        self.Gnet = Gnet
        self.Dnet_list = Dnet_list
        self.loss_fn = loss_fn
        self.ensemble_num = ensemble_num
        self.loss = Parameter(Tensor(0.0, mstype.float32))
        self.args = vars(args)

    def construct(self, z, fake_y, weights_rf=None, weights_cat=None, out_category_list=None):
        ''' forward '''
        gen_loss = 0
        fake_x = self.Gnet(z, fake_y)
        for i in range(self.ensemble_num):
            output, out_category = self.Dnet_list[i](fake_x, fake_y)

            if self.args['LR_flag']:
                out_category_list_2 = ms.numpy.tile(
                    out_category, (1, self.args['ensemble_num']))
                out_category_ref_rate = ((ms.numpy.absolute((out_category_list - out_category_list_2))) <= 0.1).sum(
                    1) * (self.args['refurbish_rate']/self.args['ensemble_num'])
                loss, _, _ = self.loss_fn(
                    output, out_category, fake_y, i, weights_rf, weights_cat, out_category_ref_rate)
            else:
                loss, weights_rf, weights_cat = self.loss_fn(
                    output, out_category, fake_y, i, weights_rf, weights_cat)
            gen_loss += loss

        self.loss = gen_loss
        return gen_loss

    def backbone_network(self):
        return self.Gnet, self.Dnet

class MyTrainStep(nn.TrainOneStepCell):
    ''' train step '''
    def __init__(self, network, optimizer):
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label, weights_rf=None, weights_cat=None):
        weights = self.weights
        loss = self.network(data, label, weights_rf, weights_cat)
        grads = self.grad(self.network, weights)(
            data, label, weights_rf, weights_cat)
        return loss, self.optimizer(grads)


class Generator(nn.Cell):
    ''' generator '''
    def __init__(self, dim_z=64, hidden_dim=128, output_dim=128, n_classes=2, hidden_number=1,
                 init_type='normal'):
        super(Generator, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.init_type = init_type
        self.shared_dim = dim_z//2

        self.shared = nn.Embedding(n_classes, self.shared_dim)
        self.input_fc = nn.Dense(dim_z+self.shared_dim, self.hidden_dim)

        self.output_fc = nn.Dense(self.hidden_dim, output_dim)

        self.model = nn.SequentialCell([self.input_fc,
                                        nn.ReLU()])

        for _ in range(hidden_number):
            middle_fc = nn.Dense(self.hidden_dim, self.hidden_dim)
            self.model.append(middle_fc)
            self.model.append(nn.ReLU())

        self.model.append(self.output_fc)
        self.init_weights()

    def init_weights(self, init_type='normal', init_gain=0.02):
        ''' init weights of generator '''
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense)):
                if init_type == 'normal':
                    cell.weight.set_data(init.initializer(
                        init.Normal(init_gain), cell.weight.shape))
                elif init_type == 'xavier':
                    cell.weight.set_data(init.initializer(
                        init.XavierUniform(init_gain), cell.weight.shape))
                elif init_type == 'KaimingUniform':
                    cell.weight.set_data(init.initializer(
                        init.HeUniform(init_gain), cell.weight.shape))
                elif init_type == 'constant':
                    cell.weight.set_data(
                        init.initializer(0.001, cell.weight.shape))
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
            elif isinstance(cell, nn.GroupNorm):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape))

    def construct(self, z, y):
        y = self.shared(y)
        op = ops.Concat(axis=1)
        h = op((z, y))
        h = self.model(h)
        return h


class Discriminator(nn.Cell):
    ''' discriminator '''
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=1, n_classes=2,
                 hidden_number=1, init_type='normal', activation_func="relu"):
        super(Discriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.init_type = init_type
        self.activation_func = activation_func
        self.which_linear = nn.Dense
        self.which_embedding = nn.Embedding

        self.input_fc = self.which_linear(input_dim, self.hidden_dim)
        self.output_fc = self.which_linear(self.hidden_dim, output_dim)
        self.output_category = nn.SequentialCell(self.which_linear(self.hidden_dim, output_dim),
                                                 nn.Sigmoid())
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.hidden_dim)
        self.model = nn.SequentialCell(self.input_fc,
                                       nn.ReLU())

        for _ in range(hidden_number):
            middle_fc = self.which_linear(self.hidden_dim, self.hidden_dim)
            self.model.append(middle_fc)
            if self.activation_func == "tanh":
                self.model.append(nn.Tanh())
            elif self.activation_func == "sigmoid":
                self.model.append(nn.Sigmoid())
            else:
                self.model.append(nn.ReLU())
        self.init_weights()

    def init_weights(self, init_type='normal', init_gain=0.02):
        ''' init weights of discriminator '''
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense)):
                if init_type == 'normal':
                    cell.weight.set_data(init.initializer(
                        init.Normal(init_gain), cell.weight.shape))
                elif init_type == 'xavier':
                    cell.weight.set_data(init.initializer(
                        init.XavierUniform(init_gain), cell.weight.shape))
                elif init_type == 'KaimingUniform':
                    cell.weight.set_data(init.initializer(
                        init.HeUniform(init_gain), cell.weight.shape))
                elif init_type == 'constant':
                    cell.weight.set_data(
                        init.initializer(0.001, cell.weight.shape))
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
            elif isinstance(cell, nn.GroupNorm):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape))

    def construct(self, x, y=None, mode=0):
        ''' forward '''
        # mode 0: train the whole discriminator network
        sum_ = ops.ReduceSum(keep_dims=True)
        if mode == 0:
            h = self.model(x)
            out = self.output_fc(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out_real_fake = out + sum_((self.embed(y) * h), 1)
            out_category = self.output_category(h)
            return out_real_fake, out_category
        # mode 1: train self.output_fc, only classify whether an input is fake or real
        if mode == 1:
            h = self.model(x)
            out = self.output_fc(h)
            return out
        # mode 2: train self.output_category, used in fine_tunning stage

        h = self.model(x)
        out = self.output_category(h)
        return out


class CB_GAN(nn.Cell):
    ''' CB_GAN model '''
    def __init__(self, args, data_x, data_y, test_x, test_y, val_x, val_y, visualize=False):
        super(CB_GAN, self).__init__()

        self.print = ops.Print()
        z, _ = utils.prepare_z_y(20, args.dim_z, 2)
        self.noise = z
        self.args = args
        self.feature_size = data_x.shape[1]
        self.data_size = data_x.shape[0]
        self.batch_size = min(args.batch_size, self.data_size)
        args.batch_size = self.batch_size
        self.hidden_dim = self.feature_size*2
        self.dim_z = args.dim_z

        self.data_x = data_x
        self.data_y = data_y
        self.test_x = test_x
        self.test_y = test_y
        self.val_x = val_x
        self.val_y = val_y

        traindata_generator = DatasetGenerator(data_x, data_y)
        testdata_generator = DatasetGenerator(test_x, test_y)
        valdata_generator = DatasetGenerator(val_x, val_y)

        self.dataset_train = ds.GeneratorDataset(traindata_generator,\
["data", "label"], shuffle=False).batch(self.batch_size, drop_remainder=True)
        self.dataset_test = ds.GeneratorDataset(testdata_generator,\
["data", "label"], shuffle=False).batch(self.batch_size, drop_remainder=True)
        self.dataset_val = ds.GeneratorDataset(valdata_generator,\
["data", "label"], shuffle=False).batch(self.batch_size, drop_remainder=True)

        self.iterations = 0
        self.visualize = visualize

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        ms.set_seed(manualSeed)

        # 1: prepare Generator
        self.netG = Generator(dim_z=self.dim_z, hidden_dim=self.hidden_dim, output_dim=self.feature_size, n_classes=2,
                              hidden_number=args.gen_layer, init_type=args.init_type)
        self.optimizerG = nn.Adam(self.netG.trainable_params(
        ), learning_rate=args.lr_g, beta1=0.001, beta2=0.99)

        # 2: create ensemble of discriminator
        self.NetD_Ensemble = []
        self.opti_Ensemble = []
        self.trainOneStep_DEnsemble = []

        if args.LR_flag:
            loss_func_fake = MyLoss(args, losses.loss_dis_fake_ave_weights_LR)
            loss_func_real = MyLoss(args, losses.loss_dis_real_ave_weights_LR)
        else:
            loss_func_fake = MyLoss(args, losses.loss_dis_fake_ave_weights)
            loss_func_real = MyLoss(args, losses.loss_dis_real_ave_weights)

        lr_ds = np.random.rand(args.ensemble_num) * \
            (args.lr_d*5-args.lr_d)+args.lr_d  # learning rate
        for index in range(args.ensemble_num):
            netD = Discriminator(input_dim=self.feature_size,\
hidden_dim=self.hidden_dim, output_dim=1, n_classes=2,\
hidden_number=args.dis_layer, init_type=args.init_type, activation_func=args.dis_activation_func)
            optimizerD = nn.Adam(netD.trainable_params(
            ), learning_rate=lr_ds[index], beta1=0.001, beta2=0.99)

            net_with_criterion = MyDetectorWithLossCell(
                netD, loss_func_real, loss_func_fake, self.args)
            train_onestep_Dcell = nn.TrainOneStepCell(
                net_with_criterion, optimizerD)

            self.NetD_Ensemble += [netD]
            self.opti_Ensemble += [optimizerD]
            self.trainOneStep_DEnsemble += [train_onestep_Dcell]

        if args.LR_flag:
            loss_func = MyLoss(args, losses.loss_dis_real_ave_weights_LR)
        else:
            loss_func = MyLoss(args, losses.loss_dis_real_ave_weights)
        net_with_criterion = MyGeneratorWithLossCell(
            args, self.netG, self.NetD_Ensemble, loss_func, self.args.ensemble_num)
        self.train_onestep_Gcell = nn.TrainOneStepCell(
            net_with_criterion, self.optimizerG)

    def fit(self):
        ''' fit '''
        if self.args.LR_flag:
            weights_root = f'./trueRate{self.args.noiseLabelRate}_weights_' + \
                self.args.data_name
        else:
            weights_root = 'weights_'+self.args.data_name

        log_dir = os.path.join('./log/', self.args.data_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        z, y = utils.prepare_z_y(self.batch_size, self.dim_z, 2)
        # Start iteration
        Best_Measure_Recorded = -1
        for epoch in range(self.args.max_epochs):
            train_AUC, train_prn, train_rec, train_gmean,\
            test_auc, test_prn, test_rec, test_gmean, val_auc, val_prn, val_rec, val_gmean = self.train_one_epoch(\
            self.dataset_train, self.dataset_val, z, y, 1, self.netG, self.optimizerG,\
            self.train_onestep_Gcell, self.NetD_Ensemble, self.opti_Ensemble,\
            self.trainOneStep_DEnsemble, epoch)
            if test_gmean*test_auc >= Best_Measure_Recorded:
                Best_Measure_Recorded = test_gmean*test_auc
                states = {
                    'epoch': epoch,
                    'max_auc': val_auc
                }
                utils.save_weights(self.netG, self.NetD_Ensemble, epoch, weights_root=weights_root)

            if self.args.print:
                print(('Training for epoch %d: Train_AUC=%.4f Train_recall=%.4f'+\
'Train_prn=%.4f Train_Gmean=%.4f Test_AUC=%.4f Test_recall=%.4f Test_prn=%.4f'+\
'Test_Gmean=%.4f Val_AUC=%.4f Val_recall=%.4f Val_prn=%.4f Val_Gmean=%.4f')%(\
epoch + 1, train_AUC, train_rec, train_prn, train_gmean, test_auc, test_rec,\
test_prn, test_gmean, val_auc, val_rec, val_prn, val_gmean))

        self.netG, self.NetD_Ensemble = utils.load_weights(
            self.netG, self.NetD_Ensemble, weights_root=weights_root, epoch=states["epoch"])

    def predict(self, test_x, dis_Ensemble=None):
        ''' predict '''
        assert self.args.ensemble_num != 0
        if dis_Ensemble is None:
            final_pt = None
            for i in range(self.args.ensemble_num):
                pt = self.NetD_Ensemble[i](test_x, mode=2)
                if i == 0:
                    final_pt = pt  # .detach()
                else:
                    final_pt += pt
            final_pt /= self.args.ensemble_num
            final_pt = final_pt.view(final_pt.shape[0],)

            ps = final_pt.asnumpy()
            return ps

        final_pt = None
        for i in range(self.args.ensemble_num):
            pt = dis_Ensemble[i](test_x, mode=2)
            if i == 0:
                final_pt = pt
            else:
                final_pt += pt
        final_pt /= self.args.ensemble_num
        final_pt = final_pt.view(final_pt.shape[0],)
        ps = final_pt
        return ps

    def train_one_epoch(self, dataset_train, dataset_val, z, y, min_label, netG, optimizerG,\
trainOneStep_G, NetD_Ensemble, opti_Ensemble, trainOneStep_DEnsemble, epoch=1):
        '''  train discriminator & generator for one specific spoch '''
        start_time = time.time()
        num_batches = dataset_train.get_dataset_size()
        train_iterator = dataset_train.create_tuple_iterator()
        val_iterator = dataset_val.create_tuple_iterator()
        while True:
            try:
                # step 1: train the ensemble of discriminator
                real_x, real_y = next(train_iterator)
                _, real_y_true = next(val_iterator)
                real_x, real_y = real_x.astype(mstype.float32), real_y.astype(mstype.int32)
                real_y_true = real_y_true.astype(mstype.int32)
                self.iterations += 1
                real_weights, fake_weights, real_cat_weights, fake_cat_weights = None, None, None, None
                z_sample, y_sample = z.sample_(), y.sample_()
                generated_x = netG(z_sample, y_sample)
                losses_, dis_loss, gen_loss = [], 0, 0
                # select p% of the training data, label them
                if self.args.active_rate != 1:
                    real_x_selected, real_y_selected, _ = utils.active_sampling_V1(
                        self.args, real_x, real_y, NetD_Ensemble, need_sample=(epoch > 0))
                else:
                    real_x_selected, real_y_selected = real_x, real_y

                if self.args.LR_flag:

                    out_real_category_list, out_fake_category_list = ms.numpy.zeros((1, 1)), ms.numpy.zeros((1, 1))
                    out_category_list = ms.numpy.zeros((1, 1))

                    for i in range(self.args.ensemble_num):
                        netD = NetD_Ensemble[i]
                        _, out_real_category = netD(
                            real_x_selected, real_y_selected)
                        _, out_fake_category = netD(generated_x, y_sample)
                        _, out_category = netD(generated_x, y_sample)

                        if i == 0:
                            out_real_category_list, out_fake_category_list,\
out_category_list = out_real_category, out_fake_category, out_category

                        else:
                            out_real_category_list = ms.numpy.concatenate(
                                (out_real_category_list, out_real_category), axis=1)
                            out_fake_category_list = ms.numpy.concatenate(
                                (out_fake_category_list, out_fake_category), axis=1)
                            out_category_list = ms.numpy.concatenate(
                                (out_category_list, out_category), axis=1)

                for i in range(self.args.ensemble_num):
                    train_onestep_D = trainOneStep_DEnsemble[i]
                    if self.args.LR_flag:
                        train_onestep_D(real_x_selected, real_y_selected, real_y_true, generated_x, y_sample,\
i, real_weights, real_cat_weights, fake_weights, fake_cat_weights, out_real_category_list, out_fake_category_list)
                    else:
                        train_onestep_D(real_x_selected, real_y_selected, real_y_true, generated_x,\
y_sample, i, real_weights, real_cat_weights, fake_weights, fake_cat_weights)
                    sum_loss = Tensor(train_onestep_D.network.loss.data)
                    real_weights = Tensor(
                        train_onestep_D.network.weights_rf_real)
                    real_cat_weights = Tensor(
                        train_onestep_D.network.weights_cat_real)
                    fake_weights = Tensor(
                        train_onestep_D.network.weights_rf_fake)
                    fake_cat_weights = Tensor(
                        train_onestep_D.network.weights_cat_fake)

                    dis_loss += sum_loss
                    losses_ += [sum_loss]

                print("dis_loss", dis_loss)

                # step 2: train the generator
                z_sample = z.sample_()
                y_sample = y.sample_()
                generated_x = netG(z_sample, y_sample)
                gen_loss, gen_weights, gen_cat_weights = 0, None, None

                if self.args.LR_flag:
                    trainOneStep_G(z_sample, y_sample, gen_weights, gen_cat_weights, out_category_list)
                else:
                    trainOneStep_G(z_sample, y_sample, gen_weights, gen_cat_weights)

                gen_loss = Tensor(train_onestep_D.network.loss)
                print("gen_loss", gen_loss)

            except StopIteration:
                break

        print(f'Time cost per step is {(time.time() - start_time)/(self.args.ensemble_num*num_batches)} seconds\n')

        auc, prn, rec, gmean, test_auc, test_prn, test_rec, test_gmean = 0, 0, 0, 0, 0, 0, 0, 0
        val_auc, val_prn, val_rec, val_gmean = 0, 0, 0, 0

        if self.args.print:

            test_y_scores = self.predict(Tensor(self.test_x.astype(np.float32)), NetD_Ensemble)
            if self.args.LR_flag:
                test_auc, test_prn, test_rec, test_gmean = pyod_utils.AUC_and_PR(self.test_y, test_y_scores)
            else:
                test_auc, test_prn, test_rec, test_gmean = pyod_utils.AUC_and_Gmean(self.test_y, test_y_scores)

        return auc, prn, rec, gmean, test_auc, test_prn, test_rec, test_gmean, val_auc, val_prn, val_rec, val_gmean


def get_gpu_info():
    print('==== GPU:')
    gpu_status = os.popen('nvidia-smi').read()
    print(gpu_status)
