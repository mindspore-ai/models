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
'''eval dataset with SGAE model'''
import os
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from tqdm import tqdm
from src.SGAE import SGAE
from dataloader import LoadDocumentData, LoadImageData, LoadTabularData
import mindspore as ms
from mindspore import nn
from mindspore import Model
from mindspore import ops
from mindspore.train.callback import LossMonitor, TimeMonitor, Callback
from mindspore import context, Tensor
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net
from dataset import create_dataset
from model_utils.config import config as cfg


class CustomWithEvalCell(nn.Cell):
    """ CustomWithEvalCell """
    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        scores, _, _ = self.network(data)
        return scores, label


class AUC_PR(nn.Metric):
    ''' AUC_PR '''

    def __init__(self):
        super(AUC_PR, self).__init__()
        self.clear()

    def clear(self):
        pass

    def update(self, *inputs):
        scores = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        self.auc = roc_auc_score(y, scores)
        self.pr = average_precision_score(y, scores)

    def eval(self):
        return self.auc, self.pr


best_auc = 0
best_pr = 0


class EvalCallback(Callback):
    """
    Evaluation per epoch, and save the best AUC_PR checkpoint.
    """

    def __init__(self, model, eval_ds, save_path="./"):

        global best_auc
        global best_pr

        self.model = model
        self.eval_ds = eval_ds
        self.best_auc = best_auc
        self.best_pr = best_pr
        self.save_path = save_path
        self.print = ops.Print()

    def epoch_end(self, run_context):
        ''' epoch_end '''
        global best_auc
        global best_pr

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        res = self.model.eval(self.eval_ds)
        auc = res["auc_pr"][0]
        pr = res["auc_pr"][1]
        if auc+pr > self.best_auc + self.best_pr:
            self.best_pr = pr
            self.best_auc = auc
            best_auc = auc
            best_pr = pr
            if params_.data_name not in ['reuters', '20news']:
                ms.save_checkpoint(cb_params.train_network, \
os.path.join(self.save_path, f"{data_name}_best_auc_pr_runs{run_index}.ckpt"))
            else:
                ms.save_checkpoint(cb_params.train_network, os.path.join(self.save_path, \
f"{data_name}_best_auc_pr_runs{run_index}_normal{normal_index}.ckpt"))

            self.print("the best epoch is", cur_epoch,
                       "best auc pr is", self.best_auc, self.best_pr)


class CustomWithLossCell(nn.Cell):
    ''' CustomWithLossCell '''

    def __init__(self, backbone, norm_thresh, params):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.norm_thresh = norm_thresh
        self.params = params
        self.print = ops.Print()

    def construct(self, x, label):

        scores, x_dec, _ = self._backbone(x)
        anomal_flag = recog_anomal(
            x, x_dec, self.norm_thresh).astype(mstype.int32)
        recon_error = ms.numpy.mean(ms.numpy.multiply(x - x_dec, x - x_dec))
        dist_error = self.compute_dist_error(scores, anomal_flag, self.params)
        loss = recon_error + self.params['lam_dist'] * dist_error
        return loss

    def compute_dist_error(self, scores, anomal_flag, params):
        """ compute distance error """
        ref = ms.numpy.randn((1000,))
        dev = scores - ms.numpy.mean(ref)
        inlier_loss = ms.numpy.absolute(dev)
        # outlier loss
        anomal_flag = ms.numpy.expand_dims(anomal_flag, 1)
        outlier_loss = ms.numpy.absolute(ms.numpy.maximum(
            params['a'] - scores, ms.numpy.zeros(scores.shape)))
        dist_error = ms.numpy.mean(
            (1 - anomal_flag) * inlier_loss + params['lam_out'] * anomal_flag * outlier_loss)
        return dist_error


run_index = 0
data_name = 0


def train_tabular(params):
    """ train tabular """
    global params_
    params_ = params

    for _ in range(10):
        params.np_seed += 1

        x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(
            params)
        x_train_whole = Tensor(x_train, dtype=mstype.float32)
        x_test_whole = Tensor(x_test, dtype=mstype.float32)
        y_test = y_test.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_train = y_train.astype(np.int32)
        auc = np.zeros(params.run_num)
        ap = np.zeros(params.run_num)
        global data_name
        data_name = params.data_name

        # Start Train
        for run_idx in tqdm(range(params.run_num)):
            start_time = time.time()
            global best_auc
            global best_pr
            global run_index
            run_index = run_idx
            best_auc = 0
            best_pr = 0
            model = SGAE(x_train_whole.shape[1], params.hidden_dim)
            optim = nn.Adam(model.trainable_params(), learning_rate=params.lr)

            if params.verbose and run_idx == 0:
                print(model)

            # One run
            for epoch in range(params.epochs):
                ds_train = create_dataset(x_train, y_train, params)
                ds_val = create_dataset(x_val, y_val, params, is_batch=False)

                epoch_time_start = time.time()

                # calculate norm thresh
                _, dec_train, _ = model(x_train_whole)
                norm = calculate_norm(x_train_whole, dec_train)
                norm_thresh = np.percentile(norm, params.epsilon)

                loss = 0
                recon_error = 0
                dist_error = 0

                auc_pr = AUC_PR()
                model_withloss = CustomWithLossCell(
                    model, norm_thresh, vars(params))
                eval_net = CustomWithEvalCell(model)
                if params.device == "Ascend":
                    model_withloss = Model(model_withloss, optimizer=optim, \
                                           eval_network=eval_net, metrics={'auc_pr': auc_pr}, amp_level="O3")
                else:
                    model_withloss = Model(
                        model_withloss, optimizer=optim, eval_network=eval_net, metrics={'auc_pr': auc_pr})

                eval_callback = EvalCallback(
                    model_withloss, ds_val, save_path="./saved_model")
                num_batches = ds_train.get_dataset_size()
                model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=[TimeMonitor(
                    30), LossMonitor(100), eval_callback], dataset_sink_mode=False)
                print(
                    f'Time cost per step is {(time.time() - start_time)/(num_batches)} seconds\n')

                epoch_time = time.time() - epoch_time_start

                if params.verbose:
                    if (epoch + 1) % params.print_step == 0 or epoch == 0:
                        scores, _, _ = model(x_test_whole)
                        scores = scores.asnumpy()
                        auc_ = roc_auc_score(y_test, scores)
                        ap_ = average_precision_score(y_test, scores)
                        print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                              f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, \
                                --DE_r:{dist_error*params.lam_dist:.3f},' +\
                              f'--AUC:{auc_:.3f} --AP:{ap_:.3f}')

                # Early Stop
                if params.early_stop:
                    scores, _, _ = model(x_train_whole)
                    scores = scores.asnumpy()
                    if np.mean(scores) > params.a / 2:
                        print(
                            f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                        break

            # test
            param_dict = load_checkpoint(
                f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt")
            load_param_into_net(model, param_dict)
            scores, _, _ = model(x_test_whole)
            scores = scores.asnumpy()
            auc[run_idx] = roc_auc_score(y_test, scores)
            ap[run_idx] = average_precision_score(y_test, scores)

            print(
                f'This run finished, AUC={auc[run_idx]:.3f}, AP={ap[run_idx]:.3f}')
            # RUN JUMP
            if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
                print('RUN JUMP')
                print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
                print(f'AUC is : {auc}')
                break
        print(f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP=\
        {np.mean(ap):.3f}({np.std(ap):.3f}),np_seed={params.np_seed}')
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}


params_ = {}


def train_image(params):
    ''' train image '''
    global params_
    params_ = params

    # Load data
    x_train, x_test, y_train, y_test = LoadImageData(params)
    x_train_whole = Tensor(x_train, dtype=mstype.float32)
    x_test_whole = Tensor(x_test, dtype=mstype.float32)

    auc = np.zeros(params.run_num)
    ap = np.zeros(params.run_num)

    global data_name
    data_name = params.data_name

    # Start Train
    for run_idx in tqdm(range(params.run_num)):
        global best_auc
        global best_pr
        global run_index
        run_index = run_idx
        best_auc = 0
        best_pr = 0
        model = SGAE(x_train_whole.shape[1], params.hidden_dim)
        optim = nn.Adam(model.trainable_params(), learning_rate=params.lr)

        if params.verbose and run_idx == 0:
            print(model)

        # One run
        for epoch in range(params.epochs):
            ds_train = create_dataset(x_train, y_train, params)
            ds_val = create_dataset(x_test, y_test, params, is_batch=False)

            epoch_time_start = time.time()
            # train

            # calculate norm thresh
            _, dec_train, _ = model(x_train_whole)
            norm = calculate_norm(x_train_whole, dec_train)
            norm_thresh = np.percentile(norm, params.epsilon)

            loss = 0
            recon_error = 0
            dist_error = 0

            auc_pr = AUC_PR()
            model_withloss = CustomWithLossCell(model, norm_thresh, params)
            eval_net = CustomWithEvalCell(model)
            model_withloss = Model(
                model_withloss, optimizer=optim, eval_network=eval_net, metrics={'auc_pr': auc_pr})
            eval_callback = EvalCallback(
                model_withloss, ds_val, save_path="./saved_model")
            model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=\
                                 [TimeMonitor(30), LossMonitor(30), eval_callback], dataset_sink_mode=False)

            epoch_time = time.time() - epoch_time_start

            # test
            if params.verbose:
                if (epoch + 1) % params.print_step == 0 or epoch == 0:
                    scores, _, _ = model(x_test_whole)
                    scores = scores.asnumpy()
                    auc_ = roc_auc_score(y_test, scores)
                    ap_ = average_precision_score(y_test, scores)
                    print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                          f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:\
                            {dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},' +\
                          f'--AUC:{auc_:.3f} --AP:{ap_:.3f}')

            # Early Stop
            if params.early_stop:
                scores, _, _ = model(x_train_whole)
                scores = scores.asnumpy()
                if np.mean(scores) > params.a / 2:
                    print(
                        f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                    break

        # test
        param_dict = load_checkpoint(
            f"./saved_model/{params.data_name}_best_auc_pr_runs{run_index}.ckpt")
        load_param_into_net(model, param_dict)
        scores, _, _ = model(x_test_whole)
        scores = scores.asnumpy()
        auc[run_idx] = roc_auc_score(y_test, scores)
        ap[run_idx] = average_precision_score(y_test, scores)

        print(
            f'This run finished, AUC={auc[run_idx]:.3f}, AP={ap[run_idx]:.3f}')

        # RUN JUMP
        if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
            print('RUN JUMP')
            print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
            print(f'AUC is : {auc}')
            break

    print(
        f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}


normal_index = 0


def train_document(params):
    ''' train document '''
    global params_
    params_ = params

    # Load data
    dataloader = LoadDocumentData(params)

    auc = np.zeros((params.run_num, dataloader.class_num))
    ap = np.zeros((params.run_num, dataloader.class_num))

    global data_name
    data_name = params.data_name

    # Start Train
    for run_idx in tqdm(range(params.run_num)):

        global run_index
        run_index = run_idx
        for normal_idx in range(dataloader.class_num):

            global best_auc
            global best_pr
            global normal_index
            best_auc = 0
            best_pr = 0
            normal_index = normal_idx

            x_train, x_test, y_train, y_test = dataloader.preprocess(
                normal_idx)
            x_train_whole = Tensor(x_train, dtype=mstype.float32)
            x_test_whole = Tensor(x_test, dtype=mstype.float32)

            model = SGAE(x_train_whole.shape[1], params.hidden_dim)
            optim = nn.Adam(model.trainable_params(), learning_rate=params.lr)

            if params.verbose and normal_idx == 0 and run_idx == 0:
                print(model)

            # One run
            for epoch in range(params.epochs):

                ds_train = create_dataset(x_train, y_train, params)
                ds_val = create_dataset(x_test, y_test, params, is_batch=False)

                epoch_time_start = time.time()

                # calculate norm thresh
                _, dec_train, _ = model(x_train_whole)
                norm = calculate_norm(x_train_whole, dec_train)
                norm_thresh = np.percentile(norm, params.epsilon)

                loss = 0
                recon_error = 0
                dist_error = 0

                auc_pr = AUC_PR()
                model_withloss = CustomWithLossCell(model, norm_thresh, params)
                eval_net = CustomWithEvalCell(model)
                model_withloss = Model(
                    model_withloss, optimizer=optim, eval_network=eval_net, metrics={'auc_pr': auc_pr})
                eval_callback = EvalCallback(
                    model_withloss, ds_val, save_path="./saved_model")
                model_withloss.train(epoch=1, train_dataset=ds_train, callbacks=\
[TimeMonitor(30), LossMonitor(1), eval_callback], dataset_sink_mode=False)
                epoch_time = time.time() - epoch_time_start

                # test
                if params.verbose:
                    if (epoch + 1) % params.print_step == 0 or epoch == 0:
                        scores, _, _ = model(x_test_whole)
                        scores = scores.asnumpy()
                        auc_ = roc_auc_score(y_test, scores)
                        ap_ = average_precision_score(y_test, scores)
                        print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                              f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --\
                                DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},' +\
                              f'--AUC:{auc_:.3f} --AP:{ap_:.3f}')
                # Early Stop
                if params.early_stop:
                    scores, _, _ = model(x_train_whole)
                    scores = scores.asnumpy()
                    if np.mean(scores) > params.a / 2:
                        print(
                            f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                        break

            # test
            param_dict = \
            load_checkpoint(f"./saved_model/{params.data_name}"+\
"_best_auc_pr_runs{run_index}_normal{normal_idx}.ckpt")
            load_param_into_net(model, param_dict)
            scores, _, _ = model(x_test_whole)
            scores = scores.asnumpy()
            auc[run_idx][normal_idx] = roc_auc_score(y_test, scores)
            ap[run_idx][normal_idx] = average_precision_score(y_test, scores)
        print(
            f'This run finished, AUC={np.mean(auc[run_idx]):.3f}, AP={np.mean(ap[run_idx]):.3f}')

        # RUN JUMP
        if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
            print('RUN JUMP')
            print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
            print(f'AUC is : {auc}')
            break

    print(
        f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}


def recog_anomal(data, x_dec, thresh):
    ''' Recognize anomaly
    '''
    norm = calculate_norm(data, x_dec)
    anomal_flag = norm.copy()
    anomal_flag[norm < thresh] = 0
    anomal_flag[norm >= thresh] = 1
    return anomal_flag


def calculate_norm(data, x_dec):
    ''' Calculate l2 norm
    '''
    delta = (data - x_dec)
    norm = ms.numpy.norm(delta, ord=2, axis=1)
    return norm


if __name__ == '__main__':

    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '4'
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    start_time_ = time.time()
    time_name = str(time.strftime("%m%d")) + '_' + \
        str(time.time()).split(".")[1][-3:]
    print(f'Time name is {time_name}')
    print(os.getcwd())
    metrics = pd.DataFrame()

    args = cfg

    if args.ms_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE)
    else:
        context.set_context(mode=context.PYNATIVE_MODE)

    context.set_context(device_target=args.device, device_id=0)

    if args.data_name in ['attack', 'bcsc', 'creditcard', 'diabetic', 'donor', 'intrusion', 'market']:
        an_metrics_dict = train_tabular(args)
    elif args.data_name in ['reuters', '20news']:
        an_metrics_dict = train_document(args)
    elif args.data_name in ['mnist']:
        an_metrics_dict = train_image(args)

    metrics = pd.DataFrame(an_metrics_dict, index=[0])
    metrics.to_csv(
        f'{args.out_dir}{args.model_name}_{args.data_name}_{time_name}.csv')

    print(f'Finished!\nTotal time is {time.time()-start_time_:.2f}s')
    print(f'Current time is {time.strftime("%m%d_%H%M")}')
    print(f'Results:')
    print(metrics.sort_values('AUC', ascending=False))
    