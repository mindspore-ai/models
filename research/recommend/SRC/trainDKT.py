# Copyright 2023 Huawei Technologies Co., Ltd
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
import time
from argparse import Namespace

import numpy as np
from mindspore import load_param_into_net, load_checkpoint, save_checkpoint
from mindspore.dataset import GeneratorDataset
from mindspore.nn import Adam, BCELoss, PolynomialDecayLR
from tqdm import tqdm

from KTScripts.BackModels import nll_loss
from KTScripts.DataLoader import KTDataset, RecDataset, RetrievalDataset
from KTScripts.PredictModel import ModelWithLoss, ModelWithLossMask, ModelWithOptimizer
from KTScripts.utils import set_random_seed, load_model, evaluate_utils


def main(args: Namespace):
    print()
    set_random_seed(args.rand_seed)
    dataset = RecDataset if args.forRec else (RetrievalDataset if args.retrieval else KTDataset)
    dataset = dataset(os.path.join(args.data_dir, args.dataset))
    args.feat_nums, args.user_nums = dataset.feats_num, dataset.users_num
    if args.retrieval:
        dataset = GeneratorDataset(source=dataset,
                                   column_names=['intra_x', 'inter_his', 'inter_r', 'y', 'mask', 'inter_len'],
                                   shuffle=False, num_parallel_workers=8, python_multiprocessing=False)
        dataset = dataset.batch(args.batch_size, num_parallel_workers=1)
        train_data, test_data = dataset.split([0.8, 0.2], randomize=False)
    else:
        dataset = GeneratorDataset(source=dataset, column_names=['x', 'y', 'mask'], shuffle=True)
        dataset = dataset.batch(args.batch_size, num_parallel_workers=1)
        train_data, test_data = dataset.split([0.8, 0.2], randomize=True)
    if args.forRec:
        args.output_size = args.feat_nums
    # Model
    model = load_model(args)
    model_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.load_model:
        load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
        print(f"Load Model From {model_path}")
    # Optimizer
    polynomial_decay_lr = PolynomialDecayLR(learning_rate=args.lr,
                                            end_learning_rate=1e-5,
                                            decay_steps=train_data.get_dataset_size() // 10 + 1,
                                            power=0.5,
                                            update_decay_steps=True)
    optimizer = Adam(model.trainable_params(), learning_rate=polynomial_decay_lr, weight_decay=args.l2_reg)
    if args.forRec:
        model_with_loss = ModelWithLossMask(model, nll_loss)
    else:
        model_with_loss = ModelWithLoss(model, BCELoss(reduction='mean'))
    model_train = ModelWithOptimizer(model_with_loss, optimizer, args.forRec)
    best_val_auc = 0
    train_total, test_total = train_data.get_dataset_size(), test_data.get_dataset_size()
    print('-' * 20 + "Training Start" + '-' * 20)
    for epoch in range(args.num_epochs):
        avg_time = 0
        model_train.set_train()
        for i, data in tqdm(enumerate(train_data.create_tuple_iterator()), total=train_total):
            t0 = time.perf_counter()
            loss, output_data = model_train(*data)
            loss = loss.asnumpy()
            acc, auc = evaluate_utils(*output_data)
            avg_time += time.perf_counter() - t0
            print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\tacc:{:.4f}\tauc:{:.4f}'
                  .format(epoch, i, avg_time / (i + 1), loss, acc, auc))
        print('-' * 20 + "Validating Start" + '-' * 20)
        val_eval = [[], []]
        loss_total, data_total = 0, 0
        model_with_loss.set_train(False)
        for data in tqdm(test_data.create_tuple_iterator(), total=test_total):
            loss, output_data = model_with_loss.output(*data)
            val_eval[0].append(output_data[0].asnumpy())
            val_eval[1].append(output_data[1].asnumpy())
            loss_total += loss.asnumpy() * len(data[0])
            data_total += len(data[0])
        val_eval = [np.concatenate(_) for _ in val_eval]
        acc, auc = evaluate_utils(*val_eval)
        print(f"Validating loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")
        if auc >= best_val_auc:
            best_val_auc = auc
            save_checkpoint(model, model_path)
            print("New best result Saved!")
        print(f"Best Auc Now:{best_val_auc:.4f}")

    print('-' * 20 + "Testing Start" + '-' * 20)
    val_eval = [[], []]
    loss_total, data_total = 0, 0
    model_with_loss.set_train(False)
    for data in tqdm(test_data.create_tuple_iterator(), total=test_total):
        loss, output_data = model_with_loss.output(*data)
        val_eval[0].append(output_data[0].asnumpy())
        val_eval[1].append(output_data[1].asnumpy())
        loss_total += loss.asnumpy() * len(data[0])
        data_total += len(data[0])
    val_eval = [np.concatenate(_) for _ in val_eval]
    print(val_eval[0], val_eval[0].mean())
    print(val_eval[1], val_eval[1].mean())
    acc, auc = evaluate_utils(*val_eval)
    print(f"Testing loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    from KTScripts.options import get_options

    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser)
    main(args_)
