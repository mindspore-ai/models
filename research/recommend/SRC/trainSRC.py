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
from argparse import ArgumentParser

import numpy as np
from mindspore import load_param_into_net, load_checkpoint, save_checkpoint
from mindspore.nn import PolynomialDecayLR, Adam
from tqdm import tqdm

from KTScripts.DataLoader import KTDataset
from KTScripts.utils import set_random_seed
from Scripts.Agent.utils import pl_loss
from Scripts.Envs import KESEnv
from Scripts.Optimizer import ModelWithLoss, ModelWithOptimizer
from Scripts.options import get_options
from Scripts.utils import load_agent, get_data


def main(args):
    set_random_seed(args.rand_seed)
    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    env = KESEnv(dataset, args.model, args.dataset)
    args.skill_num = env.skill_num
    # Create Agent
    model = load_agent(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_path = os.path.join(args.save_dir, args.exp_name + str(args.path))
    if args.load_model:
        load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
        print(f"Load Model From {model_path}")
    # Optimizer
    polynomial_decay_lr = PolynomialDecayLR(learning_rate=args.lr,
                                            end_learning_rate=1e-5,
                                            decay_steps=200,
                                            power=0.5,
                                            update_decay_steps=True)
    optimizer = Adam(model.trainable_params(), learning_rate=polynomial_decay_lr)
    criterion = pl_loss
    model_with_loss = ModelWithLoss(model, criterion)
    model_train = ModelWithOptimizer(model_with_loss, optimizer)
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size = args.skill_num, args.batch_size
    targets, result = None, None
    best_reward = -1e9
    print('-' * 20 + "Training Start" + '-' * 20)
    model_train.set_train()
    for epoch in range(args.num_epochs):
        avg_time = 0
        epoch_mean_rewards = []
        for i in tqdm(range(200)):
            t0 = time.perf_counter()
            targets, initial_logs, origin_path = get_data(batch_size, skill_num, 3, 10, args.path, args.steps)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            result = model(*data)
            env.n_step(result[0], binary=True)
            rewards = env.end_episode()
            loss = model_train(*data[:-1], result[2], rewards).asnumpy()
            mean_reward = np.mean(rewards.asnumpy())
            avg_time += time.perf_counter() - t0
            epoch_mean_rewards.append(mean_reward)
            all_rewards.append(mean_reward)
            print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\treward:{:.4f}'
                  .format(epoch, i, avg_time / (i + 1), loss, mean_reward))
        print(targets[:10], '\n', result[0][:10])
        all_mean_rewards.append(np.mean(epoch_mean_rewards))
        if all_mean_rewards[-1] > best_reward:
            best_reward = all_mean_rewards[-1]
            save_checkpoint(model, model_path)
            print("New Best Result Saved!")
        print(f"Best Reward Now:{best_reward:.4f}")
    for i in all_mean_rewards:
        print(i)
    if not os.path.exists(args.visual_dir):
        os.makedirs(args.visual_dir)
    np.save(os.path.join(args.visual_dir, f'{args.exp_name}_{args.path}'), np.array(all_rewards))

    print('-' * 20 + "Testing Start" + '-' * 20)
    test_rewards = []
    model_with_loss.set_train(False)
    load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
    for i in tqdm(range(200)):
        targets, initial_logs, origin_path = get_data(batch_size, skill_num, 3, 10, args.path, args.steps)
        initial_log_scores = env.begin_episode(targets, initial_logs)
        data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
        result = model(*data)
        env.n_step(result[0], binary=True)
        rewards = env.end_episode()
        loss = criterion(result[1], rewards).asnumpy()
        mean_reward = np.mean(rewards.asnumpy())
        test_rewards.append(mean_reward)
        print(f'batch:{i}\tloss:{loss:.4f}\treward:{mean_reward:.4f}')
    print(result[0][:10])
    print(f"Mean Reward for Test:{np.mean(test_rewards)}")


if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'SRC', 'simulator': 'KES'})
    main(args_)
