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
import argparse
import os

from mindspore import context, ops
from tqdm import tqdm

from KTScripts.DataLoader import KTDataset
from Scripts.Envs import KESEnv
from Scripts.Envs.KES.utils import load_d_agent
from Scripts.utils import get_data


def rank(env, path):
    path = path.reshape(-1, 1)
    targets, initial_logs, _ = get_data(path.shape[0], env.skill_num, 3, 10, 0, 1)
    env.begin_episode(targets, initial_logs)
    env.n_step(path)
    rewards = env.end_episode().squeeze(-1)
    return path.reshape(-1)[ops.sort(rewards)[1]]


def testRuleBased(args):
    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    env = KESEnv(dataset, args.model, args.dataset)
    targets, initial_logs, origin_paths = get_data(args.batch, env.skill_num, 3, 10, args.p, args.n)
    ranked_paths = [None] * args.batch
    if args.agent == 'rule':
        for i in tqdm(range(args.batch)):
            path = rank(env, origin_paths[i])[-args.n:]
            ranked_paths[i] = path
        ranked_paths = ops.stack(ranked_paths, 0)
    elif args.agent == 'random':
        ranked_paths = origin_paths[:, -args.n:]
    elif args.agent == 'GRU4Rec':
        d_model = load_d_agent(args.agent, args.dataset, env.skill_num, False)
        ranked_paths = d_model.GRU4RecSelect(origin_paths, args.n, env.skill_num, initial_logs)
    print(ranked_paths)
    env.begin_episode(targets, initial_logs)
    env.n_step(ranked_paths)
    print(ops.mean(env.end_episode()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='./data/')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='assist09', choices=['assist09', 'junyi', 'assist15'])
    parser.add_argument('-a', '--agent', type=str, default='rule', choices=['rule', 'random', 'GRU4Rec'])
    parser.add_argument('-m', '--model', type=str, default='DKT', choices=['DKT', 'CoKT'])
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=128)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('-p', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args_ = parser.parse_args()

    if args_.cuda >= 0:
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=args_.cuda)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    testRuleBased(args_)
