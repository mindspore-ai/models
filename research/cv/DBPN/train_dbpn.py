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

"""General-purpose training script for super resolution.
You need to specify the dataset ('--train_GT_path','val_LR_path','val_GT_path')
Example:
    Train a resnet model:
       python train_dbpn.py --device_id=0 --train_GT_path="/data/DBPN_data/DIV2K_train_HR"
       --val_LR_path="/data/DBPN_data/Set5/LR" --val_GT_path="/data/DBPN_data/Set5/HR"
"""
import os
import time
import warnings

import mindspore
import mindspore.nn as nn
from mindspore import save_checkpoint, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from mindspore.ops import functional as F

from src.dataset.dataset import DBPNDataset, DatasetVal, create_train_dataset, create_val_dataset
from src.loss.generatorloss import GeneratorLoss
from src.model.generator import get_generator
from src.util.config import get_args
from src.util.utils import save_img, save_losses, save_psnr, compute_psnr

warnings.filterwarnings(action="ignore", category=UserWarning, module="DBPN")

args = get_args()
mindspore.set_seed(args.seed)
print(args)

epoch_loss = []
best_avgpsnr = 0
eval_mean_psnr = []

save_eval_path = os.path.join(args.Results, args.valDataset, args.model_type)
if not os.path.exists(save_eval_path):
    os.makedirs(save_eval_path)

save_loss_path = 'results/genloss/'
if not os.path.exists(save_loss_path):
    os.makedirs(save_loss_path)


def train(trainoneStep, trainds, valds, net, eval_flag=False):
    """train the generator
    Args:
        trainoneStep(Cell): the network of
        trainds(dataset): train datasets
        valds(dataset): validation datasets
        net(Cell): the generator network
        eval_flag(boolean): whether train and eval or not
    """
    global best_avgpsnr
    trainoneStep.set_train()
    trainoneStep.set_grad()
    steps = trainds.get_dataset_size()
    val_steps = valds.get_dataset_size()

    for epoch in range(args.start_iter, args.nEpochs + 1):
        e_loss = 0
        t0 = time.time()
        for iteration, batch in enumerate(trainds.create_dict_iterator(), 1):
            hr_img = batch['target_image']
            lr_img = batch['input_image']
            loss = trainoneStep(hr_img, lr_img)
            e_loss += loss.asnumpy()
            print('Epoch[{}]({}/{}): loss: {:.4f}'.format(epoch, iteration, steps, loss.asnumpy()))
        t1 = time.time()
        mean = e_loss / steps
        epoch_loss.append(mean)
        print("Epoch {} Complete: Avg. Loss: {:.4f}|| Time: {} min {}s.".format(epoch, mean, int((t1 - t0) / 60),
                                                                                int(int(t1 - t0) % 60)))
        name = os.path.join(save_loss_path, args.valDataset + '_' + args.model_type)
        save_losses(epoch_loss, None, name)
        if eval_flag:
            mean_psnr = 0
            et0 = time.time()
            for index, batch in enumerate(valds.create_dict_iterator(), 1):
                lr_img = F.stop_gradient(batch['input_image'])
                hr_img = F.stop_gradient(batch['target_image'])
                prediction = net(lr_img)
                save_img(prediction, str(index), save_eval_path)
                s_psnr = compute_psnr(hr_img.squeeze(), prediction.squeeze())
                mean_psnr += s_psnr
                print("Processing: {} PSNR on {:.4f}".format(index, s_psnr))
            et1 = time.time()
            mean_psnr = mean_psnr / val_steps
            print("Epoch {} Complete: Avg. PSNR: {:.4f}|| Timer:{:.2f} min".format(epoch, mean_psnr, (et1 - et0) / 60))
            eval_mean_psnr.append(mean_psnr)
            savepath = os.path.join(save_loss_path, "%s_%s_psnr" % (args.valDataset, args.model_type))
            save_psnr(eval_mean_psnr, savepath, args.model_type)
            if best_avgpsnr < mean_psnr:
                best_avgpsnr = mean_psnr
                save_ckpt = os.path.join(args.save_folder, '{}_{}_best.ckpt'.format(args.valDataset, args.model_type))
                save_checkpoint(trainoneStep.network, save_ckpt)


if __name__ == '__main__':
    # distribute
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.run_distribute:
        print("distribute")
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = args.device_num
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        rank = get_rank()
    else:
        device_id = args.device_id
        context.set_context(device_id=device_id)
    train_dataset = DBPNDataset(args.train_GT_path, args)
    train_ds = create_train_dataset(train_dataset, args)
    train_steps = train_ds.get_dataset_size()

    val_dataset = DatasetVal(args.val_GT_path, args.val_LR_path, args)
    val_ds = create_val_dataset(val_dataset, args)
    print('===>Building model ', args.model_type)
    model = get_generator(args.model_type, args.upscale_factor)
    print('====>start training')

    if args.load_pretrained:
        ckpt = os.path.join(args.save_folder, args.pretrained_sr)
        print('=====> load params into generator')
        params = load_checkpoint(ckpt)
        load_param_into_net(model, params)
        print('=====> finish load generator')

    lossNetwork = GeneratorLoss(model)
    milestone = [int(args.nEpochs / 2) * train_steps, args.nEpochs * train_steps]
    learning_rates = [args.lr, args.lr / 10.0]
    lr = piecewise_constant_lr(milestone, learning_rates)

    optimizer = nn.Adam(model.trainable_params(), lr, loss_scale=args.sens)
    if args.model_type == 'DDBPN' or args.model_type == 'DBPN':
        from src.trainonestep.trainonestepgenv2 import TrainOnestepGen
    else:
        from src.trainonestep.trainonestepgen import TrainOnestepGen
    trainonestepNet = TrainOnestepGen(lossNetwork, optimizer, sens=args.sens)

    train(trainonestepNet, train_ds, val_ds, model, args.eval_flag)
    print('========= best_psnr=', best_avgpsnr, "db")
