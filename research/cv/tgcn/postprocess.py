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
Calculate metrics for ascend 310 inference results
"""
import os
import argparse
import numpy as np
from src.config import ConfigTGCN
from src.dataprocess import generate_dataset_ms, load_feat_matrix


# Set related parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', help='test dataset directory', type=str, default='./data')
parser.add_argument('--targets_dir', help='targets data directory', type=str,
                    default='./preprocess_Result/targets_ids.npy')
parser.add_argument('--result_dir', help='infer result dir', type=str, default="./result_Files")
parser.add_argument('--device_target', help='device where the code will be implemented', type=str, default='Ascend')
args = parser.parse_args()


if __name__ == "__main__":

    # Config initialization
    config = ConfigTGCN()
    config.batch_size = 1
    # Load evaluation dataset
    dataset = generate_dataset_ms(config, training=False)
    # Directories of inference results
    rst_path = args.result_dir
    targets = np.load(args.targets_dir)
    # Lists to record metrics
    rmse, mae, acc, r_2, var = [], [], [], [], []
    # Calculate metrics
    num_nodes = targets.shape[3]
    _, max_val = load_feat_matrix(config.dataset)
    for i in range(len(os.listdir(rst_path))):
        file_name = os.path.join(rst_path, "T-GCN_data_bs" + str(config.batch_size) + '_' + str(i) + '_0.bin')
        pred = np.fromfile(file_name, np.float32).reshape((-1, num_nodes))
        target = targets[i].reshape((-1, num_nodes))

        rmse.append(np.sqrt(np.square(target - pred).mean()))
        mae.append(np.abs(target - pred).mean())
        acc.append(1 - np.linalg.norm(target - pred, 'fro') / np.linalg.norm(target, 'fro'))
        r_2.append(1 - np.sum((target - pred) ** 2) / np.sum((target - np.mean(pred)) ** 2))
        var.append(1 - np.var(target - pred) / np.var(target))

    RMSE = np.array(rmse).mean() * max_val
    MAE = np.array(mae).mean() * max_val
    ACC = np.array(acc).mean()
    R_2 = np.array(r_2).mean()
    VAR = np.array(var).mean()

    print(f'RMSE {RMSE:.4f} | MAE {MAE:.4f} | ACC {ACC:.4f} | R_2 {R_2:.4f}| VAR {VAR:.4f}')
