# Copyright(C) 2021. Huawei Technologies Co.,Ltd
#
# Licensed under the Apache License, Version 3.0 (the "License");
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
"""train ecolite."""
import os
import argparse
import moxing as mox


def _parse_args():
    """parse input"""
    parser = argparse.ArgumentParser('mindspore ecolite training')
    parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something', 'jhmdb'])
    parser.add_argument('--modality', default="RGB", type=str, choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str)
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument("--export_file_name", type=str, default="ecolite", help="output file name.")
    parser.add_argument("--modelarts_dataset_unzip_name", type=str, default="ucf101",
                        help="modelarts_dataset_unzip_name.")
    parser.add_argument('--pre_trained_ckpt', type=str, default='ms_model_kinetics_checkpoint0720.ckpt',
                        help='pretrained backbone')
    parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR',
                        help='file format')
    parser.add_argument('--train_url', type=str, default='', help='where training log and ckpts saved')
    # dataset
    parser.add_argument('--data_url', type=str, default='', help='path of dataset')
    # model
    parser.add_argument('--run_distribute', default=False, help='run_distribute')
    parser.add_argument('--epochs', type=int, default=40)

    args, _ = parser.parse_known_args()
    return args


def _train(args):
    """train ecolite"""
    train_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "train.py")
    trainlist = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), args.train_list)
    vallist = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), args.val_list)
    resumepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), args.pre_trained_ckpt)
    ret = os.system(
        f'python {train_file} --dataset {args.dataset} --train_url {args.train_url} \
         --batch-size {args.batch_size} --data_url {args.data_url}  --run_distribute {args.run_distribute}\
          --epochs {args.epochs} --modelarts_dataset_unzip_name {args.modelarts_dataset_unzip_name} \
          --train_list {trainlist} --val_list {vallist} --resume {resumepath}')
    return ret


def _export_air(args):
    """export to air"""
    export_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "export.py")
    dirname = os.getcwd()
    modelname = args.dataset + "bestacc.ckpt"
    resumepath = os.path.join(dirname, modelname)
    os.system(
        f'python {export_file} --dataset {args.dataset} --modality {args.modality} --batch-size {args.batch_size} \
        --file_format {args.file_format} --file_name {args.export_file_name} \
        --checkpoint_path {resumepath}')


def main():
    """main function"""
    args = _parse_args()
    local_data_url = '/cache/data/'
    mox.file.copy_parallel(args.data_url, local_data_url)
    data_name = args.modelarts_dataset_unzip_name
    zip_command = "unzip -o %s -d %s > /dev/null" % (local_data_url + data_name, local_data_url)
    os.system(zip_command)
    dirname = os.getcwd()
    zip_command = "cp -r %s %s" % (os.path.join(local_data_url, data_name), os.path.join(dirname, data_name))
    os.system(zip_command)
    _train(args)
    _export_air(args)
    modelname = args.dataset + "bestacc.ckpt"
    from_path = os.path.join(dirname, modelname)
    to_path = args.train_url + modelname
    mox.file.copy_parallel(from_path, to_path)
    filename = args.export_file_name + ".air"
    from_path = os.path.join(dirname, filename)
    to_path = args.train_url + filename
    mox.file.copy_parallel(from_path, to_path)


if __name__ == '__main__':
    main()
