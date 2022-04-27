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
'''eval'''
import os
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops
from mindspore import context
import scipy.io
from src.dataset import create_dataset
from src.model import HED
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))


def get_files(folder, name_filter=None, extension_filter=None):
    '''get file'''
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    if name_filter is None:
        def name_cond(filename):
            return True
    else:
        def name_cond(filename):
            return name_filter in filename

    if extension_filter is None:
        def ext_cond(filename):
            return True
    else:
        def ext_cond(filename):
            return filename.endswith(extension_filter)

    filtered_files = []
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)
    return filtered_files

@moxing_wrapper(pre_process=modelarts_pre_process)
def test_hed():
    '''test'''
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                        save_graphs=False, device_id=device_id)
    net = HED()
    train_path = os.path.join(config.data_path, 'output/train.lst')
    test_path = os.path.join(config.data_path, 'output/test.lst')
    val_path = os.path.join(config.data_path, 'output/val.lst')
    img_extension = '.jpg'
    lbl_extension = '.jpg'
    train_img = get_files(os.path.join(config.data_path, 'BSDS500/data/images/train'),
                          extension_filter=img_extension)
    train_label = get_files(os.path.join(config.data_path, 'BSDS500/data/labels/train'),
                            extension_filter=lbl_extension)
    test_img = get_files(os.path.join(config.data_path, 'BSDS500/data/images/test'),
                         extension_filter=img_extension)
    test_label = get_files(os.path.join(config.data_path, 'BSDS500/data/labels/test'),
                           extension_filter=lbl_extension)
    val_img = get_files(os.path.join(config.data_path, 'BSDS500/data/images/val'),
                        extension_filter=img_extension)
    val_label = get_files(os.path.join(config.data_path, 'BSDS500/data/labels/val'),
                          extension_filter=lbl_extension)

    f = open(train_path, "w")
    for img, label in zip(train_img, train_label):
        f.write(str(img) + " " + str(label))
        f.write('\n')
    f.close()
    f = open(test_path, "w")
    for img, label in zip(test_img, test_label):
        f.write(str(img) + " " + str(label))
        f.write('\n')
    f = open(val_path, "w")
    for img, label in zip(val_img, val_label):
        f.write(str(img) + " " + str(label))
        f.write('\n')
    f.close()
    test_loader = create_dataset(test_path, is_training=False, is_shuffle=False)
    print("evaluation image number:", test_loader.get_dataset_size())
    net = HED()

    # load HED ckpt
    param_dict = load_checkpoint(config.load_path)
    load_param_into_net(net, param_dict)
    print("load hed success!")
    idx = 0

    # test.lst路径
    with open(test_path, 'r') as f:
        test_list = f.readlines()
    test_list = [i.rstrip() for i in test_list]

    result_path = os.path.join(config.res_output_path, 'result/hed_result')
    if not os.path.exists(result_path):
        for data in test_loader.create_dict_iterator():
            results = net(data['test'])
            squeeze = ops.Squeeze()
            # fuse
            result = squeeze(results[-1])
            result = result.asnumpy()
            filename, _ = test_list[idx].split()
            filename = filename.split('/')[-1]
            filename = filename.split('.')[0]
            print(filename)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            try:
                result_path_mat = os.path.join(result_path, "{}.mat".format(filename))
                scipy.io.savemat(result_path_mat, {'result': result})
            except OSError:
                pass
            print("running test [%d]" % (idx + 1))
            idx += 1
        print("begin test...")
    else:
        print("begin test...")
def main():
    test_hed()
    config.result_dir = os.path.join(config.res_output_path, 'result/hed_result')
    config.save_dir = os.path.join(config.res_output_path, 'result/hed_eval_result')
    config.gt_dir = os.path.join(config.data_path, 'BSDS500/data/groundTruth/test')
    alg = [config.alg]  # algorithms for plotting
    model_name_list = [config.model_name_list]  # model name
    result_dir = os.path.abspath(config.result_dir)  # forward result directory
    save_dir = os.path.abspath(config.save_dir)  # nms result directory
    gt_dir = os.path.abspath(config.gt_dir)  # ground truth directory
    key = config.key  # x = scipy.io.loadmat(filename)[key]
    file_format_eval = config.file_format_eval  # ".mat" or ".npy"
    workers = config.workers  # number workers
    nms_process(model_name_list, result_dir, save_dir, key, file_format_eval)
    eval_edge(alg, model_name_list, save_dir, gt_dir, workers)

if __name__ == '__main__':
    from src.nms_process import nms_process
    from src.eval_edge import eval_edge
    main()
