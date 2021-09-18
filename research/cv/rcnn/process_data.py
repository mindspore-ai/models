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
process data
"""
import os.path
import pickle
import sys
import time
from multiprocessing import Pool

import cv2

from src import paths
from src.utils.util import check_dir, all_cls, compute_ious, compute_trans, parse_xml


def check_all_dirs():
    """
    check all dirs
    """
    for phase_ in ['', 'train', 'val', 'test']:
        check_dir(os.path.join(paths.Data.ss_root, phase_))
        check_dir(os.path.join(paths.Data.finetune, phase_))
        check_dir(os.path.join(paths.Data.svm, phase_))
        check_dir(os.path.join(paths.Data.regression, phase_))


def config(gs_, img_, strategy='q'):
    """
    ss config
    :param gs_: gs
    :param img_: image
    :param strategy: ss strategy
    """
    gs_.setBaseImage(img_)

    if strategy == 's':
        gs_.switchToSingleStrategy()
    elif strategy == 'f':
        gs_.switchToSelectiveSearchFast()
    elif strategy == 'q':
        gs_.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs_):
    """
    ss
    :param gs_: gs
    :return: rects
    """
    rects_ = gs_.process()
    rects_[:, 2] += rects_[:, 0]
    rects_[:, 3] += rects_[:, 1]

    return rects_


def generate_ss_result(lines_, phase_, idx_):
    """
    generate ss result
    :param lines_: lines
    :param phase_: phase
    :param idx_: id
    """
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    if phase_ == 'test':
        jpeg_root_path = paths.Data.jpeg_test
    else:
        jpeg_root_path = paths.Data.jpeg

    dict_ = {}
    for image_id in lines_:
        jpeg_path = os.path.join(jpeg_root_path, "%s.jpg" % image_id)
        img = cv2.imread(jpeg_path, cv2.IMREAD_COLOR)
        config(gs, img, strategy='f')
        print("[%5s][%d] selective search finished on img %s.jpg" % (phase_, idx_, image_id))
        try:
            rects = get_rects(gs)
            dict_[image_id] = rects
        except IndexError:
            print("Skip this picture:", jpeg_path)
    dump_path = os.path.join(paths.Data.ss_root, phase_, 'ss_result_%s.pickle' % str(idx_))
    pickle.dump(dict_, open(dump_path, 'wb'))


def generate_dataset(phase_):
    """
    generate dataset
    :param phase_: phase
    """
    ss_path = os.path.join(paths.Data.ss_root, phase_)
    if phase_ == 'test':
        anno_path = paths.Data.annotation_test
    else:
        anno_path = paths.Data.annotation

    ss_results_num = os.listdir(ss_path)
    finetune_dict = {}
    svm_dict = {}
    reg_dict = {}
    for s_num in range(len(ss_results_num)):
        ss_results_path = os.path.join(ss_path, ss_results_num[s_num])
        ss_results = pickle.load(open(ss_results_path, 'rb'))
        for key in ss_results.keys():
            rects = ss_results[key]  # [x1,y1,h,w]
            all_bndboxs, obj_names = parse_xml(os.path.join(anno_path, key + '.xml'))
            all_maximum_bndbox_size = 0
            ii = 0
            temp_svm_values = list()
            for bndbox in all_bndboxs:
                xmin, ymin, xmax, ymax = bndbox
                bndbox_size = (ymax - ymin) * (xmax - xmin)
                if bndbox_size > all_maximum_bndbox_size:
                    all_maximum_bndbox_size = bndbox_size
                values = bndbox.tolist()
                values.append(all_cls.index(obj_names[ii]))
                temp_svm_values.append(values)
                ii += 1

            max_score_iou_list, max_label_list, max_bnd_list = compute_ious(rects, all_bndboxs, obj_names)

            temp_finetune_values = list()
            temp_reg_values = list()

            for i in range(len(max_score_iou_list)):
                xmin, ymin, xmax, ymax = rects[i]
                rect_size = (ymax - ymin) * (xmax - xmin)

                if 0 < max_score_iou_list[i] < 0.5 and rect_size > all_maximum_bndbox_size / 5.0:
                    values = rects[i].tolist()
                    values.append(20)
                    temp_finetune_values.append(values)

                if 0 < max_score_iou_list[i] <= 0.3 and rect_size > all_maximum_bndbox_size / 5.0:
                    values = rects[i].tolist()
                    values.append(20)
                    temp_svm_values.append(values)

                if max_score_iou_list[i] >= 0.5:
                    values = rects[i].tolist()
                    values.append(all_cls.index(max_label_list[i]))
                    temp_finetune_values.append(values)

                if max_score_iou_list[i] > 0.7:
                    w = rects[i][2] - rects[i][0]
                    h = rects[i][3] - rects[i][1]
                    rects[i][0] = rects[i][0] + w / 2
                    rects[i][1] = rects[i][1] + h / 2
                    rects[i][2] = w
                    rects[i][3] = h
                    values = rects[i].tolist()
                    values.append(all_cls.index(max_label_list[i]))
                    trans = compute_trans(rects[i], max_bnd_list[i])
                    values.append(trans)
                    temp_reg_values.append(values)

            finetune_dict[key] = temp_finetune_values
            svm_dict[key] = temp_svm_values
            reg_dict[key] = temp_reg_values

    pickle.dump(finetune_dict, open(os.path.join(paths.Data.finetune, phase_, 'fine_data.pickle'), 'wb'))
    pickle.dump(svm_dict, open(os.path.join(paths.Data.svm, phase_, 'svm_data.pickle'), 'wb'))
    pickle.dump(reg_dict, open(os.path.join(paths.Data.regression, phase_, 'reg_data.pickle'), 'wb'))


if __name__ == '__main__':
    check_all_dirs()
    time_start = time.time()

    # selective search
    pool = Pool(19)
    for phase in ['train', 'val', 'test']:
        if phase == 'train':
            image_list = paths.Data.image_id_list_train
        elif phase == 'val':
            image_list = paths.Data.image_id_list_val
        else:
            image_list = paths.Data.image_id_list_test
        with open(image_list, 'r') as file:
            lines = file.read().strip().split()
            lines_list = [lines[i:i + 600] for i in range(0, len(lines), 600)]
            for idx, lines in enumerate(lines_list):
                pool.apply_async(generate_ss_result, (lines, phase, idx + 1))
    pool.close()
    pool.join()

    # generate dataset for finetune, svm and regression
    pool = Pool(3)
    for phase in ['train', 'val', 'test']:
        pool.apply_async(generate_dataset, (phase,))
    pool.close()
    print("generating dataset... this will take several minutes~")
    pool.join()

    print(f"data preprocess finished in {time.time()-time_start:.0f} s.")
