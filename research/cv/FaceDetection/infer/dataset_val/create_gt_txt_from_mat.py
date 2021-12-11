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
"""create_gt_txt_from_mat.py"""
import os
import argparse
import tqdm
import numpy as np
from scipy.io import loadmat
from cython_bbox import bbox_overlaps

_MAP = {
    '0': '0--Parade',
    '1': '1--Handshaking',
    '2': '2--Demonstration',
    '3': '3--Riot',
    '4': '4--Dancing',
    '5': '5--Car_Accident',
    '6': '6--Funeral',
    '7': '7--Cheering',
    '8': '8--Election_Campain',
    '9': '9--Press_Conference',
    '10': '10--People_Marching',
    '11': '11--Meeting',
    '12': '12--Group',
    '13': '13--Interview',
    '14': '14--Traffic',
    '15': '15--Stock_Market',
    '16': '16--Award_Ceremony',
    '17': '17--Ceremony',
    '18': '18--Concerts',
    '19': '19--Couple',
    '20': '20--Family_Group',
    '21': '21--Festival',
    '22': '22--Picnic',
    '23': '23--Shoppers',
    '24': '24--Soldier_Firing',
    '25': '25--Soldier_Patrol',
    '26': '26--Soldier_Drilling',
    '27': '27--Spa',
    '28': '28--Sports_Fan',
    '29': '29--Students_Schoolkids',
    '30': '30--Surgeons',
    '31': '31--Waiter_Waitress',
    '32': '32--Worker_Laborer',
    '33': '33--Running',
    '34': '34--Baseball',
    '35': '35--Basketball',
    '36': '36--Football',
    '37': '37--Soccer',
    '38': '38--Tennis',
    '39': '39--Ice_Skating',
    '40': '40--Gymnastics',
    '41': '41--Swimming',
    '42': '42--Car_Racing',
    '43': '43--Row_Boat',
    '44': '44--Aerobics',
    '45': '45--Balloonist',
    '46': '46--Jockey',
    '47': '47--Matador_Bullfighter',
    '48': '48--Parachutist_Paratrooper',
    '49': '49--Greeting',
    '50': '50--Celebration_Or_Party',
    '51': '51--Dresses',
    '52': '52--Photographers',
    '53': '53--Raid',
    '54': '54--Rescue',
    '55': '55--Sports_Coach_Trainer',
    '56': '56--Voter',
    '57': '57--Angler',
    '58': '58--Hockey',
    '59': '59--people--driving--car',
    '61': '61--Street_Battle'
}


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if v:
                _min = np.min(v[:, -1])
                _max = np.max(v[:, -1])
                max_score = max(_max, max_score)
                min_score = min(_min, min_score)
            else:
                continue

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if v:
                v[:, -1] = (v[:, -1] - min_score) / diff
            else:
                continue


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    """
        img_pr_info
    """
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if r_index:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
        else:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0

    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):
    """
        voc_ap
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    """
        evaluation
    """
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))

        outputTxtDir = './bbx_gt_txt/'
        if not os.path.exists(outputTxtDir):
            os.makedirs(outputTxtDir)

        outputTxtFile = outputTxtDir + settings[setting_id] + '.txt'
        if os.path.exists(outputTxtFile):
            os.remove(outputTxtFile)

        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            img_list = file_list[i][0]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                gt_boxes = gt_bbx_list[j][0]
                keep_index = sub_gt_list[j][0]
                imgName = img_list[j][0][0]
                imgPath = _MAP[imgName.split('_')[0]] + '/' + imgName + '.jpg'

                faceNum = len(keep_index)

                with open(outputTxtFile, 'a') as txtFile:
                    txtFile.write(imgPath + '\n')
                    txtFile.write(str(faceNum) + '\n')
                    if faceNum == 0:
                        txtFile.write(str(faceNum) + '\n')
                    for index in keep_index:
                        curI = index[0] - 1
                        bbox = gt_boxes[curI]
                        txtFile.write('%d %d %d %d\n' % (bbox[0], bbox[1], bbox[2], bbox[3]))
                txtFile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred')
    parser.add_argument('-g', '--gt', default='./eval_tools/ground_truth/')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)
