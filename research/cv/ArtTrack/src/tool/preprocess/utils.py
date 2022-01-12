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

import json

import numpy as np

from src.log import log


def util_set_scale(img_list, ref_height=200):
    """
        set images scale
    """
    head_height_ratio = 1.0 / 8

    for _, v in enumerate(img_list):
        rect = getattr(v, 'rect', None)
        if rect is not None and rect:
            for _, rv in enumerate(rect):
                points = getattr(rv, "points", None)
                if points is not None and points:
                    head_size = util_get_head_size(rv)
                    sc = ref_height * head_height_ratio / head_size
                    assert 100 > sc > 0.01
                    points.scale = 1 / sc
            v['rect'] = rect

    return img_list


def util_get_head_size(rect):
    sc_bias = 0.6
    return sc_bias * np.linalg.norm(np.array([rect['x2'], rect['y2']]) - np.array([rect['x1'], rect['y1']]))


def pad_zeros(s, npad):
    n = len(str(s))

    assert n <= npad
    return '0' * (npad - n) + str(s)


class NumpyEncoder(json.JSONDecoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.uint8):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONDecoder.default(self, obj)


def set_rect_field(rect_dict, rv, field):
    if hasattr(rv, field):
        rect_dict[field] = getattr(rv, field)[0, 0]


def set_rect_scale(rect_dict, rv):
    if hasattr(rv, 'scale'):
        scale = rv.scale.flatten()
        if scale:
            rect_dict['scale'] = scale[0]


def set_rect_vidx(rect_dict, rv):
    if hasattr(rv, 'vidx'):
        rect_dict['vidx'] = rv.vidx - 1


def set_rect_frame_sec(rect_dict, rv):
    if hasattr(rv, 'frame_sec'):
        rect_dict['frame_sec'] = rv.frame_sec


def set_rect_objpos(rect_dict, rv):
    if hasattr(rv, 'objpos'):
        objpos = rv.objpos.flatten()
        if rect_dict.get('scale') is not None:
            rect_dict['objpos_x'] = objpos[0].x[0, 0]
            rect_dict['objpos_y'] = objpos[0].y[0, 0]


def set_rect_points_list(rect_dict, rv):
    rect_points = getattr(rv, 'annopoints', None)
    if rect_points is None or not rect_points:
        points_list = []
        points = []
    else:
        rect_points = rect_points[0, 0]
        points_list = []
        points = rect_points.point[0]
    for f_p in points:
        pp = {'x': f_p.x[0, 0],
              'y': f_p.y[0, 0],
              'id': f_p.id[0, 0],
              }
        if hasattr(f_p, 'is_visible'):
            visible = f_p.is_visible.flatten()
            if visible:
                pp['is_visible'] = visible[0]
        points_list.append(pp)
    rect_dict['points'] = points_list


def set_act_name(act_dict, field, name):
    if name:
        act_dict[field] = name[0]


def mpii_mat2dict(mpii):
    """
    raw mat to dict
    """
    mpii = mpii['RELEASE'][0, 0]
    mpii.annolist = mpii.annolist.flatten()
    mpii.img_train = mpii.img_train.flatten()
    mpii.act = mpii.act.flatten()
    mpii.single_person = mpii.single_person.flatten()
    mpii.video_list = mpii.video_list.flatten()
    img_list = []
    for imgidx, alv in enumerate(mpii.annolist):
        img_train = mpii.img_train[imgidx]
        act = mpii.act[imgidx]
        single_person = mpii.single_person[imgidx].flatten()
        name = alv.image[0, 0].name[0]

        # annorect
        rect = alv.annorect.flatten()
        rect_list = list()
        for _, rv in enumerate(rect):
            rect_dict = dict()
            set_rect_field(rect_dict, rv, 'x1')
            set_rect_field(rect_dict, rv, 'y1')
            set_rect_field(rect_dict, rv, 'x2')
            set_rect_field(rect_dict, rv, 'y2')
            set_rect_scale(rect_dict, rv)
            set_rect_vidx(rect_dict, rv)
            set_rect_frame_sec(rect_dict, rv)
            set_rect_objpos(rect_dict, rv)
            set_rect_points_list(rect_dict, rv)
            rect_list.append(rect_dict)

        single_person = [i - 1 for i in single_person]
        act_dict = dict()
        act_name = act.act_name.flatten()
        cat_name = act.cat_name.flatten()
        set_act_name(act_dict, 'act_name', act_name)
        set_act_name(act_dict, 'cat_name', cat_name)
        act_dict['act_id'] = act.act_id[0, 0]
        if len(act_name) > 1 or len(cat_name) > 1:
            log.debug("%s %s", act_name, cat_name)

        value = dict()
        value['name'] = name
        value['rect'] = rect_list
        value['img_train'] = img_train
        value['single_person'] = single_person
        value['act'] = act_dict
        img_list.append(value)

    video_list = [i[0] for i in mpii.video_list]
    result = {"img_list": img_list,
              "video_list": video_list
              }
    return result


def json_default(obj):
    """
    numpy to json
    """
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj.item()
    return obj
