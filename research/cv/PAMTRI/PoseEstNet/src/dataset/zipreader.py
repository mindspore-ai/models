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
"""zipreader"""
import os
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np

_im_zfile = []
_xml_path_zip = []
_xml_zfile = []

def imread(filename, flags=cv2.IMREAD_COLOR):
    """imread"""
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    path_img = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

def xmlread(filename):
    """xmlread"""
    global _xml_path_zip
    global _xml_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    path_xml = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in range(len(_xml_path_zip)):
        if _xml_path_zip[i] == path_zip:
            data = _xml_zfile[i].open(path_xml)
            return ET.fromstring(data.read())
    _xml_path_zip.append(path_zip)
    print("read new xml file '%s'"%(path_zip))
    _xml_zfile.append(zipfile.ZipFile(path_zip, 'r'))
    data = _xml_zfile[-1].open(path_xml)
    return ET.fromstring(data.read())
