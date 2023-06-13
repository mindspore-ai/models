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
Useful functions
"""
import sys
import os
from math import log

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


def find_entropy(stream):
    return entropy(counts, base=2)


def node_data2int(data):
    value = data * np.array([1, 2, 4, 8, 16, 32, 64, 128])
    return np.sum(value)


def int2node_data(value):
    data = np.zeros(8, dtype=np.int)
    value_vector = np.repeat(value, 8)
    tmp = np.floor(value_vector / [1, 2, 4, 8, 16, 32, 64, 128])
    data = np.mod(tmp, 2)
    return data.astype(int)


def describe_element(name, df):
    """Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {"f": "float", "u": "uchar", "i": "int"}
    element = ["element " + name + " " + str(len(df))]

    if name == "face":
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append("property " + f + " " + df.columns.values[i])

    return element


def write_ply(filename, points=None, mesh=None, as_text=True):
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    Returns
    -------
    boolean
        True if no problems
    """
    if not filename.endswith("ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as ply:
        header = ["ply"]

        if as_text:
            header.append("format ascii 1.0")
        else:
            header.append("format binary_" + sys.byteorder + "_endian 1.0")

        if points is not None:
            header.extend(describe_element("vertex", points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element("face", mesh))

        header.append("end_header")

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode="a", encoding="ascii")
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode="a", encoding="ascii")

    else:
        with open(filename, "ab") as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True


def bin_loader(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def file_in_folder(folder_path, extension):
    frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            frames.append("{}".format(filename))
    return frames


def entropy_from_pro(probs, base=None):
    entropy_value = 0
    base = 2 if base is None else base
    for prob in probs:
        entropy_value -= log(prob, base)
    return entropy_value


def square_distance(src, dst):
    return np.sum((src[:, None] - dst[None]) ** 2, axis=-1)


def nn_distance(src, ref, n=1):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(ref)
    distances, _ = nbrs.kneighbors(src)
    return distances


def chamfer_distance(recon_pcd, gt_pcd):
    distance_recon_gt = nn_distance(recon_pcd, gt_pcd)
    distance_gt_recon = nn_distance(gt_pcd, recon_pcd)

    scd = np.mean(distance_recon_gt) + np.mean(distance_gt_recon)
    return scd
