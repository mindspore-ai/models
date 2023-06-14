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
Implement octree.
"""

import numpy as np


def int2node_data(value):
    data = np.zeros(8, dtype=np.int)
    value_vector = np.repeat(value, 8)
    tmp = np.floor(value_vector / [1, 2, 4, 8, 16, 32, 64, 128])
    data = np.mod(tmp, 2)
    return data.astype(int)


class OctreeBranchNode:
    def __init__(self, depth, parent, position, size, octant):
        """
        Here we use number to simply eight branches
        + here means greater then the center, vice versa
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +

        Parameters
        ----------
        position: ndarray
            center of voxel/node
        """
        self.attribute = "branch"
        self.depth = depth
        self.size = size
        self.position = position
        self.parent = parent
        self.data = np.zeros(8, dtype=np.int)
        self.branches = [None, None, None, None, None, None, None, None]
        self.octant = octant

    def __str__(self):
        return "position: {0}, size: {1}, depth: {2} parent: {3}, data: {4}".format(
            self.position, self.size, self.depth, self.parent, self.data
        )


class OctreeLeafNode:
    def __init__(self, depth, parent, position_gt, size):
        """
        leafnode do not need branch and data
        Different with position in the branch, position_gt here is not the center of voxel bur ground truth xyz
        """
        self.attribute = "leaf"
        self.depth = depth
        self.size = size
        self.position_center = None
        self.position_gt = position_gt
        self.parent = parent


class Octree:
    def __init__(self, max_range, precision, origin=(0, 0, 0)):
        """
        Parameters
        ----------
        max_range:
            the size of whole octree
        origin:
            the position of octree root node
        leaf_num:
            number of lead nodes, aka the number of points after converting into octree
        """
        self.attribute = "root"
        self.size = max_range
        self.precision = precision
        self.depth = 0
        self.max_depth = int(np.log2(max_range / precision)) + 1
        self.position = origin
        self.data = np.zeros(8, dtype=np.int)
        self.branches = [None, None, None, None, None, None, None, None]
        self.leaf_num = 0

    @staticmethod
    def find_branch(root, position):
        """
        helper function
        returns an index corresponding to a branch
        pointing in the direction we want to go
        """
        index = 0
        if position[0] >= root.position[0]:
            index |= 4
        if position[1] >= root.position[1]:
            index |= 2
        if position[2] >= root.position[2]:
            index |= 1
        return index

    def insert_node(self, root, size, parent, position, depth):
        if depth == self.max_depth:
            if root is None:
                self.leaf_num += 1
            return OctreeLeafNode(depth, parent, position, size)

        if depth < self.max_depth:
            branch = self.find_branch(root, position)
            branch_size = root.size / 2
            root.data[branch] = 1

            if (root.branches[branch] is None) and (depth != self.max_depth - 1):
                pos = root.position
                offset = size / 2
                new_center = (0, 0, 0)
                if branch == 0:
                    new_center = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
                elif branch == 1:
                    new_center = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
                elif branch == 2:
                    new_center = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
                elif branch == 3:
                    new_center = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
                elif branch == 4:
                    new_center = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
                elif branch == 5:
                    new_center = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
                elif branch == 6:
                    new_center = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
                elif branch == 7:
                    new_center = (pos[0] + offset, pos[1] + offset, pos[2] + offset)
                root.branches[branch] = OctreeBranchNode(depth + 1, root, new_center, branch_size, branch)

            root.branches[branch] = self.insert_node(root.branches[branch], branch_size, root, position, depth + 1)
        return root

    def serialize_depth_first(self):
        def extract_data(node):
            if node and node.attribute != "leaf":
                values.append(node_data2int(node.data))
                # vals.append(node.data)
                for branch_id in range(8):
                    extract_data(node.branches[branch_id])

        values = []
        extract_data(self)
        return values


def deserialize_depth_first(values, max_depth, octree):
    def reconstruct_tree(root):
        value = next(values)
        node_data = int2node_data(value)
        root.data = node_data
        for branch in range(8):
            if node_data[branch] == 0:
                root.branches[branch] = None
            else:
                depth = root.depth
                pos = root.position
                offset = root.size / 2
                new_center = (0, 0, 0)

                if branch == 0:
                    new_center = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
                elif branch == 1:
                    new_center = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
                elif branch == 2:
                    new_center = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
                elif branch == 3:
                    new_center = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
                elif branch == 4:
                    new_center = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
                elif branch == 5:
                    new_center = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
                elif branch == 6:
                    new_center = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
                elif branch == 7:
                    new_center = (pos[0] + offset, pos[1] + offset, pos[2] + offset)

                if depth < max_depth - 1:
                    node = OctreeBranchNode(depth + 1, root, new_center, offset, branch)
                    root.branches[branch] = node
                    reconstruct_tree(root.branches[branch])
                else:
                    node = OctreeLeafNode(depth + 1, root, 0, offset)
                    node.position_center = new_center
                    recon_points.append(new_center)
                    root.branches[branch] = node
        return root

    recon_points = []
    return reconstruct_tree(octree), recon_points
