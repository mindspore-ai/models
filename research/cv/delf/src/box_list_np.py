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
"""bbox definition"""
import numpy as np

class BoxList:
    """Box collection."""

    def __init__(self, boxes):
        """Constructs box collection.

        Args:
            boxes: an array of shape [N, 4] representing box corners

        Raises:
            ValueError: if invalid dimensions for bbox data or if bbox data is not in
                    float32 format.
        """
        if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
            raise ValueError('Invalid dimensions for box data: {}'.format(
                boxes.shape))
        if boxes.dtype != np.float32:
            raise ValueError('Invalid array type: should be np.float32')
        self.data = {'boxes': boxes}

    def num_boxes(self):
        """Returns number of boxes held in collection.

        Returns:
            an array representing the number of boxes held in the collection.
        """
        return self.data['boxes'].shape[0]


    def get_all_fields(self):
        """Returns all fields."""
        return self.data.keys()

    def get_extra_fields(self):
        """Returns all non-box fields (i.e., everything not named 'boxes')."""
        return [k for k in self.data if k != 'boxes']

    def add_field(self, field, field_data):
        """Add field to box list.

        This method can be used to add related box data such as
        weights/labels, etc.

        Args:
            field: a string key to access the data via `get`
            field_data: an array containing the data to store in the BoxList
        """
        self.data[field] = field_data

    def has_field(self, field):
        return field in self.data

    def get(self):
        """Convenience function for accessing box coordinates.

        Returns:
            an array with shape [N, 4] representing box coordinates.
        """
        return self.get_field('boxes')

    def get_field(self, field):
        """Accesses a box collection and associated fields.

        This function returns specified field with object; if no field is specified,
        it returns the box coordinates.

        Args:
            field: this optional string parameter can be used to specify
                a related field to be accessed.

        Returns:
            an array representing the box collection or an associated field.

        Raises:
            ValueError: if invalid field
        """
        if not self.has_field(field):
            raise ValueError('field ' + str(field) + ' does not exist')
        return self.data[field]
