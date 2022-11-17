/*
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef EAST_RECOGNITION_H
#define EAST_RECOGNITION_H

#include <limits>
#include <algorithm>
#include <vector>
#include "clipper/clipper.hpp"

// locality-aware NMS
namespace lanms {
namespace cl = ClipperLib;

struct Polygon {
    cl::Path poly;
    float score;
};

float PathsArea(const ClipperLib::Paths &paths);

float PolyIou(const Polygon &polygonA, const Polygon &polygonB);

bool ShouldMerge(const Polygon &polygonA, const Polygon &polygonB, float iou_threshold);

/**
* Incrementally merge polygons
*/
class PolyMerger {
 public:
    PolyMerger(): score_(0), nrPolys_(0) {
        memset(data_, 0, sizeof(data_));
    }

    /**
     * Add a new polygon to be merged.
     */
    void Add(const Polygon &given);
    std::int64_t sqr(std::int64_t x);
    Polygon NormalizePoly(const Polygon &ref, const Polygon &polygon);
    Polygon Get() const;

 private:
    std::int64_t data_[8];
    float score_;
    std::int32_t nrPolys_;
};


/**
* The standard NMS algorithm.
*/
std::vector<Polygon> StandardNms(const std::vector<Polygon> &polys, float iou_threshold);

std::vector<Polygon>
MergeQuadrangleN9(const std::vector<std::vector<float>> &data, float iou_threshold);
}  // namespace lanms
#endif
