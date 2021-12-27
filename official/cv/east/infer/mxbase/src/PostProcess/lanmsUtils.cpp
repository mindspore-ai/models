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
#include <cassert>
#include <limits>
#include <numeric>
#include <algorithm>
#include <vector>
#include "lanmsUtils.h"
#include "clipper/clipper.hpp"

namespace {
    const int SHAPE_SIZE = 4;
    const float SCORE_THRESH = 0.0;
}

namespace lanms {
    namespace cl = ClipperLib;

    float PathsArea(const ClipperLib::Paths &paths) {
        float area = 0;
        for (auto &&path : paths)
            area += cl::Area(path);
        return area;
    }

    float PolyIou(const Polygon &polygonA, const Polygon &polygonB) {
        cl::Clipper clpr;
        clpr.AddPath(polygonA.poly, cl::ptSubject, true);
        clpr.AddPath(polygonB.poly, cl::ptClip, true);

        cl::Paths inter, uni;
        clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
        clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);

        auto inter_area = PathsArea(inter),
                uni_area = PathsArea(uni);
        return std::abs(inter_area) / std::max(std::abs(uni_area), 1.0f);
    }

    bool ShouldMerge(const Polygon &polygonA, const Polygon &polygonB, float iou_threshold) {
        return PolyIou(polygonA, polygonB) > iou_threshold;
    }

    /**
     * Add a new polygon to be merged.
     */
    void PolyMerger::Add(const Polygon &given) {
        Polygon polygon;
        if (nrPolys_ > 0) {
            // vertices of two polygons to merge may not in the same order;
            // we match their vertices by choosing the ordering that
            // minimizes the total squared distance.
            // see function normalize_poly for details.
            polygon = NormalizePoly(Get(), given);
        } else {
            polygon = given;
        }
        assert(polygon.poly.size() == SHAPE_SIZE);
        const auto &poly = polygon.poly;
        auto s = polygon.score;
        data_[0] += poly[0].X * s;
        data_[1] += poly[0].Y * s;

        data_[2] += poly[1].X * s;
        data_[3] += poly[1].Y * s;

        data_[4] += poly[2].X * s;
        data_[5] += poly[2].Y * s;

        data_[6] += poly[3].X * s;
        data_[7] += poly[3].Y * s;

        score_ += polygon.score;

        nrPolys_ += 1;
    }

    inline std::int64_t PolyMerger::sqr(std::int64_t x) { return x * x; }

    Polygon PolyMerger::NormalizePoly(const Polygon &ref, const Polygon &polygon) {
        std::int64_t min_d = std::numeric_limits<std::int64_t>::max();
        size_t best_start = 0, best_order = 0;

        for (size_t start = 0; start < SHAPE_SIZE; start++) {
            size_t j = start;
            std::int64_t d = (
                    sqr(ref.poly[(j + 0) % SHAPE_SIZE].X - polygon.poly[(j + 0) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 0) % SHAPE_SIZE].Y - polygon.poly[(j + 0) % SHAPE_SIZE].Y)
                    + sqr(ref.poly[(j + 1) % SHAPE_SIZE].X - polygon.poly[(j + 1) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 1) % SHAPE_SIZE].Y - polygon.poly[(j + 1) % SHAPE_SIZE].Y)
                    + sqr(ref.poly[(j + 2) % SHAPE_SIZE].X - polygon.poly[(j + 2) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 2) % SHAPE_SIZE].Y - polygon.poly[(j + 2) % SHAPE_SIZE].Y)
                    + sqr(ref.poly[(j + 3) % SHAPE_SIZE].X - polygon.poly[(j + 3) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 3) % SHAPE_SIZE].Y - polygon.poly[(j + 3) % SHAPE_SIZE].Y));
            if (d < min_d) {
                min_d = d;
                best_start = start;
                best_order = 0;
            }

            d = (
                    sqr(ref.poly[(j + 0) % SHAPE_SIZE].X - polygon.poly[(j + 3) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 0) % SHAPE_SIZE].Y - polygon.poly[(j + 3) % SHAPE_SIZE].Y)
                    + sqr(ref.poly[(j + 1) % SHAPE_SIZE].X - polygon.poly[(j + 2) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 1) % SHAPE_SIZE].Y - polygon.poly[(j + 2) % SHAPE_SIZE].Y)
                    + sqr(ref.poly[(j + 2) % SHAPE_SIZE].X - polygon.poly[(j + 1) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 2) % SHAPE_SIZE].Y - polygon.poly[(j + 1) % SHAPE_SIZE].Y)
                    + sqr(ref.poly[(j + 3) % SHAPE_SIZE].X - polygon.poly[(j + 0) % SHAPE_SIZE].X)
                    + sqr(ref.poly[(j + 3) % SHAPE_SIZE].Y - polygon.poly[(j + 0) % SHAPE_SIZE].Y));
            if (d < min_d) {
                min_d = d;
                best_start = start;
                best_order = 1;
            }
        }

        Polygon polygonR;
        polygonR.poly.resize(SHAPE_SIZE);
        auto j = best_start;
        if (best_order == 0) {
            for (size_t i = 0; i < SHAPE_SIZE; i++)
                polygonR.poly[i] = polygon.poly[(j + i) % SHAPE_SIZE];
        } else {
            for (size_t i = 0; i < SHAPE_SIZE; i++)
                polygonR.poly[i] = polygon.poly[(j + SHAPE_SIZE - i - 1) % SHAPE_SIZE];
        }
        polygonR.score = polygon.score;
        return polygonR;
    }

    Polygon PolyMerger::Get() const {
        Polygon polygon;

        auto &poly = polygon.poly;
        poly.resize(SHAPE_SIZE);
        auto score_inv = 1.0f / std::max(1e-8f, score_);
        poly[0].X = data_[0] * score_inv;
        poly[0].Y = data_[1] * score_inv;
        poly[1].X = data_[2] * score_inv;
        poly[1].Y = data_[3] * score_inv;
        poly[2].X = data_[4] * score_inv;
        poly[2].Y = data_[5] * score_inv;
        poly[3].X = data_[6] * score_inv;
        poly[3].Y = data_[7] * score_inv;

        assert(score_ > SCORE_THRESH);
        polygon.score = score_;

        return polygon;
    }

    /**
    * The standard NMS algorithm.
    */
    std::vector<Polygon> StandardNms(const std::vector<Polygon> &polys, float iou_threshold) {
        size_t n = polys.size();
        if (n == 0)
            return {};
        std::vector<size_t> indices(n);
        std::iota(std::begin(indices), std::end(indices), 0);
        std::sort(std::begin(indices), std::end(indices),
                  [&](size_t i, size_t j) { return polys[i].score > polys[j].score; });

        std::vector<size_t> keep;
        while (indices.size()) {
            size_t p = 0, cur = indices[0];
            keep.emplace_back(cur);
            for (size_t i = 1; i < indices.size(); i++) {
                if (!ShouldMerge(polys[cur], polys[indices[i]], iou_threshold)) {
                    indices[p++] = indices[i];
                }
            }
            indices.resize(p);
        }

        std::vector<Polygon> ret(keep.size());
        std::transform(keep.begin(), keep.end(), ret.begin(), [&](size_t i) { return polys[i]; });
        return ret;
    }

    std::vector<Polygon>
    MergeQuadrangleN9(const std::vector<std::vector<float>> &data, float iou_threshold) {
        using cInt = cl::cInt;

        // first pass
        std::vector<Polygon> polys;
        for (size_t i = 0; i < data.size(); i++) {
            auto p = data[i];
            Polygon poly{
                    {
                            {cInt(p[0]), cInt(p[1])},
                            {cInt(p[2]), cInt(p[3])},
                            {cInt(p[4]), cInt(p[5])},
                            {cInt(p[6]), cInt(p[7])},
                    },
                    p[8],
            };

            if (polys.size()) {
                // merge with the last one
                auto &bpoly = polys.back();
                if (ShouldMerge(poly, bpoly, iou_threshold)) {
                    PolyMerger merger;
                    merger.Add(bpoly);
                    merger.Add(poly);
                    bpoly = merger.Get();
                } else {
                    polys.emplace_back(poly);
                }
            } else {
                polys.emplace_back(poly);
            }
        }
        return StandardNms(polys, iou_threshold);
    }
}  // namespace lanms


