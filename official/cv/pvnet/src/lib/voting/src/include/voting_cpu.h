/*Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
*/

#ifndef RANSAC_VOTING_VOTING_CPU_H
#define RANSAC_VOTING_VOTING_CPU_H

#include <cassert>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cstring>

#define LOGD(format, ...) printf("[%s:%d](%s)" format"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
#define LOGT(format, ...) printf("[%s:%d](%s)" format"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
#define LOGI(format, ...) printf("[%s:%d](%s)" format"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
#define LOGW(format, ...) printf("[%s:%d](%s)" format"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
#define LOGE(format, ...) printf("[%s:%d](%s)" format"\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)
#define AR_UTIL_ASSERT(exp) do { assert(!!(exp)); } while (0)

class Vector2f {
 public:
     float x = 0.0f;
     float y = 0.0f;

 public:
     Vector2f() = default;
     Vector2f(float a, float b) : x(a), y(b) {}
     explicit Vector2f(float s) : x(s), y(s) {}
     Vector2f(const Vector2f& src) : x(src.x), y(src.y) {}

     ~Vector2f() = default;

     Vector2f& operator=(const Vector2f& b) {
         if (&b == this) {
             return *this;
         }

         x = b.x;
         y = b.y;
         return *this;
     }

     Vector2f operator-() const {
         return Vector2f(-x, -y);
     }

     Vector2f operator+(float s) const {
         return Vector2f(x + s, y + s);
     }

     Vector2f operator-(float s) const {
         return Vector2f(x - s, y - s);
     }

     Vector2f operator*(float s) const {
         return Vector2f(x * s, y * s);
     }

     Vector2f operator/(float s) const {
         return Vector2f(x / s, y / s);
     }

     Vector2f& operator*=(float s) {
         x *= s;
         y *= s;
         return *this;
     }

     Vector2f& operator/=(float s) {
         if (std::fabs(s) < FLT_EPSILON) {
             s = 1.0f;
         }

         x /= s;
         y /= s;
         return *this;
     }

     Vector2f operator+(const Vector2f& b) const {
         return Vector2f(x + b.x, y + b.y);
     }

     Vector2f operator-(const Vector2f& b) const {
         return Vector2f(x - b.x, y - b.y);
     }

     Vector2f& operator+=(const Vector2f& b) {
         x += b.x;
         y += b.y;
         return *this;
     }

     Vector2f& operator-=(const Vector2f& b) {
         x -= b.x;
         y -= b.y;
         return *this;
     }

     bool operator==(const Vector2f& b) const {
         return std::fabs(x - b.x) < FLT_EPSILON && std::fabs(y - b.y) < FLT_EPSILON;
     }

     // entrywise product
     Vector2f operator*(const Vector2f& b) const {
         return Vector2f(x * b.x, y * b.y);
     }

     Vector2f operator/(const Vector2f& b) const {
         return Vector2f(x / b.x, y / b.y);
     }

     float& operator[](int32_t idx) {
         AR_UTIL_ASSERT(0 <= idx && idx < 2);  // dimension 2
         return *(&x + idx);
     }

     const float& operator[](int32_t idx) const {
         AR_UTIL_ASSERT(0 <= idx && idx < 2);  // dimension 2
         return *(&x + idx);
     }

     static Vector2f Min(const Vector2f& a, const Vector2f& b) {
         return Vector2f((a.x < b.x) ? a.x : b.x, (a.y < b.y) ? a.y : b.y);
     }

     static Vector2f Max(const Vector2f& a, const Vector2f& b) {
         return Vector2f((a.x > b.x) ? a.x : b.x, (a.y > b.y) ? a.y : b.y);
     }

     // Used to calculate angle between two vectors among other things,
     // as (A dot B) = |a||b|cos(q).
     float Dot(const Vector2f& b) const {
         return x * b.x + y * b.y;
     }

     // length of the vector squared.
     float LengthSquare() const {
         return (x * x + y * y);
     }

     // vector length.
     float Length() const {
         return std::sqrt(LengthSquare());
     }

     // squared distance between two points represented by vectors.
     float distanceSqare(const Vector2f& b) const {
         return (*this - b).LengthSquare();
     }

     // distance between two points represented by vectors.
     float distance(const Vector2f& b) const {
         return (*this - b).Length();
     }

     // return a unit version of the vector without modifying itself.
     Vector2f Normalized() const {
         float l = Length();
         AR_UTIL_ASSERT(l != 0.0f);
         return *this / l;
     }
};

class Matrix2f {
 private:
     float m[4] = {1.0f, 0.0f, 0.0f, 1.0f};

 public:
     Matrix2f() = default;
     Matrix2f(float m00, float m01, float m10, float m11) {
         m[0] = m00;
         m[1] = m01;
         m[2] = m10;
         m[3] = m11;
     }
     Matrix2f(const Vector2f& a, const Vector2f& b) {
         m[0] = a.x;
         m[1] = a.y;
         m[2] = b.x;
         m[3] = b.y;
     }

     ~Matrix2f() = default;

     Matrix2f& operator=(const Matrix2f& b) {
         memcpy(m, b.m, sizeof(float) * 4);
         return *this;
     }

     Matrix2f operator*(float s) const {
         return Matrix2f(m[0] * s, m[1] * s, m[2] * s, m[3] * s);
     }

     Matrix2f operator*(const Matrix2f& b) const {
         float m0 = m[0] * b[0] + m[1] * b[2];
         float m1 = m[0] * b[1] + m[1] * b[3];
         float m2 = m[2] * b[0] + m[3] * b[2];
         float m3 = m[2] * b[1] + m[3] * b[3];
         return Matrix2f(m0, m1, m2, m3);
     }

     Vector2f operator*(const Vector2f& b) const {
         return Vector2f(m[0] * b.x + m[1] * b.y, m[2] * b.x + m[3] * b.y);
     }

     Matrix2f& operator*=(float s) {
         m[0] = m[0] * s;
         m[1] = m[1] * s;
         m[2] = m[2] * s;
         m[3] = m[3] * s;
         return *this;
     }

     Matrix2f operator+(const Matrix2f& b) const {
         return Matrix2f(m[0] + b[0], m[1] + b[1], m[2] + b[2], m[3] + b[3]);
     }

     Matrix2f& operator+=(const Matrix2f& b) {
         m[0] += b[0];
         m[1] += b[1];
         m[2] += b[2];
         m[3] += b[3];
         return *this;
     }

     Matrix2f operator/(float s) const {
         return Matrix2f(m[0] / s, m[1] / s, m[2] / s, m[3] / s);
     }

     Matrix2f& operator/=(float s) {
         if (std::fabs(s) < FLT_EPSILON) {
             s = 1.0f;
         }

         m[0] = m[0] / s;
         m[1] = m[1] / s;
         m[2] = m[2] / s;
         m[3] = m[3] / s;
         return *this;
     }


     float& operator[](int32_t idx) {
         AR_UTIL_ASSERT(0 <= idx && idx < 4);  // length 2
         return m[idx];
     }

     const float& operator[](int32_t idx) const {
         AR_UTIL_ASSERT(0 <= idx && idx < 4);  // length 4
         return m[idx];
     }

     float DetA() {
         return m[0] * m[3] - m[1] * m[2];     // ad-bc
     }

     Matrix2f Inverse() {
         float detA = DetA();
         AR_UTIL_ASSERT(std::fabs(detA) > FLT_EPSILON);

         float scale = 1.0f / detA;
         return Matrix2f(m[3] * scale, -m[1] * scale, -m[2] * scale, m[0] * scale);
     }

     Matrix2f Transpose() {
         return Matrix2f(m[0], m[2], m[1], m[3]);
     }

     static Matrix2f FromAxB(const Vector2f& a, const Vector2f& b) {
         return Matrix2f(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y);
     }
};

struct ModelShape {
    uint32_t H = 0;
    uint32_t W = 0;
    uint32_t C = 0;
};

struct VotingOptions {
    float inlierThresh = 0.998f;   // cos_angle thresh : 0.99 * 0.99
    float confidence = 0.99f;
    uint32_t maxIter = 100;
    uint32_t minNum = 1000;
    uint32_t maxNum = 10000;
};

class VotingProcess {
 public:
     VotingProcess() = default;
     VotingProcess(const VotingProcess&) = delete;
     VotingProcess& operator=(const VotingProcess&) = delete;
     ~VotingProcess() = default;

     int32_t Init(const ModelShape &outputShape, int32_t classNum, int32_t controlPointNum);

     int32_t PostProcess(const std::vector<float> &data, int32_t &classId, std::vector<float> &box2D);

 private:
     int32_t FindClassAndMask(const std::vector<float> &data, int32_t &classId);

     Vector2f SelectVoteKeyPoint(uint32_t index) const;

     int32_t RefinePosition(uint32_t kpIndex, const std::vector<uint32_t> &inliers, Vector2f &keypoint) const;

     Vector2f GetHypoPoint(uint32_t kpIndex) const;

     inline Vector2f PixelIndex2XY(uint32_t pointIdx) const {
         uint32_t pointX = pointIdx % mNetShape.W;
         uint32_t pointY = (pointIdx - pointX) / mNetShape.W;
         return Vector2f(static_cast<float>(pointX), static_cast<float>(pointY));
     }

     inline Vector2f GetNormal(uint32_t pointIdx, uint32_t layerIndex) const {
         auto xMap = mVectorMapAddr[2 * layerIndex];
         auto yMap = mVectorMapAddr[2 * layerIndex + 1];

         return Vector2f(-yMap[pointIdx], xMap[pointIdx]);
     }

     inline Vector2f GetVoteVec(uint32_t pointIdx, uint32_t layerIndex) const {
         auto xMap = mVectorMapAddr[2 * layerIndex];
         auto yMap = mVectorMapAddr[2 * layerIndex + 1];

         return Vector2f(xMap[pointIdx], yMap[pointIdx]);
     }

 private:
     bool mInitial = false;
     ModelShape mNetShape = {};
     uint32_t mImgSize = 0;
     uint32_t mClassMax = 0;
     uint32_t mControlPoints = 9;  // 9 corners default

     std::vector<char> mSegMask = {};
     std::vector<uint32_t > mPixel2DInMask = {};
     std::vector<const float* > mVectorMapAddr = {};

     VotingOptions mOptions = {0.99f, 0.99f, 100, 1000, 10000};
};
#endif  // RANSAC_VOTING_VOTING_CPU_H
