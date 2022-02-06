/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef READER_H
#define READER_H

#include "Setting.h"
#include "Triple.h"
#include <algorithm>
#include <stdexcept>

INT *g_headStartIndices, *g_headEndIndices;
INT *g_tailStartIndices, *g_tailEndIndices;
REAL *g_headCorruptProb;

Triple *g_trainList;
Triple *g_trainHead;
Triple *g_trainTail;

extern "C"
void loadDataset(INT relationNum, INT entityNum, INT tripletsNum, const INT *tripletsArray) {
    INT *freqRel;
    REAL *leftMean, *rightMean;

    relationTotal = relationNum;
    entityTotal = entityNum;
    trainTotal = tripletsNum;

    g_trainList = new Triple[trainTotal]();
    g_trainHead = new Triple[trainTotal]();
    g_trainTail = new Triple[trainTotal]();

    for (INT i = 0; i < trainTotal; i++) {
        auto *tripletsPointer = &tripletsArray[i * 3];
        g_trainList[i].h = tripletsPointer[0];
        g_trainList[i].r = tripletsPointer[1];
        g_trainList[i].t = tripletsPointer[2];
    }

    freqRel = new INT[relationTotal]();
    for (INT i = 0; i < trainTotal; i++) freqRel[g_trainList[i].r]++;

    std::memcpy(g_trainHead, g_trainList, sizeof(Triple) * trainTotal);
    std::memcpy(g_trainTail, g_trainList, sizeof(Triple) * trainTotal);

    std::sort(g_trainHead, g_trainHead + trainTotal, Triple::cmp_head);
    std::sort(g_trainTail, g_trainTail + trainTotal, Triple::cmp_tail);

    g_headStartIndices = new INT[entityTotal]();
    g_headEndIndices = new INT[entityTotal]();
    g_tailStartIndices = new INT[entityTotal]();
    g_tailEndIndices = new INT[entityTotal]();

    memset(g_headEndIndices, -1, sizeof(INT) * entityTotal);
    memset(g_tailEndIndices, -1, sizeof(INT) * entityTotal);

    for (INT i = 1; i < trainTotal; i++) {
        if (g_trainTail[i].t != g_trainTail[i - 1].t) {
            g_tailEndIndices[g_trainTail[i - 1].t] = i - 1;
            g_tailStartIndices[g_trainTail[i].t] = i;
        }
        if (g_trainHead[i].h != g_trainHead[i - 1].h) {
            g_headEndIndices[g_trainHead[i - 1].h] = i - 1;
            g_headStartIndices[g_trainHead[i].h] = i;
        }
    }
    g_headStartIndices[g_trainHead[0].h] = 0;
    g_headEndIndices[g_trainHead[trainTotal - 1].h] = trainTotal - 1;
    g_tailStartIndices[g_trainTail[0].t] = 0;
    g_tailEndIndices[g_trainTail[trainTotal - 1].t] = trainTotal - 1;

    leftMean = new REAL[relationTotal]();
    rightMean = new REAL[relationTotal]();

    for (INT i = 0; i < entityTotal; i++) {
        for (INT j = g_headStartIndices[i] + 1; j <= g_headEndIndices[i]; j++) {
            if (g_trainHead[j].r != g_trainHead[j - 1].r) {
                leftMean[g_trainHead[j].r] += 1.0;
            }
        }

        if (g_headStartIndices[i] <= g_headEndIndices[i]) {
            leftMean[g_trainHead[g_headStartIndices[i]].r] += 1.0;
        }

        for (INT j = g_tailStartIndices[i] + 1; j <= g_tailEndIndices[i]; j++) {
            if (g_trainTail[j].r != g_trainTail[j - 1].r) {
                rightMean[g_trainTail[j].r] += 1.0;
            }
        }

        if (g_tailStartIndices[i] <= g_tailEndIndices[i]) {
            rightMean[g_trainTail[g_tailStartIndices[i]].r] += 1.0;
        }
    }

    g_headCorruptProb = new REAL[entityTotal];

    for (INT i = 0; i < relationTotal; i++) {
        leftMean[i] = static_cast<float>(freqRel[i]) / leftMean[i];
        rightMean[i] = static_cast<float>(freqRel[i]) / rightMean[i];

        g_headCorruptProb[i] = 1000 * rightMean[i] / (rightMean[i] + leftMean[i]);
    }

    delete[] freqRel;
    delete[] leftMean;
    delete[] rightMean;
}

#endif  // READER_H
