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
#include "Corrupt.h"
#include "Random.h"
#include "Reader.h"
#include "Setting.h"
#include <pthread.h>

extern "C" void setSeed(unsigned int seed);

extern "C" void loadDataset(INT relationNum, INT entityNum, INT tripletsNum,
                            const INT *tripletsArray);

struct ThreadParameters {
  INT threadIndex;
  INT *subBatchHrtBuffer;
  INT subBatchSize;
  INT *subBatchIndices;
  INT negRate;
};

void *prepare_batch(void *con) {
  auto *para = reinterpret_cast<ThreadParameters *>(con);
  INT threadIndex = para->threadIndex;
  INT *sub_batch_hrt_buffer = para->subBatchHrtBuffer;
  INT sub_batchSize = para->subBatchSize;
  INT *sub_batchIndices = para->subBatchIndices;
  INT negRate = para->negRate;

  for (INT i = 0; i < sub_batchSize; i++) {
    INT srcTripletIndex = sub_batchIndices[i];
    INT targetIndex = i * (1 + negRate) * 3;

    INT *tripletBuf = &sub_batch_hrt_buffer[targetIndex];
    tripletBuf[0] = g_trainList[srcTripletIndex].h;
    tripletBuf[1] = g_trainList[srcTripletIndex].r;
    tripletBuf[2] = g_trainList[srcTripletIndex].t;

    for (INT times = 1; times <= negRate; times++) {
      tripletBuf = &sub_batch_hrt_buffer[targetIndex + times * 3];
      if (static_cast<float>(randd(threadIndex) % 1000) <
          g_headCorruptProb[g_trainList[srcTripletIndex].r]) {
        tripletBuf[0] = g_trainList[srcTripletIndex].h;
        tripletBuf[1] = g_trainList[srcTripletIndex].r;
        tripletBuf[2] =
            corrupt_tail(threadIndex, g_trainList[srcTripletIndex].h,
                         g_trainList[srcTripletIndex].r);
      } else {
        tripletBuf[0] =
            corrupt_head(threadIndex, g_trainList[srcTripletIndex].t,
                         g_trainList[srcTripletIndex].r);
        tripletBuf[1] = g_trainList[srcTripletIndex].r;
        tripletBuf[2] = g_trainList[srcTripletIndex].t;
      }
    }
  }

  pthread_exit(nullptr);
}

extern "C" void getBatch(INT *batchHrtBuffer, INT *batchIndices, INT batchSize,
                         INT negRate, INT workThreads) {
  if (workThreads < 1)
    workThreads = 1;
  if (workThreads > MAX_THREADS)
    workThreads = MAX_THREADS;

  auto *pThreadsHandles = new pthread_t[workThreads];
  auto *threadsParameters = new ThreadParameters[workThreads];

  INT subBatchSize;

  if (batchSize % workThreads == 0) {
    subBatchSize = batchSize / workThreads;
  } else {
    subBatchSize = batchSize / workThreads + 1;
  }

  for (INT threadIndex = 0; threadIndex < workThreads; threadIndex++) {
    auto *threadParams = &threadsParameters[threadIndex];

    threadParams->threadIndex = threadIndex;
    threadParams->subBatchHrtBuffer =
        batchHrtBuffer + threadIndex * subBatchSize * (1 + negRate) * 3;
    threadParams->subBatchIndices = batchIndices + threadIndex * subBatchSize;

    if ((threadIndex + 1) == workThreads)
      threadParams->subBatchSize = batchSize - (workThreads - 1) * subBatchSize;
    else
      threadParams->subBatchSize = subBatchSize;

    threadParams->negRate = negRate;
    pthread_create(&pThreadsHandles[threadIndex], nullptr, prepare_batch,
                   threadParams);
  }
  for (INT threads = 0; threads < workThreads; threads++)
    pthread_join(pThreadsHandles[threads], nullptr);

  delete[] pThreadsHandles;
  delete[] threadsParameters;
}

int main() {
  setSeed(0);
  return 0;
}
