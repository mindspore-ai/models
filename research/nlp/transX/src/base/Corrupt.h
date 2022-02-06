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
#ifndef CORRUPT_H
#define CORRUPT_H

#include "Random.h"
#include "Reader.h"
#include "Triple.h"

INT corrupt_tail(INT id, INT h, INT r) {
  INT i_start, i_end, i_mid;
  INT hr_start, hr_end;

  // Find triplets sub-range in head->relation->tail sorted triplets array
  // (Using binary search)
  i_start = g_headStartIndices[h] - 1;
  i_end = g_headEndIndices[h];
  while (i_start + 1 < i_end) {
    i_mid = (i_start + i_end) >> 1;
    if (g_trainHead[i_mid].r >= r)
      i_end = i_mid;
    else
      i_start = i_mid;
  }
  hr_start = i_end;

  i_start = g_headStartIndices[h];
  i_end = g_headEndIndices[h] + 1;
  while (i_start + 1 < i_end) {
    i_mid = (i_start + i_end) >> 1;
    if (g_trainHead[i_mid].r <= r)
      i_start = i_mid;
    else
      i_end = i_mid;
  }
  hr_end = i_start;

  // Select the tail index different from tail indices
  // presented in the triplets with the specified head and relation.
  INT tmp = randMax(id, entityTotal - (hr_end - hr_start + 1));
  if (tmp < g_trainHead[hr_start].t)
    return tmp;
  if (tmp > g_trainHead[hr_end].t - hr_end + hr_start - 1)
    return tmp + hr_end - hr_start + 1;
  i_start = hr_start, i_end = hr_end + 1;
  while (i_start + 1 < i_end) {
    i_mid = (i_start + i_end) >> 1;
    if (g_trainHead[i_mid].t - i_mid + hr_start - 1 < tmp)
      i_start = i_mid;
    else
      i_end = i_mid;
  }
  return tmp + i_start - hr_start + 1;
}

INT corrupt_head(INT id, INT t, INT r) {
  INT i_start, i_end, i_mid;
  INT tr_start, tr_end;

  // Find triplets sub-range in tail->relation->head sorted triplets array
  // (Using binary search)
  i_start = g_tailStartIndices[t] - 1;
  i_end = g_tailEndIndices[t];
  while (i_start + 1 < i_end) {
    i_mid = (i_start + i_end) >> 1;
    if (g_trainTail[i_mid].r >= r)
      i_end = i_mid;
    else
      i_start = i_mid;
  }
  tr_start = i_end;

  i_start = g_tailStartIndices[t];
  i_end = g_tailEndIndices[t] + 1;
  while (i_start + 1 < i_end) {
    i_mid = (i_start + i_end) >> 1;
    if (g_trainTail[i_mid].r <= r)
      i_start = i_mid;
    else
      i_end = i_mid;
  }
  tr_end = i_start;

  // Select the head index different from head indices
  // presented in the triplets with the specified tail and relation.
  INT tmp = randMax(id, entityTotal - (tr_end - tr_start + 1));
  if (tmp < g_trainTail[tr_start].h)
    return tmp;
  if (tmp > g_trainTail[tr_end].h - tr_end + tr_start - 1)
    return tmp + tr_end - tr_start + 1;
  i_start = tr_start, i_end = tr_end + 1;
  while (i_start + 1 < i_end) {
    i_mid = (i_start + i_end) >> 1;
    if (g_trainTail[i_mid].h - i_mid + tr_start - 1 < tmp)
      i_start = i_mid;
    else
      i_end = i_mid;
  }
  return tmp + i_start - tr_start + 1;
}

#endif  // CORRUPT_H
