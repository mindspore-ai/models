# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from common.network import SRNet

mode_pad_dict = {
    "s": 1,
    "d": 2,
    "y": 2,
    "c": 3,
    "t": 3,
}

rint_op = ops.Rint()


class LeRFNet(nn.Cell):
    def __init__(self, opt, in_c=1, out_c=1):
        super(LeRFNet, self).__init__()
        self.modes = opt.modes
        self.modes2 = opt.modes2
        self.stages = opt.stages
        self.norm = opt.norm
        nf = opt.nf

        # self.stages=2 w/ pre-filtering
        for s in range(self.stages):
            if (s + 1) == self.stages:
                upscale = 1
                o_c = out_c
                flag = "N"
                # hyper stage
                for mode in self.modes2:
                    for r in [0, 1]:
                        self.insert_child_to_cell(
                            "s{}_{}r{}".format(str(s + 1), mode, r),
                            SRNet(
                                "{}x{}".format(mode.upper(), flag),
                                nf=nf,
                                upscale=upscale,
                                out_c=o_c,
                            ),
                        )
            else:
                upscale = None
                o_c = 1
                flag = "1"
                for mode in self.modes:
                    self.insert_child_to_cell(
                        "s{}_{}r0".format(str(s + 1), mode),
                        SRNet(
                            "{}x{}".format(mode.upper(), flag),
                            nf=nf,
                            upscale=upscale,
                            out_c=o_c,
                        ),
                    )

    def construct(self, x, stage, mode, r):
        key = "s{}_{}r{}".format(str(stage), mode, r)
        module = getattr(self, key)
        return module(x)

    def predict(self, x, stage=1):
        if stage == 2:  # hyper stage
            pred = 0
            for mode in self.modes2:
                pad = mode_pad_dict.get(mode, "")
                pad_op = nn.Pad(
                    paddings=((0, 0), (0, 0), (0, pad), (0, pad)), mode="SYMMETRIC"
                )
                for r in [0, 2]:
                    pred += rint_op(
                        ms.numpy.rot90(
                            self.construct(
                                pad_op(ms.numpy.rot90(x, r, [2, 3])),
                                stage=self.stages,
                                mode=mode,
                                r=0,
                            ),
                            (4 - r) % 4,
                            [2, 3],
                        )
                        * 127
                    )
                for r in [1, 3]:
                    pred += rint_op(
                        ms.numpy.rot90(
                            self.construct(
                                pad_op(ms.numpy.rot90(x, r, [2, 3])),
                                stage=self.stages,
                                mode=mode,
                                r=1,
                            ),
                            (4 - r) % 4,
                            [2, 3],
                        )
                        * 127
                    )

            avg_factor, bias, norm = (
                len(self.modes2) * 4,
                self.norm // 2,
                float(self.norm),
            )
            x = (
                ops.clip_by_value(rint_op((pred / avg_factor) + bias), 0, self.norm)
                / norm
            )
        else:  # pre-filtering stage
            for s in range(self.stages - 1):
                pred = 0
                for mode in self.modes:
                    pad = mode_pad_dict.get(mode, "")
                    pad_op = nn.Pad(
                        paddings=((0, 0), (0, 0), (0, pad), (0, pad)), mode="SYMMETRIC"
                    )
                    for r in [0, 1, 2, 3]:
                        pred += rint_op(
                            ms.numpy.rot90(
                                self.construct(
                                    pad_op(ms.numpy.rot90(x, r, [2, 3])),
                                    stage=s + 1,
                                    mode=mode,
                                    r=0,
                                ),
                                (4 - r) % 4,
                                [2, 3],
                            )
                            * (self.norm // 2)
                        )
                if (s + 1) == (self.stages - 1):
                    avg_factor, bias, norm = len(self.modes), 0, 1
                else:
                    avg_factor, bias, norm = (
                        len(self.modes) * 4,
                        self.norm // 2,
                        float(self.norm),
                    )
                x = (
                    ops.clip_by_value(rint_op((pred / avg_factor) + bias), 0, self.norm)
                    / norm
                )

        return x
