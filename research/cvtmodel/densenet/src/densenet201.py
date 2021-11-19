import mindspore.ops as P
from mindspore import nn


class Module0(nn.Cell):
    def __init__(self, batchnorm2d_1_num_features, conv2d_3_in_channels):
        super(Module0, self).__init__()
        self.concat_0 = P.Concat(axis=1)
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=batchnorm2d_1_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(in_channels=128,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)

    def construct(self, x):
        opt_concat_0 = self.concat_0((x,))
        opt_batchnorm2d_1 = self.batchnorm2d_1(opt_concat_0)
        opt_relu_2 = self.relu_2(opt_batchnorm2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        return opt_conv2d_5


class Module1(nn.Cell):
    def __init__(self, batchnorm2d_0_num_features):
        super(Module1, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        return opt_relu_1


class Module4(nn.Cell):
    def __init__(self, conv2d_0_in_channels, module1_0_batchnorm2d_0_num_features):
        super(Module4, self).__init__()
        self.module1_0 = Module1(batchnorm2d_0_num_features=module1_0_batchnorm2d_0_num_features)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=128,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        opt_conv2d_0 = self.conv2d_0(module1_0_opt)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        return opt_conv2d_2


class MainModel(nn.Cell):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.pad_maxpool2d_2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module0_0 = Module0(batchnorm2d_1_num_features=64, conv2d_3_in_channels=64)
        self.concat_9 = P.Concat(axis=1)
        self.module4_0 = Module4(conv2d_0_in_channels=96, module1_0_batchnorm2d_0_num_features=96)
        self.concat_15 = P.Concat(axis=1)
        self.module4_1 = Module4(conv2d_0_in_channels=128, module1_0_batchnorm2d_0_num_features=128)
        self.concat_21 = P.Concat(axis=1)
        self.module4_2 = Module4(conv2d_0_in_channels=160, module1_0_batchnorm2d_0_num_features=160)
        self.concat_27 = P.Concat(axis=1)
        self.module4_3 = Module4(conv2d_0_in_channels=192, module1_0_batchnorm2d_0_num_features=192)
        self.concat_33 = P.Concat(axis=1)
        self.module4_4 = Module4(conv2d_0_in_channels=224, module1_0_batchnorm2d_0_num_features=224)
        self.concat_39 = P.Concat(axis=1)
        self.module1_0 = Module1(batchnorm2d_0_num_features=256)
        self.conv2d_42 = nn.Conv2d(in_channels=256,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.pad_avgpool2d_43 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_43 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_1 = Module0(batchnorm2d_1_num_features=128, conv2d_3_in_channels=128)
        self.concat_50 = P.Concat(axis=1)
        self.module4_5 = Module4(conv2d_0_in_channels=160, module1_0_batchnorm2d_0_num_features=160)
        self.concat_56 = P.Concat(axis=1)
        self.module4_6 = Module4(conv2d_0_in_channels=192, module1_0_batchnorm2d_0_num_features=192)
        self.concat_62 = P.Concat(axis=1)
        self.module4_7 = Module4(conv2d_0_in_channels=224, module1_0_batchnorm2d_0_num_features=224)
        self.concat_68 = P.Concat(axis=1)
        self.module4_8 = Module4(conv2d_0_in_channels=256, module1_0_batchnorm2d_0_num_features=256)
        self.concat_74 = P.Concat(axis=1)
        self.module4_9 = Module4(conv2d_0_in_channels=288, module1_0_batchnorm2d_0_num_features=288)
        self.concat_80 = P.Concat(axis=1)
        self.module4_10 = Module4(conv2d_0_in_channels=320, module1_0_batchnorm2d_0_num_features=320)
        self.concat_86 = P.Concat(axis=1)
        self.module4_11 = Module4(conv2d_0_in_channels=352, module1_0_batchnorm2d_0_num_features=352)
        self.concat_92 = P.Concat(axis=1)
        self.module4_12 = Module4(conv2d_0_in_channels=384, module1_0_batchnorm2d_0_num_features=384)
        self.concat_98 = P.Concat(axis=1)
        self.module4_13 = Module4(conv2d_0_in_channels=416, module1_0_batchnorm2d_0_num_features=416)
        self.concat_104 = P.Concat(axis=1)
        self.module4_14 = Module4(conv2d_0_in_channels=448, module1_0_batchnorm2d_0_num_features=448)
        self.concat_110 = P.Concat(axis=1)
        self.module4_15 = Module4(conv2d_0_in_channels=480, module1_0_batchnorm2d_0_num_features=480)
        self.concat_116 = P.Concat(axis=1)
        self.module1_1 = Module1(batchnorm2d_0_num_features=512)
        self.conv2d_119 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.pad_avgpool2d_120 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_120 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_2 = Module0(batchnorm2d_1_num_features=256, conv2d_3_in_channels=256)
        self.concat_127 = P.Concat(axis=1)
        self.module4_16 = Module4(conv2d_0_in_channels=288, module1_0_batchnorm2d_0_num_features=288)
        self.concat_133 = P.Concat(axis=1)
        self.module4_17 = Module4(conv2d_0_in_channels=320, module1_0_batchnorm2d_0_num_features=320)
        self.concat_139 = P.Concat(axis=1)
        self.module4_18 = Module4(conv2d_0_in_channels=352, module1_0_batchnorm2d_0_num_features=352)
        self.concat_145 = P.Concat(axis=1)
        self.module4_19 = Module4(conv2d_0_in_channels=384, module1_0_batchnorm2d_0_num_features=384)
        self.concat_151 = P.Concat(axis=1)
        self.module4_20 = Module4(conv2d_0_in_channels=416, module1_0_batchnorm2d_0_num_features=416)
        self.concat_157 = P.Concat(axis=1)
        self.module4_21 = Module4(conv2d_0_in_channels=448, module1_0_batchnorm2d_0_num_features=448)
        self.concat_163 = P.Concat(axis=1)
        self.module4_22 = Module4(conv2d_0_in_channels=480, module1_0_batchnorm2d_0_num_features=480)
        self.concat_169 = P.Concat(axis=1)
        self.module4_23 = Module4(conv2d_0_in_channels=512, module1_0_batchnorm2d_0_num_features=512)
        self.concat_175 = P.Concat(axis=1)
        self.module4_24 = Module4(conv2d_0_in_channels=544, module1_0_batchnorm2d_0_num_features=544)
        self.concat_181 = P.Concat(axis=1)
        self.module4_25 = Module4(conv2d_0_in_channels=576, module1_0_batchnorm2d_0_num_features=576)
        self.concat_187 = P.Concat(axis=1)
        self.module4_26 = Module4(conv2d_0_in_channels=608, module1_0_batchnorm2d_0_num_features=608)
        self.concat_193 = P.Concat(axis=1)
        self.module4_27 = Module4(conv2d_0_in_channels=640, module1_0_batchnorm2d_0_num_features=640)
        self.concat_199 = P.Concat(axis=1)
        self.module4_28 = Module4(conv2d_0_in_channels=672, module1_0_batchnorm2d_0_num_features=672)
        self.concat_205 = P.Concat(axis=1)
        self.module4_29 = Module4(conv2d_0_in_channels=704, module1_0_batchnorm2d_0_num_features=704)
        self.concat_211 = P.Concat(axis=1)
        self.module4_30 = Module4(conv2d_0_in_channels=736, module1_0_batchnorm2d_0_num_features=736)
        self.concat_217 = P.Concat(axis=1)
        self.module4_31 = Module4(conv2d_0_in_channels=768, module1_0_batchnorm2d_0_num_features=768)
        self.concat_223 = P.Concat(axis=1)
        self.module4_32 = Module4(conv2d_0_in_channels=800, module1_0_batchnorm2d_0_num_features=800)
        self.concat_229 = P.Concat(axis=1)
        self.module4_33 = Module4(conv2d_0_in_channels=832, module1_0_batchnorm2d_0_num_features=832)
        self.concat_235 = P.Concat(axis=1)
        self.module4_34 = Module4(conv2d_0_in_channels=864, module1_0_batchnorm2d_0_num_features=864)
        self.concat_241 = P.Concat(axis=1)
        self.module4_35 = Module4(conv2d_0_in_channels=896, module1_0_batchnorm2d_0_num_features=896)
        self.concat_247 = P.Concat(axis=1)
        self.module4_36 = Module4(conv2d_0_in_channels=928, module1_0_batchnorm2d_0_num_features=928)
        self.concat_253 = P.Concat(axis=1)
        self.module4_37 = Module4(conv2d_0_in_channels=960, module1_0_batchnorm2d_0_num_features=960)
        self.concat_259 = P.Concat(axis=1)
        self.module4_38 = Module4(conv2d_0_in_channels=992, module1_0_batchnorm2d_0_num_features=992)
        self.concat_265 = P.Concat(axis=1)
        self.module4_39 = Module4(conv2d_0_in_channels=1024, module1_0_batchnorm2d_0_num_features=1024)
        self.concat_271 = P.Concat(axis=1)
        self.module4_40 = Module4(conv2d_0_in_channels=1056, module1_0_batchnorm2d_0_num_features=1056)
        self.concat_277 = P.Concat(axis=1)
        self.module4_41 = Module4(conv2d_0_in_channels=1088, module1_0_batchnorm2d_0_num_features=1088)
        self.concat_283 = P.Concat(axis=1)
        self.module4_42 = Module4(conv2d_0_in_channels=1120, module1_0_batchnorm2d_0_num_features=1120)
        self.concat_289 = P.Concat(axis=1)
        self.module4_43 = Module4(conv2d_0_in_channels=1152, module1_0_batchnorm2d_0_num_features=1152)
        self.concat_295 = P.Concat(axis=1)
        self.module4_44 = Module4(conv2d_0_in_channels=1184, module1_0_batchnorm2d_0_num_features=1184)
        self.concat_301 = P.Concat(axis=1)
        self.module4_45 = Module4(conv2d_0_in_channels=1216, module1_0_batchnorm2d_0_num_features=1216)
        self.concat_307 = P.Concat(axis=1)
        self.module4_46 = Module4(conv2d_0_in_channels=1248, module1_0_batchnorm2d_0_num_features=1248)
        self.concat_313 = P.Concat(axis=1)
        self.module4_47 = Module4(conv2d_0_in_channels=1280, module1_0_batchnorm2d_0_num_features=1280)
        self.concat_319 = P.Concat(axis=1)
        self.module4_48 = Module4(conv2d_0_in_channels=1312, module1_0_batchnorm2d_0_num_features=1312)
        self.concat_325 = P.Concat(axis=1)
        self.module4_49 = Module4(conv2d_0_in_channels=1344, module1_0_batchnorm2d_0_num_features=1344)
        self.concat_331 = P.Concat(axis=1)
        self.module4_50 = Module4(conv2d_0_in_channels=1376, module1_0_batchnorm2d_0_num_features=1376)
        self.concat_337 = P.Concat(axis=1)
        self.module4_51 = Module4(conv2d_0_in_channels=1408, module1_0_batchnorm2d_0_num_features=1408)
        self.concat_343 = P.Concat(axis=1)
        self.module4_52 = Module4(conv2d_0_in_channels=1440, module1_0_batchnorm2d_0_num_features=1440)
        self.concat_349 = P.Concat(axis=1)
        self.module4_53 = Module4(conv2d_0_in_channels=1472, module1_0_batchnorm2d_0_num_features=1472)
        self.concat_355 = P.Concat(axis=1)
        self.module4_54 = Module4(conv2d_0_in_channels=1504, module1_0_batchnorm2d_0_num_features=1504)
        self.concat_361 = P.Concat(axis=1)
        self.module4_55 = Module4(conv2d_0_in_channels=1536, module1_0_batchnorm2d_0_num_features=1536)
        self.concat_367 = P.Concat(axis=1)
        self.module4_56 = Module4(conv2d_0_in_channels=1568, module1_0_batchnorm2d_0_num_features=1568)
        self.concat_373 = P.Concat(axis=1)
        self.module4_57 = Module4(conv2d_0_in_channels=1600, module1_0_batchnorm2d_0_num_features=1600)
        self.concat_379 = P.Concat(axis=1)
        self.module4_58 = Module4(conv2d_0_in_channels=1632, module1_0_batchnorm2d_0_num_features=1632)
        self.concat_385 = P.Concat(axis=1)
        self.module4_59 = Module4(conv2d_0_in_channels=1664, module1_0_batchnorm2d_0_num_features=1664)
        self.concat_391 = P.Concat(axis=1)
        self.module4_60 = Module4(conv2d_0_in_channels=1696, module1_0_batchnorm2d_0_num_features=1696)
        self.concat_397 = P.Concat(axis=1)
        self.module4_61 = Module4(conv2d_0_in_channels=1728, module1_0_batchnorm2d_0_num_features=1728)
        self.concat_403 = P.Concat(axis=1)
        self.module4_62 = Module4(conv2d_0_in_channels=1760, module1_0_batchnorm2d_0_num_features=1760)
        self.concat_409 = P.Concat(axis=1)
        self.module1_2 = Module1(batchnorm2d_0_num_features=1792)
        self.conv2d_412 = nn.Conv2d(in_channels=1792,
                                    out_channels=896,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.pad_avgpool2d_413 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_413 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_3 = Module0(batchnorm2d_1_num_features=896, conv2d_3_in_channels=896)
        self.concat_420 = P.Concat(axis=1)
        self.module4_63 = Module4(conv2d_0_in_channels=928, module1_0_batchnorm2d_0_num_features=928)
        self.concat_426 = P.Concat(axis=1)
        self.module4_64 = Module4(conv2d_0_in_channels=960, module1_0_batchnorm2d_0_num_features=960)
        self.concat_432 = P.Concat(axis=1)
        self.module4_65 = Module4(conv2d_0_in_channels=992, module1_0_batchnorm2d_0_num_features=992)
        self.concat_438 = P.Concat(axis=1)
        self.module4_66 = Module4(conv2d_0_in_channels=1024, module1_0_batchnorm2d_0_num_features=1024)
        self.concat_444 = P.Concat(axis=1)
        self.module4_67 = Module4(conv2d_0_in_channels=1056, module1_0_batchnorm2d_0_num_features=1056)
        self.concat_450 = P.Concat(axis=1)
        self.module4_68 = Module4(conv2d_0_in_channels=1088, module1_0_batchnorm2d_0_num_features=1088)
        self.concat_456 = P.Concat(axis=1)
        self.module4_69 = Module4(conv2d_0_in_channels=1120, module1_0_batchnorm2d_0_num_features=1120)
        self.concat_462 = P.Concat(axis=1)
        self.module4_70 = Module4(conv2d_0_in_channels=1152, module1_0_batchnorm2d_0_num_features=1152)
        self.concat_468 = P.Concat(axis=1)
        self.module4_71 = Module4(conv2d_0_in_channels=1184, module1_0_batchnorm2d_0_num_features=1184)
        self.concat_474 = P.Concat(axis=1)
        self.module4_72 = Module4(conv2d_0_in_channels=1216, module1_0_batchnorm2d_0_num_features=1216)
        self.concat_480 = P.Concat(axis=1)
        self.module4_73 = Module4(conv2d_0_in_channels=1248, module1_0_batchnorm2d_0_num_features=1248)
        self.concat_486 = P.Concat(axis=1)
        self.module4_74 = Module4(conv2d_0_in_channels=1280, module1_0_batchnorm2d_0_num_features=1280)
        self.concat_492 = P.Concat(axis=1)
        self.module4_75 = Module4(conv2d_0_in_channels=1312, module1_0_batchnorm2d_0_num_features=1312)
        self.concat_498 = P.Concat(axis=1)
        self.module4_76 = Module4(conv2d_0_in_channels=1344, module1_0_batchnorm2d_0_num_features=1344)
        self.concat_504 = P.Concat(axis=1)
        self.module4_77 = Module4(conv2d_0_in_channels=1376, module1_0_batchnorm2d_0_num_features=1376)
        self.concat_510 = P.Concat(axis=1)
        self.module4_78 = Module4(conv2d_0_in_channels=1408, module1_0_batchnorm2d_0_num_features=1408)
        self.concat_516 = P.Concat(axis=1)
        self.module4_79 = Module4(conv2d_0_in_channels=1440, module1_0_batchnorm2d_0_num_features=1440)
        self.concat_522 = P.Concat(axis=1)
        self.module4_80 = Module4(conv2d_0_in_channels=1472, module1_0_batchnorm2d_0_num_features=1472)
        self.concat_528 = P.Concat(axis=1)
        self.module4_81 = Module4(conv2d_0_in_channels=1504, module1_0_batchnorm2d_0_num_features=1504)
        self.concat_534 = P.Concat(axis=1)
        self.module4_82 = Module4(conv2d_0_in_channels=1536, module1_0_batchnorm2d_0_num_features=1536)
        self.concat_540 = P.Concat(axis=1)
        self.module4_83 = Module4(conv2d_0_in_channels=1568, module1_0_batchnorm2d_0_num_features=1568)
        self.concat_546 = P.Concat(axis=1)
        self.module4_84 = Module4(conv2d_0_in_channels=1600, module1_0_batchnorm2d_0_num_features=1600)
        self.concat_552 = P.Concat(axis=1)
        self.module4_85 = Module4(conv2d_0_in_channels=1632, module1_0_batchnorm2d_0_num_features=1632)
        self.concat_558 = P.Concat(axis=1)
        self.module4_86 = Module4(conv2d_0_in_channels=1664, module1_0_batchnorm2d_0_num_features=1664)
        self.concat_564 = P.Concat(axis=1)
        self.module4_87 = Module4(conv2d_0_in_channels=1696, module1_0_batchnorm2d_0_num_features=1696)
        self.concat_570 = P.Concat(axis=1)
        self.module4_88 = Module4(conv2d_0_in_channels=1728, module1_0_batchnorm2d_0_num_features=1728)
        self.concat_576 = P.Concat(axis=1)
        self.module4_89 = Module4(conv2d_0_in_channels=1760, module1_0_batchnorm2d_0_num_features=1760)
        self.concat_582 = P.Concat(axis=1)
        self.module4_90 = Module4(conv2d_0_in_channels=1792, module1_0_batchnorm2d_0_num_features=1792)
        self.concat_588 = P.Concat(axis=1)
        self.module4_91 = Module4(conv2d_0_in_channels=1824, module1_0_batchnorm2d_0_num_features=1824)
        self.concat_594 = P.Concat(axis=1)
        self.module4_92 = Module4(conv2d_0_in_channels=1856, module1_0_batchnorm2d_0_num_features=1856)
        self.concat_600 = P.Concat(axis=1)
        self.module4_93 = Module4(conv2d_0_in_channels=1888, module1_0_batchnorm2d_0_num_features=1888)
        self.concat_606 = P.Concat(axis=1)
        self.module1_3 = Module1(batchnorm2d_0_num_features=1920)
        self.avgpool2d_609 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_610 = nn.Flatten()
        self.dense_611 = nn.Dense(in_channels=1920, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module0_0_opt = self.module0_0(opt_maxpool2d_2)
        opt_concat_9 = self.concat_9((opt_maxpool2d_2, module0_0_opt,))
        module4_0_opt = self.module4_0(opt_concat_9)
        opt_concat_15 = self.concat_15((opt_maxpool2d_2, module0_0_opt, module4_0_opt,))
        module4_1_opt = self.module4_1(opt_concat_15)
        opt_concat_21 = self.concat_21((opt_maxpool2d_2, module0_0_opt, module4_0_opt, module4_1_opt,))
        module4_2_opt = self.module4_2(opt_concat_21)
        opt_concat_27 = self.concat_27((opt_maxpool2d_2, module0_0_opt, module4_0_opt, module4_1_opt, module4_2_opt,))
        module4_3_opt = self.module4_3(opt_concat_27)
        opt_concat_33 = self.concat_33(
            (opt_maxpool2d_2, module0_0_opt, module4_0_opt, module4_1_opt, module4_2_opt, module4_3_opt,))
        module4_4_opt = self.module4_4(opt_concat_33)
        opt_concat_39 = self.concat_39(
            (opt_maxpool2d_2, module0_0_opt, module4_0_opt, module4_1_opt, module4_2_opt, module4_3_opt, module4_4_opt,
             ))
        module1_0_opt = self.module1_0(opt_concat_39)
        opt_conv2d_42 = self.conv2d_42(module1_0_opt)
        opt_avgpool2d_43 = self.pad_avgpool2d_43(opt_conv2d_42)
        opt_avgpool2d_43 = self.avgpool2d_43(opt_avgpool2d_43)
        module0_1_opt = self.module0_1(opt_avgpool2d_43)
        opt_concat_50 = self.concat_50((opt_avgpool2d_43, module0_1_opt,))
        module4_5_opt = self.module4_5(opt_concat_50)
        opt_concat_56 = self.concat_56((opt_avgpool2d_43, module0_1_opt, module4_5_opt,))
        module4_6_opt = self.module4_6(opt_concat_56)
        opt_concat_62 = self.concat_62((opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt,))
        module4_7_opt = self.module4_7(opt_concat_62)
        opt_concat_68 = self.concat_68((opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt,))
        module4_8_opt = self.module4_8(opt_concat_68)
        opt_concat_74 = self.concat_74(
            (opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt, module4_8_opt,
             ))
        module4_9_opt = self.module4_9(opt_concat_74)
        opt_concat_80 = self.concat_80(
            (opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt, module4_8_opt, module4_9_opt,
             ))
        module4_10_opt = self.module4_10(opt_concat_80)
        opt_concat_86 = self.concat_86((opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt,
                                        module4_8_opt, module4_9_opt, module4_10_opt,
                                        ))
        module4_11_opt = self.module4_11(opt_concat_86)
        opt_concat_92 = self.concat_92((opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt,
                                        module4_8_opt, module4_9_opt, module4_10_opt, module4_11_opt,
                                        ))
        module4_12_opt = self.module4_12(opt_concat_92)
        opt_concat_98 = self.concat_98((opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt,
                                        module4_8_opt, module4_9_opt, module4_10_opt, module4_11_opt, module4_12_opt,
                                        ))
        module4_13_opt = self.module4_13(opt_concat_98)
        opt_concat_104 = self.concat_104(
            (opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt, module4_8_opt, module4_9_opt,
             module4_10_opt, module4_11_opt, module4_12_opt, module4_13_opt,
             ))
        module4_14_opt = self.module4_14(opt_concat_104)
        opt_concat_110 = self.concat_110(
            (opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt, module4_8_opt, module4_9_opt,
             module4_10_opt, module4_11_opt, module4_12_opt, module4_13_opt, module4_14_opt,
             ))
        module4_15_opt = self.module4_15(opt_concat_110)
        opt_concat_116 = self.concat_116(
            (opt_avgpool2d_43, module0_1_opt, module4_5_opt, module4_6_opt, module4_7_opt, module4_8_opt, module4_9_opt,
             module4_10_opt, module4_11_opt, module4_12_opt, module4_13_opt, module4_14_opt, module4_15_opt,
             ))
        module1_1_opt = self.module1_1(opt_concat_116)
        opt_conv2d_119 = self.conv2d_119(module1_1_opt)
        opt_avgpool2d_120 = self.pad_avgpool2d_120(opt_conv2d_119)
        opt_avgpool2d_120 = self.avgpool2d_120(opt_avgpool2d_120)
        module0_2_opt = self.module0_2(opt_avgpool2d_120)
        opt_concat_127 = self.concat_127((opt_avgpool2d_120, module0_2_opt,))
        module4_16_opt = self.module4_16(opt_concat_127)
        opt_concat_133 = self.concat_133((opt_avgpool2d_120, module0_2_opt, module4_16_opt,))
        module4_17_opt = self.module4_17(opt_concat_133)
        opt_concat_139 = self.concat_139((opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt,))
        module4_18_opt = self.module4_18(opt_concat_139)
        opt_concat_145 = self.concat_145(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt,
             ))
        module4_19_opt = self.module4_19(opt_concat_145)
        opt_concat_151 = self.concat_151(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             ))
        module4_20_opt = self.module4_20(opt_concat_151)
        opt_concat_157 = self.concat_157((opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt,
                                          module4_18_opt, module4_19_opt, module4_20_opt,
                                          ))
        module4_21_opt = self.module4_21(opt_concat_157)
        opt_concat_163 = self.concat_163((opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt,
                                          module4_18_opt, module4_19_opt, module4_20_opt, module4_21_opt,
                                          ))
        module4_22_opt = self.module4_22(opt_concat_163)
        opt_concat_169 = self.concat_169(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt,
             ))
        module4_23_opt = self.module4_23(opt_concat_169)
        opt_concat_175 = self.concat_175(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt,
             ))
        module4_24_opt = self.module4_24(opt_concat_175)
        opt_concat_181 = self.concat_181(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt,
             ))
        module4_25_opt = self.module4_25(opt_concat_181)
        opt_concat_187 = self.concat_187(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             ))
        module4_26_opt = self.module4_26(opt_concat_187)
        opt_concat_193 = self.concat_193(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt,
             ))
        module4_27_opt = self.module4_27(opt_concat_193)
        opt_concat_199 = self.concat_199(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt,
             ))
        module4_28_opt = self.module4_28(opt_concat_199)
        opt_concat_205 = self.concat_205(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt,
             ))
        module4_29_opt = self.module4_29(opt_concat_205)
        opt_concat_211 = self.concat_211(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt,
             ))
        module4_30_opt = self.module4_30(opt_concat_211)
        opt_concat_217 = self.concat_217(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt,
             ))
        module4_31_opt = self.module4_31(opt_concat_217)
        opt_concat_223 = self.concat_223(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             ))
        module4_32_opt = self.module4_32(opt_concat_223)
        opt_concat_229 = self.concat_229(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt,
             ))
        module4_33_opt = self.module4_33(opt_concat_229)
        opt_concat_235 = self.concat_235(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt,
             ))
        module4_34_opt = self.module4_34(opt_concat_235)
        opt_concat_241 = self.concat_241(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt,
             ))
        module4_35_opt = self.module4_35(opt_concat_241)
        opt_concat_247 = self.concat_247(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt,
             ))
        module4_36_opt = self.module4_36(opt_concat_247)
        opt_concat_253 = self.concat_253(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt,
             ))
        module4_37_opt = self.module4_37(opt_concat_253)
        opt_concat_259 = self.concat_259(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             ))
        module4_38_opt = self.module4_38(opt_concat_259)
        opt_concat_265 = self.concat_265(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt,
             ))
        module4_39_opt = self.module4_39(opt_concat_265)
        opt_concat_271 = self.concat_271(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt,
             ))
        module4_40_opt = self.module4_40(opt_concat_271)
        opt_concat_277 = self.concat_277(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt,
             ))
        module4_41_opt = self.module4_41(opt_concat_277)
        opt_concat_283 = self.concat_283(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt,
             ))
        module4_42_opt = self.module4_42(opt_concat_283)
        opt_concat_289 = self.concat_289(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             ))
        module4_43_opt = self.module4_43(opt_concat_289)
        opt_concat_295 = self.concat_295(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             ))
        module4_44_opt = self.module4_44(opt_concat_295)
        opt_concat_301 = self.concat_301(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt,
             ))
        module4_45_opt = self.module4_45(opt_concat_301)
        opt_concat_307 = self.concat_307(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt,
             ))
        module4_46_opt = self.module4_46(opt_concat_307)
        opt_concat_313 = self.concat_313(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt,
             ))
        module4_47_opt = self.module4_47(opt_concat_313)
        opt_concat_319 = self.concat_319(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt,
             ))
        module4_48_opt = self.module4_48(opt_concat_319)
        opt_concat_325 = self.concat_325(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             ))
        module4_49_opt = self.module4_49(opt_concat_325)
        opt_concat_331 = self.concat_331(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             ))
        module4_50_opt = self.module4_50(opt_concat_331)
        opt_concat_337 = self.concat_337(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt,
             ))
        module4_51_opt = self.module4_51(opt_concat_337)
        opt_concat_343 = self.concat_343(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt,
             ))
        module4_52_opt = self.module4_52(opt_concat_343)
        opt_concat_349 = self.concat_349(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt,
             ))
        module4_53_opt = self.module4_53(opt_concat_349)
        opt_concat_355 = self.concat_355(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt,
             ))
        module4_54_opt = self.module4_54(opt_concat_355)
        opt_concat_361 = self.concat_361(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt,
             ))
        module4_55_opt = self.module4_55(opt_concat_361)
        opt_concat_367 = self.concat_367(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             ))
        module4_56_opt = self.module4_56(opt_concat_367)
        opt_concat_373 = self.concat_373(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt,
             ))
        module4_57_opt = self.module4_57(opt_concat_373)
        opt_concat_379 = self.concat_379(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt, module4_57_opt,
             ))
        module4_58_opt = self.module4_58(opt_concat_379)
        opt_concat_385 = self.concat_385(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt, module4_57_opt, module4_58_opt,
             ))
        module4_59_opt = self.module4_59(opt_concat_385)
        opt_concat_391 = self.concat_391(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt, module4_57_opt, module4_58_opt, module4_59_opt,
             ))
        module4_60_opt = self.module4_60(opt_concat_391)
        opt_concat_397 = self.concat_397(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt, module4_57_opt, module4_58_opt, module4_59_opt, module4_60_opt,
             ))
        module4_61_opt = self.module4_61(opt_concat_397)
        opt_concat_403 = self.concat_403(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt, module4_57_opt, module4_58_opt, module4_59_opt, module4_60_opt, module4_61_opt,
             ))
        module4_62_opt = self.module4_62(opt_concat_403)
        opt_concat_409 = self.concat_409(
            (opt_avgpool2d_120, module0_2_opt, module4_16_opt, module4_17_opt, module4_18_opt, module4_19_opt,
             module4_20_opt, module4_21_opt, module4_22_opt, module4_23_opt, module4_24_opt, module4_25_opt,
             module4_26_opt, module4_27_opt, module4_28_opt, module4_29_opt, module4_30_opt, module4_31_opt,
             module4_32_opt, module4_33_opt, module4_34_opt, module4_35_opt, module4_36_opt, module4_37_opt,
             module4_38_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt, module4_43_opt,
             module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt, module4_49_opt,
             module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt, module4_54_opt, module4_55_opt,
             module4_56_opt, module4_57_opt, module4_58_opt, module4_59_opt, module4_60_opt, module4_61_opt,
             module4_62_opt,
             ))
        module1_2_opt = self.module1_2(opt_concat_409)
        opt_conv2d_412 = self.conv2d_412(module1_2_opt)
        opt_avgpool2d_413 = self.pad_avgpool2d_413(opt_conv2d_412)
        opt_avgpool2d_413 = self.avgpool2d_413(opt_avgpool2d_413)
        module0_3_opt = self.module0_3(opt_avgpool2d_413)
        opt_concat_420 = self.concat_420((opt_avgpool2d_413, module0_3_opt,))
        module4_63_opt = self.module4_63(opt_concat_420)
        opt_concat_426 = self.concat_426((opt_avgpool2d_413, module0_3_opt, module4_63_opt,))
        module4_64_opt = self.module4_64(opt_concat_426)
        opt_concat_432 = self.concat_432((opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt,))
        module4_65_opt = self.module4_65(opt_concat_432)
        opt_concat_438 = self.concat_438(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt,
             ))
        module4_66_opt = self.module4_66(opt_concat_438)
        opt_concat_444 = self.concat_444(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             ))
        module4_67_opt = self.module4_67(opt_concat_444)
        opt_concat_450 = self.concat_450((opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt,
                                          module4_65_opt, module4_66_opt, module4_67_opt,
                                          ))
        module4_68_opt = self.module4_68(opt_concat_450)
        opt_concat_456 = self.concat_456((opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt,
                                          module4_65_opt, module4_66_opt, module4_67_opt, module4_68_opt,
                                          ))
        module4_69_opt = self.module4_69(opt_concat_456)
        opt_concat_462 = self.concat_462(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt,
             ))
        module4_70_opt = self.module4_70(opt_concat_462)
        opt_concat_468 = self.concat_468(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt,
             ))
        module4_71_opt = self.module4_71(opt_concat_468)
        opt_concat_474 = self.concat_474(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt,
             ))
        module4_72_opt = self.module4_72(opt_concat_474)
        opt_concat_480 = self.concat_480(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             ))
        module4_73_opt = self.module4_73(opt_concat_480)
        opt_concat_486 = self.concat_486(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt,
             ))
        module4_74_opt = self.module4_74(opt_concat_486)
        opt_concat_492 = self.concat_492(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt,
             ))
        module4_75_opt = self.module4_75(opt_concat_492)
        opt_concat_498 = self.concat_498(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt,
             ))
        module4_76_opt = self.module4_76(opt_concat_498)
        opt_concat_504 = self.concat_504(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt,
             ))
        module4_77_opt = self.module4_77(opt_concat_504)
        opt_concat_510 = self.concat_510(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt,
             ))
        module4_78_opt = self.module4_78(opt_concat_510)
        opt_concat_516 = self.concat_516(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             ))
        module4_79_opt = self.module4_79(opt_concat_516)
        opt_concat_522 = self.concat_522(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt,
             ))
        module4_80_opt = self.module4_80(opt_concat_522)
        opt_concat_528 = self.concat_528(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt,
             ))
        module4_81_opt = self.module4_81(opt_concat_528)
        opt_concat_534 = self.concat_534(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt,
             ))
        module4_82_opt = self.module4_82(opt_concat_534)
        opt_concat_540 = self.concat_540(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt,
             ))
        module4_83_opt = self.module4_83(opt_concat_540)
        opt_concat_546 = self.concat_546(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt,
             ))
        module4_84_opt = self.module4_84(opt_concat_546)
        opt_concat_552 = self.concat_552(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             ))
        module4_85_opt = self.module4_85(opt_concat_552)
        opt_concat_558 = self.concat_558(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt,
             ))
        module4_86_opt = self.module4_86(opt_concat_558)
        opt_concat_564 = self.concat_564(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt,
             ))
        module4_87_opt = self.module4_87(opt_concat_564)
        opt_concat_570 = self.concat_570(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt,
             ))
        module4_88_opt = self.module4_88(opt_concat_570)
        opt_concat_576 = self.concat_576(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt, module4_88_opt,
             ))
        module4_89_opt = self.module4_89(opt_concat_576)
        opt_concat_582 = self.concat_582(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt, module4_88_opt, module4_89_opt,
             ))
        module4_90_opt = self.module4_90(opt_concat_582)
        opt_concat_588 = self.concat_588(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt, module4_88_opt, module4_89_opt, module4_90_opt,
             ))
        module4_91_opt = self.module4_91(opt_concat_588)
        opt_concat_594 = self.concat_594(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt, module4_88_opt, module4_89_opt, module4_90_opt,
             module4_91_opt,
             ))
        module4_92_opt = self.module4_92(opt_concat_594)
        opt_concat_600 = self.concat_600(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt, module4_88_opt, module4_89_opt, module4_90_opt,
             module4_91_opt, module4_92_opt,
             ))
        module4_93_opt = self.module4_93(opt_concat_600)
        opt_concat_606 = self.concat_606(
            (opt_avgpool2d_413, module0_3_opt, module4_63_opt, module4_64_opt, module4_65_opt, module4_66_opt,
             module4_67_opt, module4_68_opt, module4_69_opt, module4_70_opt, module4_71_opt, module4_72_opt,
             module4_73_opt, module4_74_opt, module4_75_opt, module4_76_opt, module4_77_opt, module4_78_opt,
             module4_79_opt, module4_80_opt, module4_81_opt, module4_82_opt, module4_83_opt, module4_84_opt,
             module4_85_opt, module4_86_opt, module4_87_opt, module4_88_opt, module4_89_opt, module4_90_opt,
             module4_91_opt, module4_92_opt, module4_93_opt,
             ))
        module1_3_opt = self.module1_3(opt_concat_606)
        opt_avgpool2d_609 = self.avgpool2d_609(module1_3_opt)
        opt_flatten_610 = self.flatten_610(opt_avgpool2d_609)
        opt_dense_611 = self.dense_611(opt_flatten_610)
        return opt_dense_611
