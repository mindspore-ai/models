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
        self.module1_2 = Module1(batchnorm2d_0_num_features=1024)
        self.conv2d_268 = nn.Conv2d(in_channels=1024,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.pad_avgpool2d_269 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_269 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_3 = Module0(batchnorm2d_1_num_features=512, conv2d_3_in_channels=512)
        self.concat_276 = P.Concat(axis=1)
        self.module4_39 = Module4(conv2d_0_in_channels=544, module1_0_batchnorm2d_0_num_features=544)
        self.concat_282 = P.Concat(axis=1)
        self.module4_40 = Module4(conv2d_0_in_channels=576, module1_0_batchnorm2d_0_num_features=576)
        self.concat_288 = P.Concat(axis=1)
        self.module4_41 = Module4(conv2d_0_in_channels=608, module1_0_batchnorm2d_0_num_features=608)
        self.concat_294 = P.Concat(axis=1)
        self.module4_42 = Module4(conv2d_0_in_channels=640, module1_0_batchnorm2d_0_num_features=640)
        self.concat_300 = P.Concat(axis=1)
        self.module4_43 = Module4(conv2d_0_in_channels=672, module1_0_batchnorm2d_0_num_features=672)
        self.concat_306 = P.Concat(axis=1)
        self.module4_44 = Module4(conv2d_0_in_channels=704, module1_0_batchnorm2d_0_num_features=704)
        self.concat_312 = P.Concat(axis=1)
        self.module4_45 = Module4(conv2d_0_in_channels=736, module1_0_batchnorm2d_0_num_features=736)
        self.concat_318 = P.Concat(axis=1)
        self.module4_46 = Module4(conv2d_0_in_channels=768, module1_0_batchnorm2d_0_num_features=768)
        self.concat_324 = P.Concat(axis=1)
        self.module4_47 = Module4(conv2d_0_in_channels=800, module1_0_batchnorm2d_0_num_features=800)
        self.concat_330 = P.Concat(axis=1)
        self.module4_48 = Module4(conv2d_0_in_channels=832, module1_0_batchnorm2d_0_num_features=832)
        self.concat_336 = P.Concat(axis=1)
        self.module4_49 = Module4(conv2d_0_in_channels=864, module1_0_batchnorm2d_0_num_features=864)
        self.concat_342 = P.Concat(axis=1)
        self.module4_50 = Module4(conv2d_0_in_channels=896, module1_0_batchnorm2d_0_num_features=896)
        self.concat_348 = P.Concat(axis=1)
        self.module4_51 = Module4(conv2d_0_in_channels=928, module1_0_batchnorm2d_0_num_features=928)
        self.concat_354 = P.Concat(axis=1)
        self.module4_52 = Module4(conv2d_0_in_channels=960, module1_0_batchnorm2d_0_num_features=960)
        self.concat_360 = P.Concat(axis=1)
        self.module4_53 = Module4(conv2d_0_in_channels=992, module1_0_batchnorm2d_0_num_features=992)
        self.concat_366 = P.Concat(axis=1)
        self.module1_3 = Module1(batchnorm2d_0_num_features=1024)
        self.avgpool2d_369 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_370 = nn.Flatten()
        self.dense_371 = nn.Dense(in_channels=1024, out_channels=1000, has_bias=True)

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
            (opt_maxpool2d_2, module0_0_opt, module4_0_opt, module4_1_opt, module4_2_opt, module4_3_opt,
             ))
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
        module1_2_opt = self.module1_2(opt_concat_265)
        opt_conv2d_268 = self.conv2d_268(module1_2_opt)
        opt_avgpool2d_269 = self.pad_avgpool2d_269(opt_conv2d_268)
        opt_avgpool2d_269 = self.avgpool2d_269(opt_avgpool2d_269)
        module0_3_opt = self.module0_3(opt_avgpool2d_269)
        opt_concat_276 = self.concat_276((opt_avgpool2d_269, module0_3_opt,))
        module4_39_opt = self.module4_39(opt_concat_276)
        opt_concat_282 = self.concat_282((opt_avgpool2d_269, module0_3_opt, module4_39_opt,))
        module4_40_opt = self.module4_40(opt_concat_282)
        opt_concat_288 = self.concat_288((opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt,))
        module4_41_opt = self.module4_41(opt_concat_288)
        opt_concat_294 = self.concat_294(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt,
             ))
        module4_42_opt = self.module4_42(opt_concat_294)
        opt_concat_300 = self.concat_300(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             ))
        module4_43_opt = self.module4_43(opt_concat_300)
        opt_concat_306 = self.concat_306((opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt,
                                          module4_41_opt, module4_42_opt, module4_43_opt,
                                          ))
        module4_44_opt = self.module4_44(opt_concat_306)
        opt_concat_312 = self.concat_312((opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt,
                                          module4_41_opt, module4_42_opt, module4_43_opt, module4_44_opt,
                                          ))
        module4_45_opt = self.module4_45(opt_concat_312)
        opt_concat_318 = self.concat_318(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt,
             ))
        module4_46_opt = self.module4_46(opt_concat_318)
        opt_concat_324 = self.concat_324(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt,
             ))
        module4_47_opt = self.module4_47(opt_concat_324)
        opt_concat_330 = self.concat_330(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt,
             ))
        module4_48_opt = self.module4_48(opt_concat_330)
        opt_concat_336 = self.concat_336(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             ))
        module4_49_opt = self.module4_49(opt_concat_336)
        opt_concat_342 = self.concat_342(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             module4_49_opt,
             ))
        module4_50_opt = self.module4_50(opt_concat_342)
        opt_concat_348 = self.concat_348(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             module4_49_opt, module4_50_opt,
             ))
        module4_51_opt = self.module4_51(opt_concat_348)
        opt_concat_354 = self.concat_354(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             module4_49_opt, module4_50_opt, module4_51_opt,
             ))
        module4_52_opt = self.module4_52(opt_concat_354)
        opt_concat_360 = self.concat_360(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             module4_49_opt, module4_50_opt, module4_51_opt, module4_52_opt,
             ))
        module4_53_opt = self.module4_53(opt_concat_360)
        opt_concat_366 = self.concat_366(
            (opt_avgpool2d_269, module0_3_opt, module4_39_opt, module4_40_opt, module4_41_opt, module4_42_opt,
             module4_43_opt, module4_44_opt, module4_45_opt, module4_46_opt, module4_47_opt, module4_48_opt,
             module4_49_opt, module4_50_opt, module4_51_opt, module4_52_opt, module4_53_opt,
             ))
        module1_3_opt = self.module1_3(opt_concat_366)
        opt_avgpool2d_369 = self.avgpool2d_369(module1_3_opt)
        opt_flatten_370 = self.flatten_370(opt_avgpool2d_369)
        opt_dense_371 = self.dense_371(opt_flatten_370)
        return opt_dense_371
