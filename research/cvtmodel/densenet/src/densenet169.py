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


class Module21(nn.Cell):
    def __init__(self, batchnorm2d_0_num_features):
        super(Module21, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        return opt_relu_1


class Module24(nn.Cell):
    def __init__(self, conv2d_0_in_channels, module21_0_batchnorm2d_0_num_features):
        super(Module24, self).__init__()
        self.module21_0 = Module21(batchnorm2d_0_num_features=module21_0_batchnorm2d_0_num_features)
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
        module21_0_opt = self.module21_0(x)
        opt_conv2d_0 = self.conv2d_0(module21_0_opt)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        return opt_conv2d_2


class Module26(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module21_0_batchnorm2d_0_num_features):
        super(Module26, self).__init__()
        self.module21_0 = Module21(batchnorm2d_0_num_features=module21_0_batchnorm2d_0_num_features)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)

    def construct(self, x):
        module21_0_opt = self.module21_0(x)
        opt_conv2d_0 = self.conv2d_0(module21_0_opt)
        return opt_conv2d_0


class Module20(nn.Cell):
    def __init__(
            self, batchnorm2d_2_num_features, conv2d_4_in_channels, batchnorm2d_8_num_features, conv2d_10_in_channels,
            batchnorm2d_14_num_features, conv2d_16_in_channels, batchnorm2d_20_num_features, conv2d_22_in_channels,
            batchnorm2d_26_num_features, conv2d_28_in_channels, batchnorm2d_32_num_features, conv2d_34_in_channels,
            batchnorm2d_38_num_features, conv2d_40_in_channels, batchnorm2d_44_num_features, conv2d_46_in_channels,
            batchnorm2d_50_num_features, conv2d_52_in_channels, batchnorm2d_56_num_features, conv2d_58_in_channels,
            batchnorm2d_62_num_features, conv2d_64_in_channels, batchnorm2d_68_num_features, conv2d_70_in_channels,
            batchnorm2d_74_num_features, conv2d_76_in_channels, batchnorm2d_80_num_features, conv2d_82_in_channels,
            batchnorm2d_86_num_features, conv2d_88_in_channels, batchnorm2d_92_num_features, conv2d_94_in_channels,
            batchnorm2d_98_num_features, conv2d_100_in_channels, batchnorm2d_104_num_features, conv2d_106_in_channels,
            batchnorm2d_110_num_features, conv2d_112_in_channels, batchnorm2d_116_num_features, conv2d_118_in_channels,
            batchnorm2d_122_num_features, conv2d_124_in_channels, batchnorm2d_128_num_features, conv2d_130_in_channels,
            batchnorm2d_134_num_features, conv2d_136_in_channels, batchnorm2d_140_num_features, conv2d_142_in_channels,
            batchnorm2d_146_num_features, conv2d_148_in_channels, batchnorm2d_152_num_features, conv2d_154_in_channels,
            batchnorm2d_158_num_features, conv2d_160_in_channels, batchnorm2d_164_num_features, conv2d_166_in_channels,
            batchnorm2d_170_num_features, conv2d_172_in_channels, batchnorm2d_176_num_features, conv2d_178_in_channels,
            batchnorm2d_182_num_features, conv2d_184_in_channels, module0_0_batchnorm2d_1_num_features,
            module0_0_conv2d_3_in_channels):
        super(Module20, self).__init__()
        self.pad_avgpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_0 = Module0(batchnorm2d_1_num_features=module0_0_batchnorm2d_1_num_features,
                                 conv2d_3_in_channels=module0_0_conv2d_3_in_channels)
        self.concat_1 = P.Concat(axis=1)
        self.batchnorm2d_2 = nn.BatchNorm2d(num_features=batchnorm2d_2_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()
        self.conv2d_6 = nn.Conv2d(in_channels=128,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.concat_7 = P.Concat(axis=1)
        self.batchnorm2d_8 = nn.BatchNorm2d(num_features=batchnorm2d_8_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_9 = nn.ReLU()
        self.conv2d_10 = nn.Conv2d(in_channels=conv2d_10_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_11 = nn.ReLU()
        self.conv2d_12 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_13 = P.Concat(axis=1)
        self.batchnorm2d_14 = nn.BatchNorm2d(num_features=batchnorm2d_14_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_15 = nn.ReLU()
        self.conv2d_16 = nn.Conv2d(in_channels=conv2d_16_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_17 = nn.ReLU()
        self.conv2d_18 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_19 = P.Concat(axis=1)
        self.batchnorm2d_20 = nn.BatchNorm2d(num_features=batchnorm2d_20_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_21 = nn.ReLU()
        self.conv2d_22 = nn.Conv2d(in_channels=conv2d_22_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_23 = nn.ReLU()
        self.conv2d_24 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_25 = P.Concat(axis=1)
        self.batchnorm2d_26 = nn.BatchNorm2d(num_features=batchnorm2d_26_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_27 = nn.ReLU()
        self.conv2d_28 = nn.Conv2d(in_channels=conv2d_28_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_29 = nn.ReLU()
        self.conv2d_30 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_31 = P.Concat(axis=1)
        self.batchnorm2d_32 = nn.BatchNorm2d(num_features=batchnorm2d_32_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_33 = nn.ReLU()
        self.conv2d_34 = nn.Conv2d(in_channels=conv2d_34_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_35 = nn.ReLU()
        self.conv2d_36 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_37 = P.Concat(axis=1)
        self.batchnorm2d_38 = nn.BatchNorm2d(num_features=batchnorm2d_38_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_39 = nn.ReLU()
        self.conv2d_40 = nn.Conv2d(in_channels=conv2d_40_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_41 = nn.ReLU()
        self.conv2d_42 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_43 = P.Concat(axis=1)
        self.batchnorm2d_44 = nn.BatchNorm2d(num_features=batchnorm2d_44_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_45 = nn.ReLU()
        self.conv2d_46 = nn.Conv2d(in_channels=conv2d_46_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_47 = nn.ReLU()
        self.conv2d_48 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_49 = P.Concat(axis=1)
        self.batchnorm2d_50 = nn.BatchNorm2d(num_features=batchnorm2d_50_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_51 = nn.ReLU()
        self.conv2d_52 = nn.Conv2d(in_channels=conv2d_52_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_53 = nn.ReLU()
        self.conv2d_54 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_55 = P.Concat(axis=1)
        self.batchnorm2d_56 = nn.BatchNorm2d(num_features=batchnorm2d_56_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_57 = nn.ReLU()
        self.conv2d_58 = nn.Conv2d(in_channels=conv2d_58_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_59 = nn.ReLU()
        self.conv2d_60 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_61 = P.Concat(axis=1)
        self.batchnorm2d_62 = nn.BatchNorm2d(num_features=batchnorm2d_62_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_63 = nn.ReLU()
        self.conv2d_64 = nn.Conv2d(in_channels=conv2d_64_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_65 = nn.ReLU()
        self.conv2d_66 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_67 = P.Concat(axis=1)
        self.batchnorm2d_68 = nn.BatchNorm2d(num_features=batchnorm2d_68_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_69 = nn.ReLU()
        self.conv2d_70 = nn.Conv2d(in_channels=conv2d_70_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_71 = nn.ReLU()
        self.conv2d_72 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_73 = P.Concat(axis=1)
        self.batchnorm2d_74 = nn.BatchNorm2d(num_features=batchnorm2d_74_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_75 = nn.ReLU()
        self.conv2d_76 = nn.Conv2d(in_channels=conv2d_76_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_77 = nn.ReLU()
        self.conv2d_78 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_79 = P.Concat(axis=1)
        self.batchnorm2d_80 = nn.BatchNorm2d(num_features=batchnorm2d_80_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_81 = nn.ReLU()
        self.conv2d_82 = nn.Conv2d(in_channels=conv2d_82_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_83 = nn.ReLU()
        self.conv2d_84 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_85 = P.Concat(axis=1)
        self.batchnorm2d_86 = nn.BatchNorm2d(num_features=batchnorm2d_86_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_87 = nn.ReLU()
        self.conv2d_88 = nn.Conv2d(in_channels=conv2d_88_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_89 = nn.ReLU()
        self.conv2d_90 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_91 = P.Concat(axis=1)
        self.batchnorm2d_92 = nn.BatchNorm2d(num_features=batchnorm2d_92_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_93 = nn.ReLU()
        self.conv2d_94 = nn.Conv2d(in_channels=conv2d_94_in_channels,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_95 = nn.ReLU()
        self.conv2d_96 = nn.Conv2d(in_channels=128,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.concat_97 = P.Concat(axis=1)
        self.batchnorm2d_98 = nn.BatchNorm2d(num_features=batchnorm2d_98_num_features,
                                             eps=9.999999747378752e-06,
                                             momentum=0.8999999761581421)
        self.relu_99 = nn.ReLU()
        self.conv2d_100 = nn.Conv2d(in_channels=conv2d_100_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_101 = nn.ReLU()
        self.conv2d_102 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_103 = P.Concat(axis=1)
        self.batchnorm2d_104 = nn.BatchNorm2d(num_features=batchnorm2d_104_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_105 = nn.ReLU()
        self.conv2d_106 = nn.Conv2d(in_channels=conv2d_106_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_107 = nn.ReLU()
        self.conv2d_108 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_109 = P.Concat(axis=1)
        self.batchnorm2d_110 = nn.BatchNorm2d(num_features=batchnorm2d_110_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_111 = nn.ReLU()
        self.conv2d_112 = nn.Conv2d(in_channels=conv2d_112_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_113 = nn.ReLU()
        self.conv2d_114 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_115 = P.Concat(axis=1)
        self.batchnorm2d_116 = nn.BatchNorm2d(num_features=batchnorm2d_116_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_117 = nn.ReLU()
        self.conv2d_118 = nn.Conv2d(in_channels=conv2d_118_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_119 = nn.ReLU()
        self.conv2d_120 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_121 = P.Concat(axis=1)
        self.batchnorm2d_122 = nn.BatchNorm2d(num_features=batchnorm2d_122_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_123 = nn.ReLU()
        self.conv2d_124 = nn.Conv2d(in_channels=conv2d_124_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_125 = nn.ReLU()
        self.conv2d_126 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_127 = P.Concat(axis=1)
        self.batchnorm2d_128 = nn.BatchNorm2d(num_features=batchnorm2d_128_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_129 = nn.ReLU()
        self.conv2d_130 = nn.Conv2d(in_channels=conv2d_130_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_131 = nn.ReLU()
        self.conv2d_132 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_133 = P.Concat(axis=1)
        self.batchnorm2d_134 = nn.BatchNorm2d(num_features=batchnorm2d_134_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_135 = nn.ReLU()
        self.conv2d_136 = nn.Conv2d(in_channels=conv2d_136_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_137 = nn.ReLU()
        self.conv2d_138 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_139 = P.Concat(axis=1)
        self.batchnorm2d_140 = nn.BatchNorm2d(num_features=batchnorm2d_140_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_141 = nn.ReLU()
        self.conv2d_142 = nn.Conv2d(in_channels=conv2d_142_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_143 = nn.ReLU()
        self.conv2d_144 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_145 = P.Concat(axis=1)
        self.batchnorm2d_146 = nn.BatchNorm2d(num_features=batchnorm2d_146_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_147 = nn.ReLU()
        self.conv2d_148 = nn.Conv2d(in_channels=conv2d_148_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_149 = nn.ReLU()
        self.conv2d_150 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_151 = P.Concat(axis=1)
        self.batchnorm2d_152 = nn.BatchNorm2d(num_features=batchnorm2d_152_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_153 = nn.ReLU()
        self.conv2d_154 = nn.Conv2d(in_channels=conv2d_154_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_155 = nn.ReLU()
        self.conv2d_156 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_157 = P.Concat(axis=1)
        self.batchnorm2d_158 = nn.BatchNorm2d(num_features=batchnorm2d_158_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_159 = nn.ReLU()
        self.conv2d_160 = nn.Conv2d(in_channels=conv2d_160_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_161 = nn.ReLU()
        self.conv2d_162 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_163 = P.Concat(axis=1)
        self.batchnorm2d_164 = nn.BatchNorm2d(num_features=batchnorm2d_164_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_165 = nn.ReLU()
        self.conv2d_166 = nn.Conv2d(in_channels=conv2d_166_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_167 = nn.ReLU()
        self.conv2d_168 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_169 = P.Concat(axis=1)
        self.batchnorm2d_170 = nn.BatchNorm2d(num_features=batchnorm2d_170_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_171 = nn.ReLU()
        self.conv2d_172 = nn.Conv2d(in_channels=conv2d_172_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_173 = nn.ReLU()
        self.conv2d_174 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_175 = P.Concat(axis=1)
        self.batchnorm2d_176 = nn.BatchNorm2d(num_features=batchnorm2d_176_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_177 = nn.ReLU()
        self.conv2d_178 = nn.Conv2d(in_channels=conv2d_178_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_179 = nn.ReLU()
        self.conv2d_180 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_181 = P.Concat(axis=1)
        self.batchnorm2d_182 = nn.BatchNorm2d(num_features=batchnorm2d_182_num_features,
                                              eps=9.999999747378752e-06,
                                              momentum=0.8999999761581421)
        self.relu_183 = nn.ReLU()
        self.conv2d_184 = nn.Conv2d(in_channels=conv2d_184_in_channels,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_185 = nn.ReLU()
        self.conv2d_186 = nn.Conv2d(in_channels=128,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.concat_187 = P.Concat(axis=1)

    def construct(self, x):
        opt_avgpool2d_0 = self.pad_avgpool2d_0(x)
        opt_avgpool2d_0 = self.avgpool2d_0(opt_avgpool2d_0)
        module0_0_opt = self.module0_0(opt_avgpool2d_0)
        opt_concat_1 = self.concat_1((opt_avgpool2d_0, module0_0_opt,))
        opt_batchnorm2d_2 = self.batchnorm2d_2(opt_concat_1)
        opt_relu_3 = self.relu_3(opt_batchnorm2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        opt_conv2d_6 = self.conv2d_6(opt_relu_5)
        opt_concat_7 = self.concat_7((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6,))
        opt_batchnorm2d_8 = self.batchnorm2d_8(opt_concat_7)
        opt_relu_9 = self.relu_9(opt_batchnorm2d_8)
        opt_conv2d_10 = self.conv2d_10(opt_relu_9)
        opt_relu_11 = self.relu_11(opt_conv2d_10)
        opt_conv2d_12 = self.conv2d_12(opt_relu_11)
        opt_concat_13 = self.concat_13((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12,))
        opt_batchnorm2d_14 = self.batchnorm2d_14(opt_concat_13)
        opt_relu_15 = self.relu_15(opt_batchnorm2d_14)
        opt_conv2d_16 = self.conv2d_16(opt_relu_15)
        opt_relu_17 = self.relu_17(opt_conv2d_16)
        opt_conv2d_18 = self.conv2d_18(opt_relu_17)
        opt_concat_19 = self.concat_19((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18,))
        opt_batchnorm2d_20 = self.batchnorm2d_20(opt_concat_19)
        opt_relu_21 = self.relu_21(opt_batchnorm2d_20)
        opt_conv2d_22 = self.conv2d_22(opt_relu_21)
        opt_relu_23 = self.relu_23(opt_conv2d_22)
        opt_conv2d_24 = self.conv2d_24(opt_relu_23)
        opt_concat_25 = self.concat_25(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24,
             ))
        opt_batchnorm2d_26 = self.batchnorm2d_26(opt_concat_25)
        opt_relu_27 = self.relu_27(opt_batchnorm2d_26)
        opt_conv2d_28 = self.conv2d_28(opt_relu_27)
        opt_relu_29 = self.relu_29(opt_conv2d_28)
        opt_conv2d_30 = self.conv2d_30(opt_relu_29)
        opt_concat_31 = self.concat_31(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             ))
        opt_batchnorm2d_32 = self.batchnorm2d_32(opt_concat_31)
        opt_relu_33 = self.relu_33(opt_batchnorm2d_32)
        opt_conv2d_34 = self.conv2d_34(opt_relu_33)
        opt_relu_35 = self.relu_35(opt_conv2d_34)
        opt_conv2d_36 = self.conv2d_36(opt_relu_35)
        opt_concat_37 = self.concat_37((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18,
                                        opt_conv2d_24, opt_conv2d_30, opt_conv2d_36,
                                        ))
        opt_batchnorm2d_38 = self.batchnorm2d_38(opt_concat_37)
        opt_relu_39 = self.relu_39(opt_batchnorm2d_38)
        opt_conv2d_40 = self.conv2d_40(opt_relu_39)
        opt_relu_41 = self.relu_41(opt_conv2d_40)
        opt_conv2d_42 = self.conv2d_42(opt_relu_41)
        opt_concat_43 = self.concat_43((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18,
                                        opt_conv2d_24, opt_conv2d_30, opt_conv2d_36, opt_conv2d_42,
                                        ))
        opt_batchnorm2d_44 = self.batchnorm2d_44(opt_concat_43)
        opt_relu_45 = self.relu_45(opt_batchnorm2d_44)
        opt_conv2d_46 = self.conv2d_46(opt_relu_45)
        opt_relu_47 = self.relu_47(opt_conv2d_46)
        opt_conv2d_48 = self.conv2d_48(opt_relu_47)
        opt_concat_49 = self.concat_49((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18,
                                        opt_conv2d_24, opt_conv2d_30, opt_conv2d_36, opt_conv2d_42, opt_conv2d_48,
                                        ))
        opt_batchnorm2d_50 = self.batchnorm2d_50(opt_concat_49)
        opt_relu_51 = self.relu_51(opt_batchnorm2d_50)
        opt_conv2d_52 = self.conv2d_52(opt_relu_51)
        opt_relu_53 = self.relu_53(opt_conv2d_52)
        opt_conv2d_54 = self.conv2d_54(opt_relu_53)
        opt_concat_55 = self.concat_55(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54,
             ))
        opt_batchnorm2d_56 = self.batchnorm2d_56(opt_concat_55)
        opt_relu_57 = self.relu_57(opt_batchnorm2d_56)
        opt_conv2d_58 = self.conv2d_58(opt_relu_57)
        opt_relu_59 = self.relu_59(opt_conv2d_58)
        opt_conv2d_60 = self.conv2d_60(opt_relu_59)
        opt_concat_61 = self.concat_61(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60,
             ))
        opt_batchnorm2d_62 = self.batchnorm2d_62(opt_concat_61)
        opt_relu_63 = self.relu_63(opt_batchnorm2d_62)
        opt_conv2d_64 = self.conv2d_64(opt_relu_63)
        opt_relu_65 = self.relu_65(opt_conv2d_64)
        opt_conv2d_66 = self.conv2d_66(opt_relu_65)
        opt_concat_67 = self.concat_67(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66,
             ))
        opt_batchnorm2d_68 = self.batchnorm2d_68(opt_concat_67)
        opt_relu_69 = self.relu_69(opt_batchnorm2d_68)
        opt_conv2d_70 = self.conv2d_70(opt_relu_69)
        opt_relu_71 = self.relu_71(opt_conv2d_70)
        opt_conv2d_72 = self.conv2d_72(opt_relu_71)
        opt_concat_73 = self.concat_73(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             ))
        opt_batchnorm2d_74 = self.batchnorm2d_74(opt_concat_73)
        opt_relu_75 = self.relu_75(opt_batchnorm2d_74)
        opt_conv2d_76 = self.conv2d_76(opt_relu_75)
        opt_relu_77 = self.relu_77(opt_conv2d_76)
        opt_conv2d_78 = self.conv2d_78(opt_relu_77)
        opt_concat_79 = self.concat_79((opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18,
                                        opt_conv2d_24, opt_conv2d_30, opt_conv2d_36, opt_conv2d_42, opt_conv2d_48,
                                        opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72, opt_conv2d_78,
                                        ))
        opt_batchnorm2d_80 = self.batchnorm2d_80(opt_concat_79)
        opt_relu_81 = self.relu_81(opt_batchnorm2d_80)
        opt_conv2d_82 = self.conv2d_82(opt_relu_81)
        opt_relu_83 = self.relu_83(opt_conv2d_82)
        opt_conv2d_84 = self.conv2d_84(opt_relu_83)
        opt_concat_85 = self.concat_85(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84,
             ))
        opt_batchnorm2d_86 = self.batchnorm2d_86(opt_concat_85)
        opt_relu_87 = self.relu_87(opt_batchnorm2d_86)
        opt_conv2d_88 = self.conv2d_88(opt_relu_87)
        opt_relu_89 = self.relu_89(opt_conv2d_88)
        opt_conv2d_90 = self.conv2d_90(opt_relu_89)
        opt_concat_91 = self.concat_91(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90,
             ))
        opt_batchnorm2d_92 = self.batchnorm2d_92(opt_concat_91)
        opt_relu_93 = self.relu_93(opt_batchnorm2d_92)
        opt_conv2d_94 = self.conv2d_94(opt_relu_93)
        opt_relu_95 = self.relu_95(opt_conv2d_94)
        opt_conv2d_96 = self.conv2d_96(opt_relu_95)
        opt_concat_97 = self.concat_97(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96,
             ))
        opt_batchnorm2d_98 = self.batchnorm2d_98(opt_concat_97)
        opt_relu_99 = self.relu_99(opt_batchnorm2d_98)
        opt_conv2d_100 = self.conv2d_100(opt_relu_99)
        opt_relu_101 = self.relu_101(opt_conv2d_100)
        opt_conv2d_102 = self.conv2d_102(opt_relu_101)
        opt_concat_103 = self.concat_103(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102,
             ))
        opt_batchnorm2d_104 = self.batchnorm2d_104(opt_concat_103)
        opt_relu_105 = self.relu_105(opt_batchnorm2d_104)
        opt_conv2d_106 = self.conv2d_106(opt_relu_105)
        opt_relu_107 = self.relu_107(opt_conv2d_106)
        opt_conv2d_108 = self.conv2d_108(opt_relu_107)
        opt_concat_109 = self.concat_109(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108,
             ))
        opt_batchnorm2d_110 = self.batchnorm2d_110(opt_concat_109)
        opt_relu_111 = self.relu_111(opt_batchnorm2d_110)
        opt_conv2d_112 = self.conv2d_112(opt_relu_111)
        opt_relu_113 = self.relu_113(opt_conv2d_112)
        opt_conv2d_114 = self.conv2d_114(opt_relu_113)
        opt_concat_115 = self.concat_115(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             ))
        opt_batchnorm2d_116 = self.batchnorm2d_116(opt_concat_115)
        opt_relu_117 = self.relu_117(opt_batchnorm2d_116)
        opt_conv2d_118 = self.conv2d_118(opt_relu_117)
        opt_relu_119 = self.relu_119(opt_conv2d_118)
        opt_conv2d_120 = self.conv2d_120(opt_relu_119)
        opt_concat_121 = self.concat_121(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120,
             ))
        opt_batchnorm2d_122 = self.batchnorm2d_122(opt_concat_121)
        opt_relu_123 = self.relu_123(opt_batchnorm2d_122)
        opt_conv2d_124 = self.conv2d_124(opt_relu_123)
        opt_relu_125 = self.relu_125(opt_conv2d_124)
        opt_conv2d_126 = self.conv2d_126(opt_relu_125)
        opt_concat_127 = self.concat_127(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126,
             ))
        opt_batchnorm2d_128 = self.batchnorm2d_128(opt_concat_127)
        opt_relu_129 = self.relu_129(opt_batchnorm2d_128)
        opt_conv2d_130 = self.conv2d_130(opt_relu_129)
        opt_relu_131 = self.relu_131(opt_conv2d_130)
        opt_conv2d_132 = self.conv2d_132(opt_relu_131)
        opt_concat_133 = self.concat_133(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132,
             ))
        opt_batchnorm2d_134 = self.batchnorm2d_134(opt_concat_133)
        opt_relu_135 = self.relu_135(opt_batchnorm2d_134)
        opt_conv2d_136 = self.conv2d_136(opt_relu_135)
        opt_relu_137 = self.relu_137(opt_conv2d_136)
        opt_conv2d_138 = self.conv2d_138(opt_relu_137)
        opt_concat_139 = self.concat_139(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138,
             ))
        opt_batchnorm2d_140 = self.batchnorm2d_140(opt_concat_139)
        opt_relu_141 = self.relu_141(opt_batchnorm2d_140)
        opt_conv2d_142 = self.conv2d_142(opt_relu_141)
        opt_relu_143 = self.relu_143(opt_conv2d_142)
        opt_conv2d_144 = self.conv2d_144(opt_relu_143)
        opt_concat_145 = self.concat_145(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144,
             ))
        opt_batchnorm2d_146 = self.batchnorm2d_146(opt_concat_145)
        opt_relu_147 = self.relu_147(opt_batchnorm2d_146)
        opt_conv2d_148 = self.conv2d_148(opt_relu_147)
        opt_relu_149 = self.relu_149(opt_conv2d_148)
        opt_conv2d_150 = self.conv2d_150(opt_relu_149)
        opt_concat_151 = self.concat_151(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             ))
        opt_batchnorm2d_152 = self.batchnorm2d_152(opt_concat_151)
        opt_relu_153 = self.relu_153(opt_batchnorm2d_152)
        opt_conv2d_154 = self.conv2d_154(opt_relu_153)
        opt_relu_155 = self.relu_155(opt_conv2d_154)
        opt_conv2d_156 = self.conv2d_156(opt_relu_155)
        opt_concat_157 = self.concat_157(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             opt_conv2d_156,
             ))
        opt_batchnorm2d_158 = self.batchnorm2d_158(opt_concat_157)
        opt_relu_159 = self.relu_159(opt_batchnorm2d_158)
        opt_conv2d_160 = self.conv2d_160(opt_relu_159)
        opt_relu_161 = self.relu_161(opt_conv2d_160)
        opt_conv2d_162 = self.conv2d_162(opt_relu_161)
        opt_concat_163 = self.concat_163(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             opt_conv2d_156, opt_conv2d_162,
             ))
        opt_batchnorm2d_164 = self.batchnorm2d_164(opt_concat_163)
        opt_relu_165 = self.relu_165(opt_batchnorm2d_164)
        opt_conv2d_166 = self.conv2d_166(opt_relu_165)
        opt_relu_167 = self.relu_167(opt_conv2d_166)
        opt_conv2d_168 = self.conv2d_168(opt_relu_167)
        opt_concat_169 = self.concat_169(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             opt_conv2d_156, opt_conv2d_162, opt_conv2d_168,
             ))
        opt_batchnorm2d_170 = self.batchnorm2d_170(opt_concat_169)
        opt_relu_171 = self.relu_171(opt_batchnorm2d_170)
        opt_conv2d_172 = self.conv2d_172(opt_relu_171)
        opt_relu_173 = self.relu_173(opt_conv2d_172)
        opt_conv2d_174 = self.conv2d_174(opt_relu_173)
        opt_concat_175 = self.concat_175(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             opt_conv2d_156, opt_conv2d_162, opt_conv2d_168, opt_conv2d_174,
             ))
        opt_batchnorm2d_176 = self.batchnorm2d_176(opt_concat_175)
        opt_relu_177 = self.relu_177(opt_batchnorm2d_176)
        opt_conv2d_178 = self.conv2d_178(opt_relu_177)
        opt_relu_179 = self.relu_179(opt_conv2d_178)
        opt_conv2d_180 = self.conv2d_180(opt_relu_179)
        opt_concat_181 = self.concat_181(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             opt_conv2d_156, opt_conv2d_162, opt_conv2d_168, opt_conv2d_174, opt_conv2d_180,
             ))
        opt_batchnorm2d_182 = self.batchnorm2d_182(opt_concat_181)
        opt_relu_183 = self.relu_183(opt_batchnorm2d_182)
        opt_conv2d_184 = self.conv2d_184(opt_relu_183)
        opt_relu_185 = self.relu_185(opt_conv2d_184)
        opt_conv2d_186 = self.conv2d_186(opt_relu_185)
        opt_concat_187 = self.concat_187(
            (opt_avgpool2d_0, module0_0_opt, opt_conv2d_6, opt_conv2d_12, opt_conv2d_18, opt_conv2d_24, opt_conv2d_30,
             opt_conv2d_36, opt_conv2d_42, opt_conv2d_48, opt_conv2d_54, opt_conv2d_60, opt_conv2d_66, opt_conv2d_72,
             opt_conv2d_78, opt_conv2d_84, opt_conv2d_90, opt_conv2d_96, opt_conv2d_102, opt_conv2d_108, opt_conv2d_114,
             opt_conv2d_120, opt_conv2d_126, opt_conv2d_132, opt_conv2d_138, opt_conv2d_144, opt_conv2d_150,
             opt_conv2d_156, opt_conv2d_162, opt_conv2d_168, opt_conv2d_174, opt_conv2d_180, opt_conv2d_186,
             ))
        return opt_concat_187


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
        self.module24_0 = Module24(conv2d_0_in_channels=96, module21_0_batchnorm2d_0_num_features=96)
        self.concat_15 = P.Concat(axis=1)
        self.module24_1 = Module24(conv2d_0_in_channels=128, module21_0_batchnorm2d_0_num_features=128)
        self.concat_21 = P.Concat(axis=1)
        self.module24_2 = Module24(conv2d_0_in_channels=160, module21_0_batchnorm2d_0_num_features=160)
        self.concat_27 = P.Concat(axis=1)
        self.module24_3 = Module24(conv2d_0_in_channels=192, module21_0_batchnorm2d_0_num_features=192)
        self.concat_33 = P.Concat(axis=1)
        self.module24_4 = Module24(conv2d_0_in_channels=224, module21_0_batchnorm2d_0_num_features=224)
        self.concat_39 = P.Concat(axis=1)
        self.module26_0 = Module26(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=128,
                                   module21_0_batchnorm2d_0_num_features=256)
        self.pad_avgpool2d_43 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_43 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module0_1 = Module0(batchnorm2d_1_num_features=128, conv2d_3_in_channels=128)
        self.concat_50 = P.Concat(axis=1)
        self.module24_5 = Module24(conv2d_0_in_channels=160, module21_0_batchnorm2d_0_num_features=160)
        self.concat_56 = P.Concat(axis=1)
        self.module24_6 = Module24(conv2d_0_in_channels=192, module21_0_batchnorm2d_0_num_features=192)
        self.concat_62 = P.Concat(axis=1)
        self.module24_7 = Module24(conv2d_0_in_channels=224, module21_0_batchnorm2d_0_num_features=224)
        self.concat_68 = P.Concat(axis=1)
        self.module24_8 = Module24(conv2d_0_in_channels=256, module21_0_batchnorm2d_0_num_features=256)
        self.concat_74 = P.Concat(axis=1)
        self.module24_9 = Module24(conv2d_0_in_channels=288, module21_0_batchnorm2d_0_num_features=288)
        self.concat_80 = P.Concat(axis=1)
        self.module24_10 = Module24(conv2d_0_in_channels=320, module21_0_batchnorm2d_0_num_features=320)
        self.concat_86 = P.Concat(axis=1)
        self.module24_11 = Module24(conv2d_0_in_channels=352, module21_0_batchnorm2d_0_num_features=352)
        self.concat_92 = P.Concat(axis=1)
        self.module24_12 = Module24(conv2d_0_in_channels=384, module21_0_batchnorm2d_0_num_features=384)
        self.concat_98 = P.Concat(axis=1)
        self.module24_13 = Module24(conv2d_0_in_channels=416, module21_0_batchnorm2d_0_num_features=416)
        self.concat_104 = P.Concat(axis=1)
        self.module24_14 = Module24(conv2d_0_in_channels=448, module21_0_batchnorm2d_0_num_features=448)
        self.concat_110 = P.Concat(axis=1)
        self.module24_15 = Module24(conv2d_0_in_channels=480, module21_0_batchnorm2d_0_num_features=480)
        self.concat_116 = P.Concat(axis=1)
        self.module26_1 = Module26(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=256,
                                   module21_0_batchnorm2d_0_num_features=512)
        self.module20_0 = Module20(batchnorm2d_2_num_features=288,
                                   conv2d_4_in_channels=288,
                                   batchnorm2d_8_num_features=320,
                                   conv2d_10_in_channels=320,
                                   batchnorm2d_14_num_features=352,
                                   conv2d_16_in_channels=352,
                                   batchnorm2d_20_num_features=384,
                                   conv2d_22_in_channels=384,
                                   batchnorm2d_26_num_features=416,
                                   conv2d_28_in_channels=416,
                                   batchnorm2d_32_num_features=448,
                                   conv2d_34_in_channels=448,
                                   batchnorm2d_38_num_features=480,
                                   conv2d_40_in_channels=480,
                                   batchnorm2d_44_num_features=512,
                                   conv2d_46_in_channels=512,
                                   batchnorm2d_50_num_features=544,
                                   conv2d_52_in_channels=544,
                                   batchnorm2d_56_num_features=576,
                                   conv2d_58_in_channels=576,
                                   batchnorm2d_62_num_features=608,
                                   conv2d_64_in_channels=608,
                                   batchnorm2d_68_num_features=640,
                                   conv2d_70_in_channels=640,
                                   batchnorm2d_74_num_features=672,
                                   conv2d_76_in_channels=672,
                                   batchnorm2d_80_num_features=704,
                                   conv2d_82_in_channels=704,
                                   batchnorm2d_86_num_features=736,
                                   conv2d_88_in_channels=736,
                                   batchnorm2d_92_num_features=768,
                                   conv2d_94_in_channels=768,
                                   batchnorm2d_98_num_features=800,
                                   conv2d_100_in_channels=800,
                                   batchnorm2d_104_num_features=832,
                                   conv2d_106_in_channels=832,
                                   batchnorm2d_110_num_features=864,
                                   conv2d_112_in_channels=864,
                                   batchnorm2d_116_num_features=896,
                                   conv2d_118_in_channels=896,
                                   batchnorm2d_122_num_features=928,
                                   conv2d_124_in_channels=928,
                                   batchnorm2d_128_num_features=960,
                                   conv2d_130_in_channels=960,
                                   batchnorm2d_134_num_features=992,
                                   conv2d_136_in_channels=992,
                                   batchnorm2d_140_num_features=1024,
                                   conv2d_142_in_channels=1024,
                                   batchnorm2d_146_num_features=1056,
                                   conv2d_148_in_channels=1056,
                                   batchnorm2d_152_num_features=1088,
                                   conv2d_154_in_channels=1088,
                                   batchnorm2d_158_num_features=1120,
                                   conv2d_160_in_channels=1120,
                                   batchnorm2d_164_num_features=1152,
                                   conv2d_166_in_channels=1152,
                                   batchnorm2d_170_num_features=1184,
                                   conv2d_172_in_channels=1184,
                                   batchnorm2d_176_num_features=1216,
                                   conv2d_178_in_channels=1216,
                                   batchnorm2d_182_num_features=1248,
                                   conv2d_184_in_channels=1248,
                                   module0_0_batchnorm2d_1_num_features=256,
                                   module0_0_conv2d_3_in_channels=256)
        self.module26_2 = Module26(conv2d_0_in_channels=1280,
                                   conv2d_0_out_channels=640,
                                   module21_0_batchnorm2d_0_num_features=1280)
        self.module20_1 = Module20(batchnorm2d_2_num_features=672,
                                   conv2d_4_in_channels=672,
                                   batchnorm2d_8_num_features=704,
                                   conv2d_10_in_channels=704,
                                   batchnorm2d_14_num_features=736,
                                   conv2d_16_in_channels=736,
                                   batchnorm2d_20_num_features=768,
                                   conv2d_22_in_channels=768,
                                   batchnorm2d_26_num_features=800,
                                   conv2d_28_in_channels=800,
                                   batchnorm2d_32_num_features=832,
                                   conv2d_34_in_channels=832,
                                   batchnorm2d_38_num_features=864,
                                   conv2d_40_in_channels=864,
                                   batchnorm2d_44_num_features=896,
                                   conv2d_46_in_channels=896,
                                   batchnorm2d_50_num_features=928,
                                   conv2d_52_in_channels=928,
                                   batchnorm2d_56_num_features=960,
                                   conv2d_58_in_channels=960,
                                   batchnorm2d_62_num_features=992,
                                   conv2d_64_in_channels=992,
                                   batchnorm2d_68_num_features=1024,
                                   conv2d_70_in_channels=1024,
                                   batchnorm2d_74_num_features=1056,
                                   conv2d_76_in_channels=1056,
                                   batchnorm2d_80_num_features=1088,
                                   conv2d_82_in_channels=1088,
                                   batchnorm2d_86_num_features=1120,
                                   conv2d_88_in_channels=1120,
                                   batchnorm2d_92_num_features=1152,
                                   conv2d_94_in_channels=1152,
                                   batchnorm2d_98_num_features=1184,
                                   conv2d_100_in_channels=1184,
                                   batchnorm2d_104_num_features=1216,
                                   conv2d_106_in_channels=1216,
                                   batchnorm2d_110_num_features=1248,
                                   conv2d_112_in_channels=1248,
                                   batchnorm2d_116_num_features=1280,
                                   conv2d_118_in_channels=1280,
                                   batchnorm2d_122_num_features=1312,
                                   conv2d_124_in_channels=1312,
                                   batchnorm2d_128_num_features=1344,
                                   conv2d_130_in_channels=1344,
                                   batchnorm2d_134_num_features=1376,
                                   conv2d_136_in_channels=1376,
                                   batchnorm2d_140_num_features=1408,
                                   conv2d_142_in_channels=1408,
                                   batchnorm2d_146_num_features=1440,
                                   conv2d_148_in_channels=1440,
                                   batchnorm2d_152_num_features=1472,
                                   conv2d_154_in_channels=1472,
                                   batchnorm2d_158_num_features=1504,
                                   conv2d_160_in_channels=1504,
                                   batchnorm2d_164_num_features=1536,
                                   conv2d_166_in_channels=1536,
                                   batchnorm2d_170_num_features=1568,
                                   conv2d_172_in_channels=1568,
                                   batchnorm2d_176_num_features=1600,
                                   conv2d_178_in_channels=1600,
                                   batchnorm2d_182_num_features=1632,
                                   conv2d_184_in_channels=1632,
                                   module0_0_batchnorm2d_1_num_features=640,
                                   module0_0_conv2d_3_in_channels=640)
        self.module21_0 = Module21(batchnorm2d_0_num_features=1664)
        self.avgpool2d_513 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_514 = nn.Flatten()
        self.dense_515 = nn.Dense(in_channels=1664, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module0_0_opt = self.module0_0(opt_maxpool2d_2)
        opt_concat_9 = self.concat_9((opt_maxpool2d_2, module0_0_opt,))
        module24_0_opt = self.module24_0(opt_concat_9)
        opt_concat_15 = self.concat_15((opt_maxpool2d_2, module0_0_opt, module24_0_opt,))
        module24_1_opt = self.module24_1(opt_concat_15)
        opt_concat_21 = self.concat_21((opt_maxpool2d_2, module0_0_opt, module24_0_opt, module24_1_opt,))
        module24_2_opt = self.module24_2(opt_concat_21)
        opt_concat_27 = self.concat_27((opt_maxpool2d_2, module0_0_opt, module24_0_opt, module24_1_opt, module24_2_opt,
                                        ))
        module24_3_opt = self.module24_3(opt_concat_27)
        opt_concat_33 = self.concat_33(
            (opt_maxpool2d_2, module0_0_opt, module24_0_opt, module24_1_opt, module24_2_opt, module24_3_opt,
             ))
        module24_4_opt = self.module24_4(opt_concat_33)
        opt_concat_39 = self.concat_39((opt_maxpool2d_2, module0_0_opt, module24_0_opt, module24_1_opt, module24_2_opt,
                                        module24_3_opt, module24_4_opt,
                                        ))
        module26_0_opt = self.module26_0(opt_concat_39)
        opt_avgpool2d_43 = self.pad_avgpool2d_43(module26_0_opt)
        opt_avgpool2d_43 = self.avgpool2d_43(opt_avgpool2d_43)
        module0_1_opt = self.module0_1(opt_avgpool2d_43)
        opt_concat_50 = self.concat_50((opt_avgpool2d_43, module0_1_opt,))
        module24_5_opt = self.module24_5(opt_concat_50)
        opt_concat_56 = self.concat_56((opt_avgpool2d_43, module0_1_opt, module24_5_opt,))
        module24_6_opt = self.module24_6(opt_concat_56)
        opt_concat_62 = self.concat_62((opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt,))
        module24_7_opt = self.module24_7(opt_concat_62)
        opt_concat_68 = self.concat_68((opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt,
                                        ))
        module24_8_opt = self.module24_8(opt_concat_68)
        opt_concat_74 = self.concat_74(
            (opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt, module24_8_opt,
             ))
        module24_9_opt = self.module24_9(opt_concat_74)
        opt_concat_80 = self.concat_80((opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt,
                                        module24_8_opt, module24_9_opt,
                                        ))
        module24_10_opt = self.module24_10(opt_concat_80)
        opt_concat_86 = self.concat_86((opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt,
                                        module24_8_opt, module24_9_opt, module24_10_opt,
                                        ))
        module24_11_opt = self.module24_11(opt_concat_86)
        opt_concat_92 = self.concat_92((opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt,
                                        module24_8_opt, module24_9_opt, module24_10_opt, module24_11_opt,
                                        ))
        module24_12_opt = self.module24_12(opt_concat_92)
        opt_concat_98 = self.concat_98(
            (opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt, module24_8_opt,
             module24_9_opt, module24_10_opt, module24_11_opt, module24_12_opt,
             ))
        module24_13_opt = self.module24_13(opt_concat_98)
        opt_concat_104 = self.concat_104(
            (opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt, module24_8_opt,
             module24_9_opt, module24_10_opt, module24_11_opt, module24_12_opt, module24_13_opt,
             ))
        module24_14_opt = self.module24_14(opt_concat_104)
        opt_concat_110 = self.concat_110(
            (opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt, module24_8_opt,
             module24_9_opt, module24_10_opt, module24_11_opt, module24_12_opt, module24_13_opt, module24_14_opt,
             ))
        module24_15_opt = self.module24_15(opt_concat_110)
        opt_concat_116 = self.concat_116(
            (opt_avgpool2d_43, module0_1_opt, module24_5_opt, module24_6_opt, module24_7_opt, module24_8_opt,
             module24_9_opt, module24_10_opt, module24_11_opt, module24_12_opt, module24_13_opt, module24_14_opt,
             module24_15_opt,
             ))
        module26_1_opt = self.module26_1(opt_concat_116)
        module20_0_opt = self.module20_0(module26_1_opt)
        module26_2_opt = self.module26_2(module20_0_opt)
        module20_1_opt = self.module20_1(module26_2_opt)
        module21_0_opt = self.module21_0(module20_1_opt)
        opt_avgpool2d_513 = self.avgpool2d_513(module21_0_opt)
        opt_flatten_514 = self.flatten_514(opt_avgpool2d_513)
        opt_dense_515 = self.dense_515(opt_flatten_514)
        return opt_dense_515
