import mindspore.ops as P
from mindspore import nn


class Module2(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, conv2d_0_group):
        super(Module2, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=conv2d_0_group,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module4(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module2_0_conv2d_0_in_channels,
                 module2_0_conv2d_0_out_channels, module2_0_conv2d_0_kernel_size, module2_0_conv2d_0_stride,
                 module2_0_conv2d_0_padding, module2_0_conv2d_0_pad_mode, module2_0_conv2d_0_group):
        super(Module4, self).__init__()
        self.module2_0 = Module2(conv2d_0_in_channels=module2_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_0_conv2d_0_stride,
                                 conv2d_0_padding=module2_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module2_0_conv2d_0_group)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module2_0_opt = self.module2_0(x)
        opt_conv2d_0 = self.conv2d_0(module2_0_opt)
        return opt_conv2d_0


class Module8(nn.Cell):
    def __init__(self, conv2d_0_in_channels, module2_0_conv2d_0_in_channels, module2_0_conv2d_0_out_channels,
                 module2_0_conv2d_0_kernel_size, module2_0_conv2d_0_stride, module2_0_conv2d_0_padding,
                 module2_0_conv2d_0_pad_mode, module2_0_conv2d_0_group, module2_1_conv2d_0_in_channels,
                 module2_1_conv2d_0_out_channels, module2_1_conv2d_0_kernel_size, module2_1_conv2d_0_stride,
                 module2_1_conv2d_0_padding, module2_1_conv2d_0_pad_mode, module2_1_conv2d_0_group):
        super(Module8, self).__init__()
        self.module2_0 = Module2(conv2d_0_in_channels=module2_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_0_conv2d_0_stride,
                                 conv2d_0_padding=module2_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module2_0_conv2d_0_group)
        self.module2_1 = Module2(conv2d_0_in_channels=module2_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module2_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module2_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module2_1_conv2d_0_stride,
                                 conv2d_0_padding=module2_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module2_1_conv2d_0_pad_mode,
                                 conv2d_0_group=module2_1_conv2d_0_group)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=640,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module2_0_opt = self.module2_0(x)
        module2_1_opt = self.module2_1(module2_0_opt)
        opt_conv2d_0 = self.conv2d_0(module2_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):
    def __init__(self, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels, conv2d_2_group,
                 conv2d_4_in_channels):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=640,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=conv2d_2_group,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=640,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_add_5 = P.Add()(opt_conv2d_4, x)
        opt_relu_6 = self.relu_6(opt_add_5)
        return opt_relu_6


class Module17(nn.Cell):
    def __init__(self, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels, module0_0_conv2d_2_out_channels,
                 module0_0_conv2d_2_group, module0_0_conv2d_4_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_conv2d_2_group,
                 module0_1_conv2d_4_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_2_group, module0_2_conv2d_4_in_channels,
                 module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels, module0_3_conv2d_2_out_channels,
                 module0_3_conv2d_2_group, module0_3_conv2d_4_in_channels):
        super(Module17, self).__init__()
        self.module0_0 = Module0(conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_group=module0_0_conv2d_2_group,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels)
        self.module0_1 = Module0(conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_group=module0_1_conv2d_2_group,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels)
        self.module0_2 = Module0(conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_2_group=module0_2_conv2d_2_group,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels)
        self.module0_3 = Module0(conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_2_group=module0_3_conv2d_2_group,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module4_0 = Module4(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 module2_0_conv2d_0_in_channels=32,
                                 module2_0_conv2d_0_out_channels=128,
                                 module2_0_conv2d_0_kernel_size=(3, 3),
                                 module2_0_conv2d_0_stride=(2, 2),
                                 module2_0_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_0_conv2d_0_pad_mode="pad",
                                 module2_0_conv2d_0_group=1)
        self.conv2d_3 = nn.Conv2d(in_channels=32,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()
        self.module4_1 = Module4(conv2d_0_in_channels=192,
                                 conv2d_0_out_channels=192,
                                 module2_0_conv2d_0_in_channels=128,
                                 module2_0_conv2d_0_out_channels=192,
                                 module2_0_conv2d_0_kernel_size=(3, 3),
                                 module2_0_conv2d_0_stride=(2, 2),
                                 module2_0_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_0_conv2d_0_pad_mode="pad",
                                 module2_0_conv2d_0_group=1)
        self.conv2d_9 = nn.Conv2d(in_channels=128,
                                  out_channels=192,
                                  kernel_size=(1, 1),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_13 = nn.ReLU()
        self.module4_2 = Module4(conv2d_0_in_channels=192,
                                 conv2d_0_out_channels=192,
                                 module2_0_conv2d_0_in_channels=192,
                                 module2_0_conv2d_0_out_channels=192,
                                 module2_0_conv2d_0_kernel_size=(3, 3),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_0_conv2d_0_pad_mode="pad",
                                 module2_0_conv2d_0_group=1)
        self.relu_18 = nn.ReLU()
        self.module8_0 = Module8(conv2d_0_in_channels=160,
                                 module2_0_conv2d_0_in_channels=192,
                                 module2_0_conv2d_0_out_channels=160,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_0_conv2d_0_group=1,
                                 module2_1_conv2d_0_in_channels=160,
                                 module2_1_conv2d_0_out_channels=160,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad",
                                 module2_1_conv2d_0_group=1)
        self.conv2d_20 = nn.Conv2d(in_channels=192,
                                   out_channels=640,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_26 = nn.ReLU()
        self.module17_0 = Module17(module0_0_conv2d_0_out_channels=160,
                                   module0_0_conv2d_2_in_channels=160,
                                   module0_0_conv2d_2_out_channels=160,
                                   module0_0_conv2d_2_group=1,
                                   module0_0_conv2d_4_in_channels=160,
                                   module0_1_conv2d_0_out_channels=160,
                                   module0_1_conv2d_2_in_channels=160,
                                   module0_1_conv2d_2_out_channels=160,
                                   module0_1_conv2d_2_group=1,
                                   module0_1_conv2d_4_in_channels=160,
                                   module0_2_conv2d_0_out_channels=160,
                                   module0_2_conv2d_2_in_channels=160,
                                   module0_2_conv2d_2_out_channels=160,
                                   module0_2_conv2d_2_group=1,
                                   module0_2_conv2d_4_in_channels=160,
                                   module0_3_conv2d_0_out_channels=160,
                                   module0_3_conv2d_2_in_channels=160,
                                   module0_3_conv2d_2_out_channels=160,
                                   module0_3_conv2d_2_group=1,
                                   module0_3_conv2d_4_in_channels=160)
        self.module0_0 = Module0(conv2d_0_out_channels=160,
                                 conv2d_2_in_channels=160,
                                 conv2d_2_out_channels=160,
                                 conv2d_2_group=1,
                                 conv2d_4_in_channels=160)
        self.module8_1 = Module8(conv2d_0_in_channels=1920,
                                 module2_0_conv2d_0_in_channels=640,
                                 module2_0_conv2d_0_out_channels=1920,
                                 module2_0_conv2d_0_kernel_size=(1, 1),
                                 module2_0_conv2d_0_stride=(1, 1),
                                 module2_0_conv2d_0_padding=0,
                                 module2_0_conv2d_0_pad_mode="valid",
                                 module2_0_conv2d_0_group=1,
                                 module2_1_conv2d_0_in_channels=1920,
                                 module2_1_conv2d_0_out_channels=1920,
                                 module2_1_conv2d_0_kernel_size=(3, 3),
                                 module2_1_conv2d_0_stride=(2, 2),
                                 module2_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module2_1_conv2d_0_pad_mode="pad",
                                 module2_1_conv2d_0_group=1920)
        self.conv2d_63 = nn.Conv2d(in_channels=640,
                                   out_channels=640,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_69 = nn.ReLU()
        self.module17_1 = Module17(module0_0_conv2d_0_out_channels=1920,
                                   module0_0_conv2d_2_in_channels=1920,
                                   module0_0_conv2d_2_out_channels=1920,
                                   module0_0_conv2d_2_group=1920,
                                   module0_0_conv2d_4_in_channels=1920,
                                   module0_1_conv2d_0_out_channels=1920,
                                   module0_1_conv2d_2_in_channels=1920,
                                   module0_1_conv2d_2_out_channels=1920,
                                   module0_1_conv2d_2_group=1920,
                                   module0_1_conv2d_4_in_channels=1920,
                                   module0_2_conv2d_0_out_channels=1920,
                                   module0_2_conv2d_2_in_channels=1920,
                                   module0_2_conv2d_2_out_channels=1920,
                                   module0_2_conv2d_2_group=1920,
                                   module0_2_conv2d_4_in_channels=1920,
                                   module0_3_conv2d_0_out_channels=1920,
                                   module0_3_conv2d_2_in_channels=1920,
                                   module0_3_conv2d_2_out_channels=1920,
                                   module0_3_conv2d_2_group=1920,
                                   module0_3_conv2d_4_in_channels=1920)
        self.module17_2 = Module17(module0_0_conv2d_0_out_channels=1920,
                                   module0_0_conv2d_2_in_channels=1920,
                                   module0_0_conv2d_2_out_channels=1920,
                                   module0_0_conv2d_2_group=1920,
                                   module0_0_conv2d_4_in_channels=1920,
                                   module0_1_conv2d_0_out_channels=1920,
                                   module0_1_conv2d_2_in_channels=1920,
                                   module0_1_conv2d_2_out_channels=1920,
                                   module0_1_conv2d_2_group=1920,
                                   module0_1_conv2d_4_in_channels=1920,
                                   module0_2_conv2d_0_out_channels=1920,
                                   module0_2_conv2d_2_in_channels=1920,
                                   module0_2_conv2d_2_out_channels=1920,
                                   module0_2_conv2d_2_group=1920,
                                   module0_2_conv2d_4_in_channels=1920,
                                   module0_3_conv2d_0_out_channels=1920,
                                   module0_3_conv2d_2_in_channels=1920,
                                   module0_3_conv2d_2_out_channels=1920,
                                   module0_3_conv2d_2_group=1920,
                                   module0_3_conv2d_4_in_channels=1920)
        self.module2_0 = Module2(conv2d_0_in_channels=640,
                                 conv2d_0_out_channels=2560,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid",
                                 conv2d_0_group=1)
        self.avgpool2d_128 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_129 = nn.Flatten()
        self.dense_130 = nn.Dense(in_channels=2560, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module4_0_opt = self.module4_0(opt_relu_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_1)
        opt_add_6 = P.Add()(module4_0_opt, opt_conv2d_3)
        opt_relu_7 = self.relu_7(opt_add_6)
        module4_1_opt = self.module4_1(opt_relu_7)
        opt_conv2d_9 = self.conv2d_9(opt_relu_7)
        opt_add_12 = P.Add()(module4_1_opt, opt_conv2d_9)
        opt_relu_13 = self.relu_13(opt_add_12)
        module4_2_opt = self.module4_2(opt_relu_13)
        opt_add_17 = P.Add()(module4_2_opt, opt_relu_13)
        opt_relu_18 = self.relu_18(opt_add_17)
        module8_0_opt = self.module8_0(opt_relu_18)
        opt_conv2d_20 = self.conv2d_20(opt_relu_18)
        opt_add_25 = P.Add()(module8_0_opt, opt_conv2d_20)
        opt_relu_26 = self.relu_26(opt_add_25)
        module17_0_opt = self.module17_0(opt_relu_26)
        module0_0_opt = self.module0_0(module17_0_opt)
        module8_1_opt = self.module8_1(module0_0_opt)
        opt_conv2d_63 = self.conv2d_63(module0_0_opt)
        opt_add_68 = P.Add()(module8_1_opt, opt_conv2d_63)
        opt_relu_69 = self.relu_69(opt_add_68)
        module17_1_opt = self.module17_1(opt_relu_69)
        module17_2_opt = self.module17_2(module17_1_opt)
        module2_0_opt = self.module2_0(module17_2_opt)
        opt_avgpool2d_128 = self.avgpool2d_128(module2_0_opt)
        opt_flatten_129 = self.flatten_129(opt_avgpool2d_128)
        opt_dense_130 = self.dense_130(opt_flatten_129)
        return opt_dense_130
