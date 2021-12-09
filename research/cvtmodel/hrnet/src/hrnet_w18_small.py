import mindspore.ops as P
from mindspore import nn


class Module3(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module3, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module10(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode, module3_0_conv2d_0_in_channels, module3_0_conv2d_0_out_channels,
                 module3_0_conv2d_0_kernel_size, module3_0_conv2d_0_stride, module3_0_conv2d_0_padding,
                 module3_0_conv2d_0_pad_mode, module3_1_conv2d_0_in_channels, module3_1_conv2d_0_out_channels,
                 module3_1_conv2d_0_kernel_size, module3_1_conv2d_0_stride, module3_1_conv2d_0_padding,
                 module3_1_conv2d_0_pad_mode):
        super(Module10, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_0_conv2d_0_stride,
                                 conv2d_0_padding=module3_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_0_conv2d_0_pad_mode)
        self.module3_1 = Module3(conv2d_0_in_channels=module3_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_1_conv2d_0_stride,
                                 conv2d_0_padding=module3_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module3_1_opt = self.module3_1(module3_0_opt)
        opt_conv2d_0 = self.conv2d_0(module3_1_opt)
        return opt_conv2d_0


class Module1(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_5_in_channels, conv2d_5_out_channels, conv2d_7_in_channels, conv2d_7_out_channels):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
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
                                  group=1,
                                  has_bias=True)
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(in_channels=conv2d_5_in_channels,
                                  out_channels=conv2d_5_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()
        self.conv2d_7 = nn.Conv2d(in_channels=conv2d_7_in_channels,
                                  out_channels=conv2d_7_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_9 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_add_3 = P.Add()(opt_conv2d_2, x)
        opt_relu_4 = self.relu_4(opt_add_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_conv2d_7 = self.conv2d_7(opt_relu_6)
        opt_add_8 = P.Add()(opt_conv2d_7, opt_relu_4)
        opt_relu_9 = self.relu_9(opt_add_8)
        return opt_relu_9


class Module8(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, resizenearestneighbor_1_size):
        super(Module8, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.resizenearestneighbor_1 = P.ResizeNearestNeighbor(size=resizenearestneighbor_1_size, align_corners=False)

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_resizenearestneighbor_1 = self.resizenearestneighbor_1(opt_conv2d_0)
        return opt_resizenearestneighbor_1


class Module6(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module3_0_conv2d_0_in_channels,
                 module3_0_conv2d_0_out_channels, module3_0_conv2d_0_kernel_size, module3_0_conv2d_0_stride,
                 module3_0_conv2d_0_padding, module3_0_conv2d_0_pad_mode):
        super(Module6, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_0_conv2d_0_stride,
                                 conv2d_0_padding=module3_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_0_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        opt_conv2d_0 = self.conv2d_0(module3_0_opt)
        return opt_conv2d_0


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module3_0 = Module3(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module10_0 = Module10(conv2d_0_in_channels=32,
                                   conv2d_0_out_channels=128,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module3_0_conv2d_0_in_channels=64,
                                   module3_0_conv2d_0_out_channels=32,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=32,
                                   module3_1_conv2d_0_out_channels=32,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_5 = nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_11 = nn.ReLU()
        self.module3_1 = Module3(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=16,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module3_2 = Module3(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=32,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module1_0 = Module1(conv2d_0_in_channels=16,
                                 conv2d_0_out_channels=16,
                                 conv2d_2_in_channels=16,
                                 conv2d_2_out_channels=16,
                                 conv2d_5_in_channels=16,
                                 conv2d_5_out_channels=16,
                                 conv2d_7_in_channels=16,
                                 conv2d_7_out_channels=16)
        self.module1_1 = Module1(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=32,
                                 conv2d_2_in_channels=32,
                                 conv2d_2_out_channels=32,
                                 conv2d_5_in_channels=32,
                                 conv2d_5_out_channels=32,
                                 conv2d_7_in_channels=32,
                                 conv2d_7_out_channels=32)
        self.module8_0 = Module8(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=16,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_44 = nn.ReLU()
        self.conv2d_36 = nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_40 = nn.ReLU()
        self.module3_3 = Module3(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module1_2 = Module1(conv2d_0_in_channels=16,
                                 conv2d_0_out_channels=16,
                                 conv2d_2_in_channels=16,
                                 conv2d_2_out_channels=16,
                                 conv2d_5_in_channels=16,
                                 conv2d_5_out_channels=16,
                                 conv2d_7_in_channels=16,
                                 conv2d_7_out_channels=16)
        self.module1_3 = Module1(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=32,
                                 conv2d_2_in_channels=32,
                                 conv2d_2_out_channels=32,
                                 conv2d_5_in_channels=32,
                                 conv2d_5_out_channels=32,
                                 conv2d_7_in_channels=32,
                                 conv2d_7_out_channels=32)
        self.module1_4 = Module1(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_5_in_channels=64,
                                 conv2d_5_out_channels=64,
                                 conv2d_7_in_channels=64,
                                 conv2d_7_out_channels=64)
        self.module8_1 = Module8(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=16,
                                 resizenearestneighbor_1_size=(56, 56))
        self.module8_2 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=16,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_91 = nn.ReLU()
        self.conv2d_78 = nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module8_3 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=32,
                                 resizenearestneighbor_1_size=(28, 28))
        self.relu_92 = nn.ReLU()
        self.module6_0 = Module6(conv2d_0_in_channels=16,
                                 conv2d_0_out_channels=64,
                                 module3_0_conv2d_0_in_channels=16,
                                 module3_0_conv2d_0_out_channels=16,
                                 module3_0_conv2d_0_kernel_size=(3, 3),
                                 module3_0_conv2d_0_stride=(2, 2),
                                 module3_0_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_0_conv2d_0_pad_mode="pad")
        self.conv2d_74 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_96 = nn.ReLU()
        self.module3_4 = Module3(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module1_5 = Module1(conv2d_0_in_channels=16,
                                 conv2d_0_out_channels=16,
                                 conv2d_2_in_channels=16,
                                 conv2d_2_out_channels=16,
                                 conv2d_5_in_channels=16,
                                 conv2d_5_out_channels=16,
                                 conv2d_7_in_channels=16,
                                 conv2d_7_out_channels=16)
        self.module1_6 = Module1(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=32,
                                 conv2d_2_in_channels=32,
                                 conv2d_2_out_channels=32,
                                 conv2d_5_in_channels=32,
                                 conv2d_5_out_channels=32,
                                 conv2d_7_in_channels=32,
                                 conv2d_7_out_channels=32)
        self.module1_7 = Module1(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_5_in_channels=64,
                                 conv2d_5_out_channels=64,
                                 conv2d_7_in_channels=64,
                                 conv2d_7_out_channels=64)
        self.module1_8 = Module1(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_5_in_channels=128,
                                 conv2d_5_out_channels=128,
                                 conv2d_7_in_channels=128,
                                 conv2d_7_out_channels=128)
        self.module8_4 = Module8(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=16,
                                 resizenearestneighbor_1_size=(56, 56))
        self.module8_5 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=16,
                                 resizenearestneighbor_1_size=(56, 56))
        self.module8_6 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=16,
                                 resizenearestneighbor_1_size=(56, 56))
        self.relu_174 = nn.ReLU()
        self.conv2d_133 = nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module8_7 = Module8(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=32,
                                 resizenearestneighbor_1_size=(28, 28))
        self.module8_8 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=32,
                                 resizenearestneighbor_1_size=(28, 28))
        self.relu_175 = nn.ReLU()
        self.module6_1 = Module6(conv2d_0_in_channels=16,
                                 conv2d_0_out_channels=64,
                                 module3_0_conv2d_0_in_channels=16,
                                 module3_0_conv2d_0_out_channels=16,
                                 module3_0_conv2d_0_kernel_size=(3, 3),
                                 module3_0_conv2d_0_stride=(2, 2),
                                 module3_0_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_0_conv2d_0_pad_mode="pad")
        self.conv2d_137 = nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module8_9 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=64,
                                 resizenearestneighbor_1_size=(14, 14))
        self.relu_176 = nn.ReLU()
        self.module10_1 = Module10(conv2d_0_in_channels=16,
                                   conv2d_0_out_channels=128,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(2, 2),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module3_0_conv2d_0_in_channels=16,
                                   module3_0_conv2d_0_out_channels=16,
                                   module3_0_conv2d_0_kernel_size=(3, 3),
                                   module3_0_conv2d_0_stride=(2, 2),
                                   module3_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_0_conv2d_0_pad_mode="pad",
                                   module3_1_conv2d_0_in_channels=16,
                                   module3_1_conv2d_0_out_channels=16,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(2, 2),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.module6_2 = Module6(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=128,
                                 module3_0_conv2d_0_in_channels=32,
                                 module3_0_conv2d_0_out_channels=32,
                                 module3_0_conv2d_0_kernel_size=(3, 3),
                                 module3_0_conv2d_0_stride=(2, 2),
                                 module3_0_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_0_conv2d_0_pad_mode="pad")
        self.conv2d_149 = nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_184 = nn.ReLU()
        self.module10_2 = Module10(conv2d_0_in_channels=32,
                                   conv2d_0_out_channels=128,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module3_0_conv2d_0_in_channels=16,
                                   module3_0_conv2d_0_out_channels=32,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=32,
                                   module3_1_conv2d_0_out_channels=32,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_179 = nn.Conv2d(in_channels=16,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_206 = nn.ReLU()
        self.module10_3 = Module10(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module3_0_conv2d_0_in_channels=32,
                                   module3_0_conv2d_0_out_channels=64,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=64,
                                   module3_1_conv2d_0_out_channels=64,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_181 = nn.Conv2d(in_channels=32,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_207 = nn.ReLU()
        self.module3_5 = Module3(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module10_4 = Module10(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module3_0_conv2d_0_in_channels=64,
                                   module3_0_conv2d_0_out_channels=128,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=128,
                                   module3_1_conv2d_0_out_channels=128,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_183 = nn.Conv2d(in_channels=64,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_208 = nn.ReLU()
        self.module3_6 = Module3(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module10_5 = Module10(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module3_0_conv2d_0_in_channels=128,
                                   module3_0_conv2d_0_out_channels=256,
                                   module3_0_conv2d_0_kernel_size=(1, 1),
                                   module3_0_conv2d_0_stride=(1, 1),
                                   module3_0_conv2d_0_padding=0,
                                   module3_0_conv2d_0_pad_mode="valid",
                                   module3_1_conv2d_0_in_channels=256,
                                   module3_1_conv2d_0_out_channels=256,
                                   module3_1_conv2d_0_kernel_size=(3, 3),
                                   module3_1_conv2d_0_stride=(1, 1),
                                   module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_189 = nn.Conv2d(in_channels=128,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_211 = nn.ReLU()
        self.module3_7 = Module3(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=1024,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module3_8 = Module3(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=2048,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.avgpool2d_222 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_223 = nn.Flatten()
        self.dense_224 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module3_0_opt = self.module3_0(opt_relu_1)
        module10_0_opt = self.module10_0(module3_0_opt)
        opt_conv2d_5 = self.conv2d_5(module3_0_opt)
        opt_add_10 = P.Add()(module10_0_opt, opt_conv2d_5)
        opt_relu_11 = self.relu_11(opt_add_10)
        module3_1_opt = self.module3_1(opt_relu_11)
        module3_2_opt = self.module3_2(opt_relu_11)
        module1_0_opt = self.module1_0(module3_1_opt)
        module1_1_opt = self.module1_1(module3_2_opt)
        module8_0_opt = self.module8_0(module1_1_opt)
        opt_add_41 = P.Add()(module1_0_opt, module8_0_opt)
        opt_relu_44 = self.relu_44(opt_add_41)
        opt_conv2d_36 = self.conv2d_36(module1_0_opt)
        opt_add_38 = P.Add()(opt_conv2d_36, module1_1_opt)
        opt_relu_40 = self.relu_40(opt_add_38)
        module3_3_opt = self.module3_3(opt_relu_40)
        module1_2_opt = self.module1_2(opt_relu_44)
        module1_3_opt = self.module1_3(opt_relu_40)
        module1_4_opt = self.module1_4(module3_3_opt)
        module8_1_opt = self.module8_1(module1_3_opt)
        opt_add_82 = P.Add()(module1_2_opt, module8_1_opt)
        module8_2_opt = self.module8_2(module1_4_opt)
        opt_add_88 = P.Add()(opt_add_82, module8_2_opt)
        opt_relu_91 = self.relu_91(opt_add_88)
        opt_conv2d_78 = self.conv2d_78(module1_2_opt)
        opt_add_83 = P.Add()(opt_conv2d_78, module1_3_opt)
        module8_3_opt = self.module8_3(module1_4_opt)
        opt_add_89 = P.Add()(opt_add_83, module8_3_opt)
        opt_relu_92 = self.relu_92(opt_add_89)
        module6_0_opt = self.module6_0(module1_2_opt)
        opt_conv2d_74 = self.conv2d_74(module1_3_opt)
        opt_add_90 = P.Add()(module6_0_opt, opt_conv2d_74)
        opt_add_93 = P.Add()(opt_add_90, module1_4_opt)
        opt_relu_96 = self.relu_96(opt_add_93)
        module3_4_opt = self.module3_4(opt_relu_96)
        module1_5_opt = self.module1_5(opt_relu_91)
        module1_6_opt = self.module1_6(opt_relu_92)
        module1_7_opt = self.module1_7(opt_relu_96)
        module1_8_opt = self.module1_8(module3_4_opt)
        module8_4_opt = self.module8_4(module1_6_opt)
        opt_add_152 = P.Add()(module1_5_opt, module8_4_opt)
        module8_5_opt = self.module8_5(module1_7_opt)
        opt_add_162 = P.Add()(opt_add_152, module8_5_opt)
        module8_6_opt = self.module8_6(module1_8_opt)
        opt_add_170 = P.Add()(opt_add_162, module8_6_opt)
        opt_relu_174 = self.relu_174(opt_add_170)
        opt_conv2d_133 = self.conv2d_133(module1_5_opt)
        opt_add_141 = P.Add()(opt_conv2d_133, module1_6_opt)
        module8_7_opt = self.module8_7(module1_7_opt)
        opt_add_163 = P.Add()(opt_add_141, module8_7_opt)
        module8_8_opt = self.module8_8(module1_8_opt)
        opt_add_171 = P.Add()(opt_add_163, module8_8_opt)
        opt_relu_175 = self.relu_175(opt_add_171)
        module6_1_opt = self.module6_1(module1_5_opt)
        opt_conv2d_137 = self.conv2d_137(module1_6_opt)
        opt_add_157 = P.Add()(module6_1_opt, opt_conv2d_137)
        opt_add_164 = P.Add()(opt_add_157, module1_7_opt)
        module8_9_opt = self.module8_9(module1_8_opt)
        opt_add_172 = P.Add()(opt_add_164, module8_9_opt)
        opt_relu_176 = self.relu_176(opt_add_172)
        module10_1_opt = self.module10_1(module1_5_opt)
        module6_2_opt = self.module6_2(module1_6_opt)
        opt_add_169 = P.Add()(module10_1_opt, module6_2_opt)
        opt_conv2d_149 = self.conv2d_149(module1_7_opt)
        opt_add_173 = P.Add()(opt_add_169, opt_conv2d_149)
        opt_add_177 = P.Add()(opt_add_173, module1_8_opt)
        opt_relu_184 = self.relu_184(opt_add_177)
        module10_2_opt = self.module10_2(opt_relu_174)
        opt_conv2d_179 = self.conv2d_179(opt_relu_174)
        opt_add_202 = P.Add()(module10_2_opt, opt_conv2d_179)
        opt_relu_206 = self.relu_206(opt_add_202)
        module10_3_opt = self.module10_3(opt_relu_175)
        opt_conv2d_181 = self.conv2d_181(opt_relu_175)
        opt_add_203 = P.Add()(module10_3_opt, opt_conv2d_181)
        opt_relu_207 = self.relu_207(opt_add_203)
        module3_5_opt = self.module3_5(opt_relu_206)
        opt_add_213 = P.Add()(opt_relu_207, module3_5_opt)
        module10_4_opt = self.module10_4(opt_relu_176)
        opt_conv2d_183 = self.conv2d_183(opt_relu_176)
        opt_add_204 = P.Add()(module10_4_opt, opt_conv2d_183)
        opt_relu_208 = self.relu_208(opt_add_204)
        module3_6_opt = self.module3_6(opt_add_213)
        opt_add_216 = P.Add()(opt_relu_208, module3_6_opt)
        module10_5_opt = self.module10_5(opt_relu_184)
        opt_conv2d_189 = self.conv2d_189(opt_relu_184)
        opt_add_209 = P.Add()(module10_5_opt, opt_conv2d_189)
        opt_relu_211 = self.relu_211(opt_add_209)
        module3_7_opt = self.module3_7(opt_add_216)
        opt_add_219 = P.Add()(opt_relu_211, module3_7_opt)
        module3_8_opt = self.module3_8(opt_add_219)
        opt_avgpool2d_222 = self.avgpool2d_222(module3_8_opt)
        opt_flatten_223 = self.flatten_223(opt_avgpool2d_222)
        opt_dense_224 = self.dense_224(opt_flatten_223)
        return opt_dense_224
