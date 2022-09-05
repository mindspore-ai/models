from mindspore import nn
from mindspore import ops

from src.models.ViT_helper import to_2tuple


class PatchEmbed(nn.Cell):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True,
                              pad_mode='valid')

    def construct(self, x):
        x = self.proj(x)

        return x


class Decoder(nn.Cell):
    def __init__(self, train=True):
        super().__init__()
        self.block1 = nn.SequentialCell(
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(512, 256, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU()
        )
        if train:
            self.up1 = ops.ResizeNearestNeighbor((64, 64))
        else:
            self.up1 = ops.ResizeNearestNeighbor((128, 128))
        self.block2 = nn.SequentialCell(
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(256, 256, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU(),
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(256, 256, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU(),
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(256, 256, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU(),
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(256, 128, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU()
        )
        if train:
            self.up2 = ops.ResizeNearestNeighbor((128, 128))
        else:
            self.up2 = ops.ResizeNearestNeighbor((256, 256))
        self.block3 = nn.SequentialCell(
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(128, 128, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU(),
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(128, 64, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU()
        )
        if train:
            self.up3 = ops.ResizeNearestNeighbor((256, 256))
        else:
            self.up3 = ops.ResizeNearestNeighbor((512, 512))
        self.block4 = nn.SequentialCell(
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(64, 64, (3, 3), has_bias=True, pad_mode='valid'),
            nn.ReLU(),
            nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
            nn.Conv2d(64, 3, (3, 3), has_bias=True, pad_mode='valid'),
        )

    def construct(self, x):
        x = self.block1(x)
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.block4(x)
        return x


vgg = nn.SequentialCell(
    nn.Conv2d(3, 3, (1, 1), has_bias=True, pad_mode='valid'),
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(3, 64, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu1-1
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(64, 64, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(64, 128, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu2-1
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(128, 128, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(128, 256, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu3-1
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(256, 256, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu3-2
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(256, 256, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu3-3
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(256, 256, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(256, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu4-1
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu4-2
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu4-3
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu5-1
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu5-2
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu5-3
    nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode='REFLECT'),
    nn.Conv2d(512, 512, (3, 3), has_bias=True, pad_mode='valid'),
    nn.ReLU(),  # relu5-4
)


class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList(
            nn.Dense(n, k, weight_init='uniform') for n, k in zip([input_dim] + h, h + [output_dim]))

    def construct(self, *inputs, **kwargs):
        relu = ops.ReLU()
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class StyTrans(nn.Cell):
    def __init__(self, decoder, embedding, transformer):
        super(StyTrans, self).__init__()
        self.transformer = transformer
        self.decode = decoder
        self.embedding = embedding

    def construct(self, content_input, style_input):
        ### Linear projection
        style = self.embedding(style_input)
        content = self.embedding(content_input)

        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask, content, pos_c, pos_s)
        Ics = self.decode(hs)

        return Ics, content, style
