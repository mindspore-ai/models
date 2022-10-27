from PIL import Image
import mindspore
from mindspore import ops


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.shape
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(axis=2) + eps
    sqrt = ops.Sqrt()
    feat_var = sqrt(feat_var)
    feat_std = feat_var.view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(axis=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std1(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    feat_var = feat.var(axis=0) + eps
    sqrt = ops.Sqrt()
    feat_std = sqrt(feat_var)
    feat_mean = feat.mean(axis=0)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


def normal_style(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std1(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


def Msave_image(images, name):
    mul = ops.Mul()
    add = ops.Add()
    cast = ops.Cast()
    transpose = ops.Transpose()
    B, _, H, W = images.shape
    newimg = Image.new('RGB', (B * H, W), (255, 0, 0))
    i = 0

    for img in images:
        tmp1 = mul(img, 255)
        tmp2 = add(tmp1, 0.5)
        tmp3 = ops.clip_by_value(tmp2, 0, 255)
        tmp4 = transpose(tmp3, (1, 2, 0))
        tmp5 = cast(tmp4, mindspore.uint8)
        tmp6 = tmp5.asnumpy()
        im = Image.fromarray(tmp6)
        newimg.paste(im, (i * W, 0))
        i += 1

    newimg.save(name)
