import argparse
from pathlib import Path
import os
import time
from src.models.WithLossCell import StyTRWithLossCell
from src.models import StyTR, transformer
from src.utils.function import Msave_image
from mindspore import nn, context, ops, ParallelMode
from mindspore import dataset as ds
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.communication import init, get_rank, get_group_size
from mindspore.dataset.vision import c_transforms, py_transforms
from mindspore.dataset.transforms.c_transforms import Compose
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True



context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


def train_transform():
    transform_list = [
        c_transforms.Resize(size=(512, 512)),
        c_transforms.RandomCrop(256),
        py_transforms.ToTensor()
    ]
    return Compose(transform_list)


def get_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', default='../COCO2014/train2014', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', default='../../wiki/wikiart/train', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--auxiliary_dir', type=str, default='auxiliary')
    # training options
    parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default: false.")
    parser.add_argument('--device_id', type=int, default=1, help='device id')
    parser.add_argument('--group_size', type=int, default=1, help='group size')
    parser.add_argument('--rank', type=int, default=0, help='rank id')
    parser.add_argument('--save_dir', default='./save_model',
                        help='Directory to save the model')
    parser.add_argument('--save_picture', default='./picture', help='Directory to save the picture')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=2 * 160000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    args = parser.parse_args()
    args.vgg = args.auxiliary_dir + '/vgg_norm.ckpt'
    args.decoder = args.auxiliary_dir + '/m_decoder.ckpt'
    args.embedding = args.auxiliary_dir + '/m_embedding.ckpt'
    args.trans = args.auxiliary_dir + '/m_transformer.ckpt'
    return args


class FlatFolderDataset():
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root + "/" + file_name + "/" + file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        print(self.root, len(self.paths))

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(args, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count * 1.0 / (8 / args.batch_size) - 1e4))
    return lr


def warmup_learning_rate(args, iteration_count):
    """Imitating the original implementation"""

    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count * 1.0 / (8 / args.batch_size))
    return lr


def dynamic_lr(args):
    """dynamic learning rate generator"""
    total_steps = int(args.max_iter)
    warmup_steps = 10000
    lr = []
    for i in range(total_steps):
        if i / (8 / args.batch_size) < warmup_steps:
            curr_lr = warmup_learning_rate(args, i)
        elif i / (8 / args.batch_size) > warmup_steps:
            curr_lr = adjust_learning_rate(args, i)
        lr.append(curr_lr)
    return lr


def train():
    args = get_args()
    content_tf = train_transform()
    style_tf = train_transform()
    dataset_c = FlatFolderDataset(args.content_dir, content_tf)
    dataset_s = FlatFolderDataset(args.style_dir, style_tf)

    if args.run_distribute:
        init("nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, device_num=get_group_size(),
                                          gradients_mean=True)
        args.rank = get_rank()
        args.group_size = get_group_size()

        DS_c = ds.GeneratorDataset(dataset_c, column_names=["content"], num_parallel_workers=args.n_threads,
                                   shuffle=True, num_shards=args.group_size, shard_id=args.rank)
        DS_s = ds.GeneratorDataset(dataset_s, column_names=["style"], num_parallel_workers=args.n_threads,
                                   shuffle=True, num_shards=args.group_size, shard_id=args.rank)
    else:
        context.set_context(device_id=int(args.device_id))
        DS_c = ds.GeneratorDataset(dataset_c, column_names=["content"], num_parallel_workers=args.n_threads,
                                   shuffle=True, num_shards=args.group_size, shard_id=args.rank)
        DS_s = ds.GeneratorDataset(dataset_s, column_names=["style"], num_parallel_workers=args.n_threads,
                                   shuffle=True, num_shards=args.group_size, shard_id=args.rank)

    DS_c = DS_c.batch(args.batch_size, drop_remainder=True)
    DS_s = DS_s.batch(args.batch_size, drop_remainder=True)
    train_data_loader_c = DS_c.create_dict_iterator()
    train_data_loader_s = DS_s.create_dict_iterator()

    if args.rank == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.rank == 0 and not os.path.exists(args.save_picture):
        os.makedirs(args.save_picture)

    vgg = StyTR.vgg
    vgg_parm_dict = load_checkpoint(args.vgg)
    load_param_into_net(vgg, vgg_parm_dict)
    vgg = vgg[:44]

    decoder = StyTR.Decoder(True)
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder_parm_dict = load_checkpoint(args.decoder)
    load_param_into_net(decoder, decoder_parm_dict)
    Trans_parm_dict = load_checkpoint(args.trans)
    load_param_into_net(Trans, Trans_parm_dict)
    embedding_parm_dict = load_checkpoint(args.embedding)
    load_param_into_net(embedding, embedding_parm_dict)

    lr = dynamic_lr(args)
    optim = nn.Adam([{'params': Trans.trainable_params()},
                     {'params': decoder.trainable_params()},
                     {'params': embedding.trainable_params()},
                     ], learning_rate=lr)
    stytran = StyTR.StyTrans(decoder, embedding, Trans)
    net_with_loss = StyTRWithLossCell(vgg, stytran, args)
    manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    StyTR_model = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=optim, scale_sense=manager)
    StyTR_model.set_train()

    j = 0
    for epoch in range(16):
        for i, (dcontent, dstyle) in enumerate(zip(train_data_loader_c, train_data_loader_s)):
            step_time = time.time()

            content = dcontent['content']
            style = dstyle['style']
            loss = StyTR_model(content, style)
            step_end_tme = time.time()
            if args.rank == 0:
                print('epoch:', epoch, 'step:', j, 'batch:', i, 'loss:', loss[0].sum())
                print('step time is', step_end_tme - step_time)

            if j % 100 == 0 and args.rank == 0:
                StyTR_model.network.stytran.set_train(False)
                StyTR_model.network.stytran.set_grad(False)
                output, _, _ = StyTR_model.network.stytran(content, style)
                cat_op = ops.Concat()
                expand_dims = ops.ExpandDims()
                pic_c = expand_dims(content[0], 0)
                pic_s = expand_dims(style[0], 0)
                pic_o = expand_dims(output[0], 0)
                pic = cat_op((pic_s, pic_o))
                pic2 = cat_op((pic_c, pic))
                Msave_image(pic2, args.save_picture + '/' + str(j) + '.jpg')
                net_with_loss.stytran.set_train(True)
                net_with_loss.stytran.set_grad(True)
                print('epoch:', epoch, 'step:', j, 'batch:', i, 'loss:', loss[0].sum(), pic_o.max(), pic_o.min())

            if ((j + 1) % args.save_model_interval == 0 or (j + 1) == args.max_iter) and args.rank == 0:
                save_checkpoint(decoder, args.save_dir + '/decoder_' + str(j + 1) + '.ckpt')
                save_checkpoint(Trans, args.save_dir + '/transformer_' + str(j + 1) + '.ckpt')
                save_checkpoint(embedding, args.save_dir + '/embedding_' + str(j + 1) + '.ckpt')

            j += 1


if __name__ == '__main__':
    start_time = time.time()
    train()
    end_time = time.time()
    print('total time is ', end_time - start_time)
