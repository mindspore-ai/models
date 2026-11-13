

import time
import os
import argparse
import glob
import random
import numpy as np
import mindspore as ms
import mindspore
import mindspore.nn as nn
from mindspore import Model, ops
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.ops import GradOperation
import mindspore.context as context

from utils import load_data, accuracy
from models1 import MultiHeadGATLayer


# Training settings
parser = argparse.ArgumentParser(description='GAT')
parser.add_argument('--path', type=str, default="./cora/", help='path of the cora dataset directory.')
parser.add_argument('--device', type=str, default="GPU", help='GPU training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()

device_id= 0

context.set_context(device_target=args.device, mode=context.GRAPH_MODE, device_id=device_id)



random.seed(args.seed)
np.random.seed(args.seed)


dataset = "cora"

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.path, dataset)

model = MultiHeadGATLayer(input_feature_size = 1433, output_size=args.hidden, nclass = 7, dropout= args.dropout, alpha = args.alpha, nheads =args.nb_heads)
loss_fn = nn.NLLLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)



# Define forward function
def forward_fn(features, adj, labels):
    logits = model(features, adj)
    loss = loss_fn(logits[idx_train], labels[idx_train])
   # acc_train = accuracy(logits[idx_train], labels[idx_train])
    
    return loss, logits

# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# Define function of one-step training
def train_step(features, adj, labels):
    (loss, logits), grads = grad_fn(features, adj, labels)
    optimizer(grads)
    acc_train = accuracy(logits[idx_train], labels[idx_train])
    return loss, acc_train, logits

def train_loop(model, features, adj, labels):
    t = time.time()
    model.set_train()
    
    loss_train ,acc_train, output = train_step(features, adj, labels)
    if not args.fastmode:
        model.set_train(False)
        output = model(features,adj)
    loss_val = loss_fn(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
   
    print('Epoch: {:d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.asnumpy()),
          'acc_train: {:.4f}%'.format(100*acc_train.asnumpy()),
          'loss_val: {:.4f}'.format(loss_val.asnumpy()),
          'acc_val: {:.4f}%'.format(100*acc_val.asnumpy()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.asnumpy()
    



def test_loop(model, features, adj, labels, loss_fn):
    model.set_train(False)
    pred = model(features, adj)
    loss_test = loss_fn(pred[idx_test], labels[idx_test])
    acc_test = accuracy(pred[idx_test], labels[idx_test])
 
    print('Testing Results\n',
          'loss_test: {:.4f}'.format(loss_test.asnumpy()),
          'acc_test: {:.4f}%'.format(100*acc_test.asnumpy()))

    


#Training
t_total = time.time()
print(args)
for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(model, features, adj, labels)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

#Testing

test_loop(model, features, adj, labels, loss_fn)







