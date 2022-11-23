MindSpore Golden Stick provides UniPruning algorithm for
ResNet18/34/50/101/152 and other ResNet-like and VGG-like models.
UniPruning is provided by Intelligent Systems and Data Science Technology center of Huawei Moscow Research Center.
UniPruning is a soft-pruning algorithm.
It measures relative importance of channels in a hardware-friendly manner.
Particularly, it groups channels in groups of size G,
where each channel's importance is measured as a L2 norm of its weights
multiplied by consecutive BatchNorm's gamma,
channels are sorted by their importance and consecutive channels are groupped.
The absolute group importance is given as the median of channel importances.
The relative importance criteria of a channel group G group of a layer L
is measured as the highest median of the layer L divided by the median of group G.
The higher the relative importance of a group,
the less a group contributes to the layer output.

During training UniPruning algorithm every N epochs searches for channel groups
with the highest relative criteria network-wise and zeroes channels (UniZeroing)
in that groups until reaching target sparsity,
which is gives as a % of parameters to prune.
To obtain pruned model after training,
pruning mask and zeroed weights from the last UniPruning step
are used to physically prune the network.

The hyper-parameters of UniPruning are:

* Target sparsity - compression rate of the model.
* Frequency - number of fine-tuning epochs between each UniZeroing step.
* Pruning step - size of groups, e.g. 32 means that filters are groupped by 32.
* Filter lower threshold - minimal number of channels that stays in layer after pruning.

