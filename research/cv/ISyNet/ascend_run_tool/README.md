Author: Vladimir Korviakov 00506636

This is a MindStudio project of the infernce tool of the OM model on the Ascend device.

This program runs the model on an input data and produces the output prediction and computes the inference time (model runtime and e2e, including reading input file and saving output file).

Input data should be organized as following:

/base/path/to/data/dir_name_1/inp.bin
/base/path/to/data/dir_name_2/inp.bin
/base/path/to/data/dir_name_3/inp.bin
...
/base/path/to/data/dir_name_NNN/inp.bin

Then the program should be executed with the following parameters:

./main /path/to/acl.json /base/path/to/data /path/to/model/model.om /base/path/to/results

The results will be organized as following:

/base/path/to/results/dir_name_1/output_0_.bin
/base/path/to/results/dir_name_2/output_0_.bin
/base/path/to/results/dir_name_3/output_0_.bin
...
/base/path/to/results/dir_name_NNN/output_0_.bin
