# Dataset preparation

AISHELL dataset is preprocessed using Kaldi library. We suggest using docker
and original repositories for preprocessing and then convert binaries produced by Kaldi scripts to
Python pickle objects.

0. Download and unzip dataset in `prepare_aishell_data`. Note that inner archives must be unzipped without the creation of a root dir. We suggest to use midnight commander.
1. Move to your `prepare_aishell_data` folder

   ```bash
   cd your/path/to/prepare_aishell_data/
   ```

2. Clone original repository

   ```bash
   git clone https://github.com/kaituoxu/Speech-Transformer
   ```

3. copy `convert_kaldi_bins_to_pickle.py` to `/prepare_aishell_data/Speech-Transformer/src/utils`
4. Run docker with `Kaldi`

   ```bash
   docker run -it -v $PWD:$PWD kaldiasr/kaldi:2020-09 bash
   ```

5. Move to `tools` folder in reference repository

   ```bash
   cd /path/to/your/Speech-Transformer/tools/
   ```

6. Build `Kaldi` tools.

   ```bash
   make KALDI=/opt/kaldi
   ```

7. Move to folder with aishell scripts.

   ```bash
   cd ../egs/aishell/
   ```

8. Specify path to unzipped AISHELL dataset in run.sh. Root dir must be:

    ```shell
    .
    └── unzipped_dataset
        ├── data_aishell
            ├── transcript
            └── wav
    ```

9. Remove or comment all lines for stage 3 and further (from line 133 `if [ -z ${tag} ]; then`)
10. Update apt for installing `bc`

    ```bash
    apt-get update
    ```

11. Install `bc`

    ```bash
    apt-get install bc
    ```

12. Run data preprocessing using `Kaldi`

    ```bash
    bash run.sh
    ```

13. Install `pip`

    ```bash
    apt-get install python3-pip
    ```

14. Install `numpy`

    ```bash
    pip3 install numpy
    ```

15. cd to `/prepare_aishell_data/Speech-Transformer/src/utils`
16. Convert `Kaldi` binaries into python pickle objects.

    ```bash
    python3 convert_kaldi_bins_to_pickle.py --processed-dataset-path /your/path/to/prepare_aishell_data/Speech-Transformer/egs/aishell
    ```

Note that you must specify absolute path!

Processed pickled data will be stored in `/your/path/to/prepare_aishell_data/Speech-Transformer/egs/aishell/pickled_dataset/`
