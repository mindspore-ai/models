# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""prepare complex for predict"""
import os
import ast
import numpy as np
import pandas as pd
import h5py
import pybel
from src.data import Featurizer

def get_pocket(configs, num_pockets, featurizer, num_ligands):
    if num_pockets > 1:
        for pocket_file in configs.pocket:
            if configs.verbose:
                print('reading %s' % pocket_file)
            try:
                pocket = next(pybel.readfile(configs.pocket_format, pocket_file))
            except:
                raise IOError('Cannot read %s file' % pocket_file)

            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            yield (pocket_coords, pocket_features)

    else:
        pocket_file = configs.pocket[0]
        try:
            pocket = next(pybel.readfile(configs.pocket_format, pocket_file))
        except:
            raise IOError('Cannot read %s file' % pocket_file)
        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
        for _ in range(num_ligands):
            yield (pocket_coords, pocket_features)

def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path

def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path

def prepare(configs):
    num_pockets = len(configs.pocket)
    num_ligands = len(configs.ligand)
    featurizer = Featurizer()
    if num_pockets not in (1, num_ligands):
        raise IOError('%s pockets specified for %s ligands. You must either provide '
                      'a single pocket or a separate pocket for each ligand' % (num_pockets, num_ligands))
    if configs.verbose:
        print('%s ligands and %s pockets to prepare:' % (num_ligands, num_pockets))
        if num_pockets == 1:
            print(' pocket: %s' % configs.pocket[0])
            for ligand_file in configs.ligand:
                print(' ligand: %s' % ligand_file)
        else:
            for ligand_file, pocket_file in zip(configs.ligand, configs.pocket):
                print(' ligand: %s, pocket: %s' % (ligand_file, pocket_file))
        print('\n\n')

    if configs.affinities:
        affinities = pd.read_csv(configs.affinities)
        if 'affinity' not in affinities.columns:
            raise ValueError('There is no `affinity` column in the table')
        if 'name' not in affinities.columns:
            raise ValueError('There is no `name` column in the table')
        affinities = affinities.set_index('name')['affinity']
    else:
        affinities = None

    with h5py.File(configs.output, configs.mode) as f:
        pocket_generator = get_pocket(configs, num_pockets, featurizer=featurizer, num_ligands=num_ligands)
        for ligand_file in configs.ligand:
            # use filename without extension as dataset name
            name = os.path.splitext(os.path.split(ligand_file)[1])[0]
            if configs.verbose:
                print('reading %s' % ligand_file)
            try:
                ligand = next(pybel.readfile(configs.ligand_format, ligand_file))
            except:
                raise IOError('Cannot read %s file' % ligand_file)

            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            pocket_coords, pocket_features = next(pocket_generator)

            centroid = ligand_coords.mean(axis=0)
            ligand_coords -= centroid
            pocket_coords -= centroid

            data = np.concatenate(
                (np.concatenate((ligand_coords, pocket_coords)),
                 np.concatenate((ligand_features, pocket_features))),
                axis=1,
            )

            dataset = f.create_dataset(name, data=data, shape=data.shape,
                                       dtype='float32', compression='lzf')
            if affinities is not None:
                dataset.attrs['affinity'] = affinities.loc[name]
    if configs.verbose:
        print('\n\ncreated %s with %s structures' % (configs.output, num_ligands))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare molecular data for the network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ligand', '-l', required=True, type=input_file, nargs='+',
                        help='files with ligands\' structures')
    parser.add_argument('--pocket', '-p', required=True, type=input_file, nargs='+',
                        help='files with pockets\' structures')
    parser.add_argument('--ligand_format', type=str, default='mol2',
                        help='file format for the ligand,'
                             ' must be supported by openbabel')

    parser.add_argument('--pocket_format', type=str, default='mol2',
                        help='file format for the pocket,'
                             ' must be supported by openbabel')
    parser.add_argument('--output', '-o', default='./complexes.hdf',
                        type=output_file,
                        help='name for the file with the prepared structures')
    parser.add_argument('--mode', '-m', default='w',
                        type=str, choices=['r+', 'w', 'w-', 'x', 'a'],
                        help='mode for the output file (see h5py documentation)')
    parser.add_argument('--affinities', '-a', default=None, type=input_file,
                        help='CSV table with affinity values.'
                             ' It must contain two columns: `name` which must be'
                             ' equal to ligand\'s file name without extension,'
                             ' and `affinity` which must contain floats')
    parser.add_argument('--verbose', '-v', default=True, type=ast.literal_eval,
                        help='whether to print messages')

    args = parser.parse_args()

    prepare(configs=args)
