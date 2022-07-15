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
"""process raw pdbbindv2016 data"""
import os
import argparse
import warnings
import h5py
import pybel
import numpy as np
import pandas as pd
from src.data import Featurizer


def extractFeature(path, affinity_data, dataset_path):
    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

    with h5py.File('%s/core2013.hdf' % path, 'w') as g:
        j = 0
        for dataset_name, data in affinity_data.groupby('set'):
            print(dataset_name, 'set')
            i = 0
            ds_path = dataset_path[dataset_name]
            print(ds_path)
            with h5py.File('%s/%s.hdf' % (path, dataset_name), 'w') as f:
                for _, row in data.iterrows():
                    name = row['pdbid']
                    affinity = row['Kd_Ki']
                    ligand = next(pybel.readfile('mol2', '%s/%s/%s/%s_ligand.mol2' % (path, ds_path, name, name)))
                    # do not add the hydrogens! they are in the structure and it would reset the charges
                    try:
                        pocket = next(pybel.readfile('mol2', '%s/%s/%s/%s_pocket.mol2' % (path, ds_path, name, name)))
                        # do not add the hydrogens! they were already added in chimera and it would reset the charges
                    except ValueError:
                        warnings.warn('no pocket for %s, %s (%s set)' % (dataset_name, name, dataset_name))
                        continue

                    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
                    assert (ligand_features[:, charge_idx] != 0).any()
                    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)

                    centroid = ligand_coords.mean(axis=0)
                    ligand_coords -= centroid
                    pocket_coords -= centroid

                    data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                                           np.concatenate((ligand_features, pocket_features))), axis=1)

                    if row['include']:
                        dataset = f.create_dataset(name, data=data, shape=data.shape,
                                                   dtype='float32', compression='lzf')
                        dataset.attrs['affinity'] = affinity
                        i += 1
                    else:
                        dataset = g.create_dataset(name, data=data, shape=data.shape,
                                                   dtype='float32', compression='lzf')
                        dataset.attrs['affinity'] = affinity
                        j += 1

            print('prepared', i, 'complexes')
        print('excluded', j, 'complexes')


def transpdb2mol2(path, dataset_name):
    for dataset in dataset_name.values():
        data_path = os.path.join(path, dataset)
        for die_path, _, pdbfile in os.walk(data_path):
            for pfile in pdbfile:
                if "_pocket.pdb" in pfile:
                    p_real_file = os.path.join(die_path, pfile)
                    molfile = p_real_file.replace(".pdb", ".mol2")
                    command = "obabel -i pdb %s -o mol2 -O %s" % (p_real_file, molfile)
                    os.system(command)
    print("Finish trans pdb to mol2 format.")

def ParseandClean(paths):
    files = os.path.join(paths, 'PDBbind_2016_plain_text_index/index/INDEX_general_PL_data.2016')
    if os.path.exists('./affinity_data.csv'):
        os.remove('./affinity_data.csv')
    # Save binding affinities to csv file
    result = pd.DataFrame(columns=('pdbid', 'Kd_Ki'))
    for line in open(files):
        line = line.rstrip()
        if line.startswith('#') or line == '':
            continue
        it = line.split(maxsplit=7)
        pdbid, log_kdki = it[0], it[3]
        result = result.append(
            pd.DataFrame({'pdbid': [pdbid], 'Kd_Ki': [log_kdki]}),
            ignore_index=True)
    result.to_csv('affinity_data.csv', sep=",", index=False)
    affinity_data = pd.read_csv('affinity_data.csv', comment='#')

    # Find affinities without structural data (i.e. with missing directories)
    missing = []
    for misdata in affinity_data['pdbid']:
        gser = os.path.join(paths, f'general-set-except-refined/{misdata}')
        refined_set = os.path.join(paths, f'refined-set/{misdata}')
        if not os.path.exists(gser) and not os.path.exists(refined_set):
            missing.append(misdata)
    missing = set(missing)
    affinity_data = affinity_data[~np.in1d(affinity_data['pdbid'], list(missing))]
    print("Missing length: ", len(missing))
    print(affinity_data['Kd_Ki'].isnull().any())

    # Separate core, refined, and general sets
    core_file = os.path.join(paths, 'PDBbind_2016_plain_text_index/index/INDEX_core_data.2016')
    core_set = []
    for c_line in open(core_file):
        c_line = c_line.rstrip()
        if c_line.startswith('#') or c_line == '':
            continue
        c_it = c_line.split(maxsplit=7)
        core_set.append(c_it[0])
    core_set = set(core_set)
    print('Core Set length: ', len(core_set))
    refined_file = os.path.join(paths, 'PDBbind_2016_plain_text_index/index/INDEX_refined_data.2016')
    refined_set = []
    for rf_line in open(refined_file):
        rf_line = rf_line.rstrip()
        if rf_line.startswith('#') or rf_line == '':
            continue
        rf_it = rf_line.split(maxsplit=7)
        refined_set.append(rf_it[0])
    refined_set = set(refined_set)
    general_set = set(affinity_data['pdbid'])

    assert core_set & refined_set == core_set
    assert refined_set & general_set == refined_set

    print("Refined Set Length: ", len(refined_set))
    print("General Set Length: ", len(general_set))
    #exclude v2013 core set -- it will be used as another test set
    core2013_file = os.path.join(paths, 'core_pdbbind2013.ids')
    core2013 = []
    for c2_line in open(core2013_file):
        c2_it = c2_line.rstrip()
        core2013.append(c2_it)
    core2013 = set(core2013)
    print("Core2013 length: ", len(core2013))
    print(affinity_data.head())
    print(len(core2013 & (general_set - core_set)))
    affinity_data['include'] = True
    affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core2013 & (general_set - core_set))), 'include'] = False

    affinity_data.loc[np.in1d(affinity_data['pdbid'], list(general_set)), 'set'] = 'general'
    affinity_data.loc[np.in1d(affinity_data['pdbid'], list(refined_set)), 'set'] = 'refined'
    affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core_set)), 'set'] = 'core'

    print(affinity_data.head())
    print(affinity_data[affinity_data['include']].groupby('set').apply(len).loc[['general', 'refined', 'core']])

    if os.path.exists('./affinity_data_cleaned.csv'):
        os.remove('./affinity_data_cleaned.csv')
    affinity_data[['pdbid']].to_csv('pdb.ids', header=False, index=False)
    affinity_data[['pdbid', 'Kd_Ki', 'set']].to_csv('affinity_data_cleaned.csv', index=False)
    #Parse Molecules
    dataset_path = {'general': 'general-set-except-refined', 'refined': 'refined-set', 'core': 'refined-set'}
    transpdb2mol2(paths, dataset_path)
    extractFeature(path=paths, affinity_data=affinity_data, dataset_path=dataset_path)
    print("Finish process data.")

    with h5py.File('%s/core.hdf' % paths, 'r') as f, \
            h5py.File('%s/core2013.hdf' % paths, 'r+') as g:
        for name in f:
            if name in core2013:
                dataset = g.create_dataset(name, data=f[name])
                dataset.attrs['affinity'] = f[name].attrs['affinity']

    print("Finish All..........")

def Extrct2013ids(in_paths):
    """Extract pdbbind2013 index"""
    filepath = os.path.join(in_paths, './v2013-core')
    file_idx = os.listdir(filepath)
    for items in file_idx:
        with open(os.path.join(in_paths, 'core_pdbbind2013.ids'), 'a') as f:
            f.write(items+'\n')
    print("extract 2013 index done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess pdbbind data")
    parser.add_argument('--data_path', type=str, required=True, default='',
                        help='Dataset process.')
    args = parser.parse_args()
    data_paths = args.data_path
    if not os.path.exists(os.path.join(data_paths, 'PDBbind_2016_plain_text_index/index/INDEX_general_PL_data.2016')):
        raise IOError("INDEX_general_PL_data.2016 file doesn't exit!")
    if not os.path.exists(os.path.join(data_paths, 'PDBbind_2016_plain_text_index/index/INDEX_core_data.2016')):
        raise IOError("INDEX_core_data.2016 file doesn't exit!")
    if not os.path.exists(os.path.join(data_paths, 'PDBbind_2016_plain_text_index/index/INDEX_refined_data.2016')):
        raise IOError("INDEX_refined_data.2016 file doesn't exit!")
    if os.path.exists(os.path.join(data_paths, 'core_pdbbind2013.ids')):
        print("Remove Exist core_pdbbind2013.ids file.")
        os.remove(os.path.join(data_paths, 'core_pdbbind2013.ids'))
    Extrct2013ids(data_paths)

    ParseandClean(data_paths)
