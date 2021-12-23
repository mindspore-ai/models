#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

# This script downloads the Google Landmarks v2 dataset TRAIN split. To download the dataset
# run the script like in the following example:
#   bash download_gldv2.sh 500 [DATASET_PATH] [NEED_ROMOVE_TAR]
# 
# The script takes the following parameters, in order:
# - number of image files from the TRAIN split to download (maximum 500)
# - path for dataset

if [[ $# -lt 2 || $# -gt 3 ]]
then
    echo "Usage: bash download_gldv2.sh 500 [DATASET_PATH] [NEED_ROMOVE_TAR]
    NEED_ROMOVE_TAR is optional, whether remove tar after extracting the images, choices: 'y' and 'n', default 'n' "
exit 1
fi

need_remove_tar="n"
if [ $# == 3 ]
then
  if [ "$3" == "y" ] || [ "$3" == "n" ];then
      need_remove_tar=$3
  else
    echo "weather need remove tar or not, it's value must be in [y, n]"
    exit 1
  fi
fi

image_files_train=$1 # Number of image files to download from the TRAIN split
dataset_root_folder=$2

split="train"

metadata_url="https://s3.amazonaws.com/google-landmark/metadata"
csv_train=("${metadata_url}/train.csv" "${metadata_url}/train_clean.csv" "${metadata_url}/train_attribution.csv" "${metadata_url}/train_label_to_category.csv")
export csv_train

images_tar_file_base_url="https://s3.amazonaws.com/google-landmark"
images_md5_file_base_url="https://s3.amazonaws.com/google-landmark/md5sum"
num_processes=8

make_folder() {
  # Creates a folder and checks if it exists. Exits if folder creation fails.
  local folder=$1
  if [ -d "${folder}" ]; then
    echo "Folder ${folder} already exists. Skipping folder creation."
  else
    echo "Creating folder ${folder}."
    if mkdir -p ${folder}; then
      echo "Successfully created folder ${folder}."
    else
      echo "Failed to create folder ${folder}. Exiting."
      exit 1
    fi
  fi
}

download_file() {
  # Downloads a file from an URL into a specified folder.
  local file_url=$1
  local folder=$2
  # local file_path="${folder}/`basename ${file_url}`"
  echo "Downloading file ${file_url} to folder ${folder}."
  pushd . > /dev/null
  cd ${folder}
  curl -Os -C - --retry 10 ${file_url}
  popd > /dev/null
}

validate_md5_checksum() {
  # Validate the MD5 checksum of a downloaded file.
  local content_file=$1
  local md5_file=$2
  echo "Checking MD5 checksum of file ${content_file} against ${md5_file}"
  if [[ "${OSTYPE}" == "linux-gnu" ]]; then
    content_md5=`md5sum ${content_file}`
  elif [[ "${OSTYPE}" == "darwin"* ]]; then
    content_md5=`md5 -r "${content_file}"`
  fi
  content_md5=`cut -d' ' -f1<<<"${content_md5}"`
  expected_md5=`cut -d' ' -f1<<<cat "${md5_file}"`
  if [[ "$content_md5" != "" && "$content_md5" = "$expected_md5" ]]; then
    echo "Check ${content_file} passed."
    return 0
  else
    echo "Check failed. MD5 checksums don't match. Exiting."
    return 1
  fi
}

extract_tar_file() {
  # Extracts the content of a tar file to a specified folder.
  local tar_file=$1
  local folder=$2
  echo "Extracting file ${tar_file} to folder ${folder}"
  tar -C ${folder} -xf ${tar_file}
  if [ $need_remove_tar == "y" ]; then
    rm -rf ${tar_file}
  fi
}

download_image_file() {
  # Downloads one image file of a split and untar it.
  local split=$1
  local idx=`printf "%03g" $2`
  local split_folder=$3

  local images_md5_file=md5.images_${idx}.txt
  local images_md5_file_url=${images_md5_file_base_url}/${split}/${images_md5_file}
  local images_md5_file_path=${split_folder}/${images_md5_file}

  download_file "${images_md5_file_url}" "${split_folder}"

  local images_tar_file=images_${idx}.tar
  local images_tar_file_url=${images_tar_file_base_url}/${split}/${images_tar_file}
  local images_tar_file_path=${split_folder}/${images_tar_file}

  download_file "${images_tar_file_url}" "${split_folder}"
  if validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}" ; then
    echo "${images_tar_file_path} error for wrong md5 file"
    download_file "${images_md5_file_url}" "${split_folder}"
    validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}"
  fi
  #extract_tar_file "${images_tar_file_path}" "${split_folder}"
  
}

check_image_file() {
  # Downloads one image file of a split and untar it.
  local split=$1
  local idx=`printf "%03g" $2`
  local split_folder=$3

  local images_md5_file=md5.images_${idx}.txt
  local images_md5_file_url=${images_md5_file_base_url}/${split}/${images_md5_file}
  local images_md5_file_path=${split_folder}/${images_md5_file}
  if  ! [ -f "${images_md5_file_path}" ]; then
    echo "${images_md5_file_path} not found!"
    download_file "${images_md5_file_url}" "${split_folder}"
  else 
    local filesize=`wc -c < "${images_md5_file_path}" `
    echo "md5file size is ${filesize}"
    if [[ "${filesize}" -lt 40 ]]; then
      echo "${images_md5_file_path} not complete"
      download_file "${images_md5_file_url}" "${split_folder}"
    fi
  fi

  local images_tar_file=images_${idx}.tar
  local images_tar_file_url=${images_tar_file_base_url}/${split}/${images_tar_file}
  local images_tar_file_path=${split_folder}/${images_tar_file}
  if ! [ -f "${images_tar_file_path}" ]; then
    echo "${images_tar_file_path} not found!"
    download_file "${images_tar_file_url}" "${split_folder}"
    if validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}" ; then
      echo "${images_tar_file_path} error for wrong md5 file"
      download_file "${images_md5_file_url}" "${split_folder}"
      validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}"
    fi
    
  else
    if ! validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}" ; then
      echo "${images_tar_file_path} not complete "
      download_file "${images_tar_file_url}" "${split_folder}"
      validate_md5_checksum "${images_tar_file_path}" "${images_md5_file_path}"
    fi
  fi
  extract_tar_file "${images_tar_file_path}" "${split_folder}"
}


download_image_files() {
  # Downloads all image files of a split and untars them.
  local split=$1
  local split_folder=$2
  local max_idx=$(expr ${image_files_train} - 1)
  echo "Downloading ${image_files_train} files form the split ${split} in the folder ${split_folder}."
  for i in $(seq 0 ${num_processes} ${max_idx}); do
    local curr_max_idx=$(expr ${i} + ${num_processes} - 1)
    local last_idx=$((${curr_max_idx}>${max_idx}?${max_idx}:${curr_max_idx}))
    for j in $(seq ${i} 1 ${last_idx}); do download_image_file "${split}" "${j}" "${split_folder}" & done
    wait
  done
}

check_image_files() {
  # Downloads all image files of a split and untars them.
  local split=$1
  local split_folder=$2
  local max_idx=$(expr ${image_files_train} - 1)
  echo "Downloading ${image_files_train} files form the split ${split} in the folder ${split_folder}."
  for i in $(seq 0 1 ${max_idx}); do
    local curr_max_idx=$(expr ${i} + 1 - 1)
    local last_idx=$((${curr_max_idx}>${max_idx}?${max_idx}:${curr_max_idx}))
    for j in $(seq ${i} 1 ${last_idx}); do check_image_file "${split}" "${j}" "${split_folder}" & done
    wait
  done
}

download_csv_files() {
  # Downloads all medatada CSV files of a split.
  local split=$1
  local split_folder=$2
  local csv_list="csv_${split}[*]"
  for csv_file in ${!csv_list}; do
    download_file "${csv_file}" "${split_folder}"
  done
}

download_split() {
  # Downloads all artifacts, metadata CSV files and image files of a single split.
  local split=$1
  local split_folder=${dataset_root_folder}/${split}
  make_folder "${split_folder}"
  download_csv_files "${split}" "${split_folder}"
  download_image_files "${split}" "${split_folder}"
  check_image_files "${split}" "${split_folder}"
}

download_all_splits() {
  # Downloads all artifacts, metadata CSV files and image files of all splits.
  make_folder "${dataset_root_folder}"
  download_split "${split}"
}

download_all_splits
python3 src/build_image_dataset.py \
  --train_csv_path=${dataset_root_folder}/train/train.csv \
  --train_clean_csv_path=${dataset_root_folder}/train/train_clean.csv \
  --train_directory=${dataset_root_folder}/train/*/*/*/ \
  --output_directory=${dataset_root_folder}/mindrecord/ \
  --num_shards=128 \
  --validation_split_size=0.2

exit 0
