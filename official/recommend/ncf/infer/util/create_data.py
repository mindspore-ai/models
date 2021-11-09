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
"""Regularize dataset for trainning"""
import os
import shutil
import tempfile
import zipfile
import argparse

import six

ML_1M = "ml-1m"
ML_20M = "ml-20m"
DATASETS = [ML_1M, ML_20M]

RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"

GENRE_COLUMN = "genres"
ITEM_COLUMN = "item_id"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "user_id"

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]

arg_parser = argparse.ArgumentParser(description='movielens dataset')
arg_parser.add_argument("--data_path", type=str, default="./dataset/")
arg_parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "ml-20m"])
args, _ = arg_parser.parse_known_args()

def _transform_csv(input_path, output_path, names, skip_first, separator=","):
    """Transform csv to a regularized format.

    Args:
      input_path: The path of the raw csv.
      output_path: The path of the cleaned csv.
      names: The csv column names.
      skip_first: Boolean of whether to skip the first line of the raw csv.
      separator: Character used to separate fields in the raw csv.
    """
    if six.PY2:
        names = [six.ensure_text(n, "utf-8") for n in names]

    with open(output_path, "wb") as f_out, \
            open(input_path, "rb") as f_in:

        # Write column names to the csv.
        f_out.write(",".join(names).encode("utf-8"))
        f_out.write(b"\n")
        for i, line in enumerate(f_in):
            if i == 0 and skip_first:
                continue  # ignore existing labels in the csv

            line = six.ensure_text(line, "utf-8", errors="ignore")
            fields = line.split(separator)
            if separator != ",":
                fields = ['"{}"'.format(field) if "," in field else field
                          for field in fields]
            f_out.write(",".join(fields).encode("utf-8"))

def _regularize_1m_dataset(temp_dir):
    """
    ratings.dat
      The file has no header row, and each line is in the following format:
      UserID::MovieID::Rating::Timestamp
        - UserIDs range from 1 and 6040
        - MovieIDs range from 1 and 3952
        - Ratings are made on a 5-star scale (whole-star ratings only)
        - Timestamp is represented in seconds since midnight Coordinated Universal
          Time (UTC) of January 1, 1970.
        - Each user has at least 20 ratings

    movies.dat
      Each line has the following format:
      MovieID::Title::Genres
        - MovieIDs range from 1 and 3952
    """
    working_dir = os.path.join(temp_dir, ML_1M)

    _transform_csv(
        input_path=os.path.join(working_dir, "ratings.dat"),
        output_path=os.path.join(temp_dir, RATINGS_FILE),
        names=RATING_COLUMNS, skip_first=False, separator="::")

    _transform_csv(
        input_path=os.path.join(working_dir, "movies.dat"),
        output_path=os.path.join(temp_dir, MOVIES_FILE),
        names=MOVIE_COLUMNS, skip_first=False, separator="::")

    shutil.rmtree(working_dir)

def _regularize_20m_dataset(temp_dir):
    """
    ratings.csv
      Each line of this file after the header row represents one rating of one
      movie by one user, and has the following format:
      userId,movieId,rating,timestamp
      - The lines within this file are ordered first by userId, then, within user,
        by movieId.
      - Ratings are made on a 5-star scale, with half-star increments
        (0.5 stars - 5.0 stars).
      - Timestamps represent seconds since midnight Coordinated Universal Time
        (UTC) of January 1, 1970.
      - All the users had rated at least 20 movies.

    movies.csv
      Each line has the following format:
      MovieID,Title,Genres
        - MovieIDs range from 1 and 3952
    """
    working_dir = os.path.join(temp_dir, ML_20M)

    _transform_csv(
        input_path=os.path.join(working_dir, "ratings.csv"),
        output_path=os.path.join(temp_dir, RATINGS_FILE),
        names=RATING_COLUMNS, skip_first=True, separator=",")

    _transform_csv(
        input_path=os.path.join(working_dir, "movies.csv"),
        output_path=os.path.join(temp_dir, MOVIES_FILE),
        names=MOVIE_COLUMNS, skip_first=True, separator=",")

    shutil.rmtree(working_dir)


if __name__ == "__main__":
    if args.dataset not in DATASETS:
        raise ValueError("dataset {} is not in {{{}}}".format(
            args.dataset, ",".join(DATASETS)))

    data_subdir = os.path.join(args.data_path, args.dataset)
    if not os.path.exists(data_subdir):
        os.makedirs(data_subdir)

    zip_path = os.path.join(args.data_path, "{}.zip".format(args.dataset))
    tempfile_dir = tempfile.mkdtemp()
    try:
        zipfile.ZipFile(zip_path, "r").extractall(tempfile_dir)
        if args.dataset == ML_1M:
            _regularize_1m_dataset(tempfile_dir)
        else:
            _regularize_20m_dataset(tempfile_dir)

        for fname in os.listdir(tempfile_dir):
            if not os.path.exists(os.path.join(data_subdir, fname)):
                shutil.copy(os.path.join(tempfile_dir, fname),
                            os.path.join(data_subdir, fname))
            else:
                logging.info("Skipping copy of {}, as it already exists in the "
                             "destination folder.".format(fname))

    finally:
        shutil.rmtree(tempfile_dir)
