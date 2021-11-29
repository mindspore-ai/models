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
"""entity vocab file"""
from typing import List, TextIO, Dict
import json
import math
from pathlib import Path

import multiprocessing
from multiprocessing.pool import Pool
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

from src.utils.interwiki_db import InterwikiDB

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, MASK_TOKEN}

Entity = namedtuple("Entity", ["title", "language"])

_dump_db = None  # global variable used in multiprocessing workers


@click.command()
@click.argument("dump_db_file", type=click.Path())
@click.argument("out_file", type=click.Path())
@click.option("--vocab-size", default=500000)
@click.option("-w", "--white-list", type=click.File(), multiple=True)
@click.option("--white-list-only", is_flag=True)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_entity_vocab(dump_db_file: str, white_list: List[TextIO], **kwargs):
    """build entity vocab"""
    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocab.build(dump_db, white_list=white_list, language=dump_db.language, **kwargs)


class EntityVocab:
    """entity vocab"""

    def __init__(self, vocab_file: str):
        """init fun"""
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        # allow tsv files for backward compatibility
        if vocab_file.endswith(".tsv"):
            self._parse_tsv_vocab_file(vocab_file)
        else:
            self._parse_jsonl_vocab_file(vocab_file)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        """parse tsv vocab file"""
        with open(vocab_file, "r", encoding="utf-8") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        """parse json vocab file"""
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)

    @property
    def size(self) -> int:
        """size"""
        return len(self)

    def __reduce__(self):
        """reduce"""
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        """len"""
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        """contain"""
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        """get item"""
        return self.get_id(key, language=None)

    def __iter__(self):
        """create iter"""
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        """contain"""
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        """get id"""
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        """get title id"""
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title
        return ""

    def get_count_by_title(self, title: str, language: str = None) -> int:
        """get title count"""
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def save(self, out_file: str):
        """save"""
        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count}
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def build(
            dump_db: DumpDB,
            out_file: str,
            vocab_size: int,
            white_list: List[str],
            white_list_only: bool,
            pool_size: int,
            chunk_size: int,
            language: str,
    ):
        """build"""
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for title, count in counter.most_common():
                if title in valid_titles and not title.startswith("Category:"):
                    title_dict[title] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, "w") as f:
            for ent_id, (title, count) in enumerate(title_dict.items()):
                json.dump({"id": ent_id, "entities": [[title, language]], "count": count}, f)
                f.write("\n")

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        """init worker"""
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        """entity count"""
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        return counter


@click.command()
@click.option("entity_vocab_files", "-v", multiple=True)
@click.option("inter_wiki_db_path", "-i", type=click.Path())
@click.option("out_file", "-o", type=click.Path())
@click.option("vocab_size", "-s", type=int)
def build_multilingual_entity_vocab(
        entity_vocab_files: List[str], inter_wiki_db_path: str, out_file: str, vocab_size: int = 1000000
):
    """build entity vocab"""
    for entity_vocab_path in entity_vocab_files:
        if Path(entity_vocab_path).suffix != ".jsonl":
            raise RuntimeError(
                f"entity_vocab_path: {entity_vocab_path}\n"
                "Entity vocab files in this format is not supported."
                "Please use the jsonl file format and try again."
            )

    db = InterwikiDB.load(inter_wiki_db_path)

    vocab: Dict[Entity, int] = {}  # title -> index
    inv_vocab = defaultdict(set)  # index -> Set[title]
    count_dict = defaultdict(int)  # index -> count

    special_token_to_idx = {special_token: idx for idx, special_token in enumerate(SPECIAL_TOKENS)}
    current_new_id = len(special_token_to_idx)

    for entity_vocab_path in entity_vocab_files:
        with open(entity_vocab_path, "r") as f:
            for line in f:
                entity_dict = json.loads(line)
                for title, lang in entity_dict["entities"]:
                    entity = Entity(title, lang)
                    multilingual_entities = {entity}
                    if title not in SPECIAL_TOKENS:
                        aligned_entities = {Entity(t, ln) for t, ln in db.query(title, lang)}
                        multilingual_entities.update(aligned_entities)
                        # judge if we should assign a new id to these entities
                        already_registered_entities = aligned_entities & vocab.keys()
                        if already_registered_entities:
                            already_registered_entity = list(already_registered_entities)[0]
                            ent_id = vocab[already_registered_entity]
                        else:
                            ent_id = current_new_id
                            current_new_id += 1
                    else:
                        ent_id = special_token_to_idx[title]

                    vocab[entity] = ent_id
                    inv_vocab[ent_id].add((entity.title, entity.language))  # Convert Entity to Tuple for json.dump
                    count_dict[ent_id] += entity_dict["count"]
    json_dicts = [
        {"entities": list(inv_vocab[ent_id]), "count": count_dict[ent_id]} for ent_id in range(current_new_id)
    ]
    json_dicts.sort(key=lambda x: -x["count"] if x["count"] != 0 else -math.inf)
    json_dicts = json_dicts[:vocab_size]

    with open(out_file, "w") as f:
        for ent_id, item in enumerate(json_dicts):
            json.dump({"id": ent_id, **item}, f)
            f.write("\n")
