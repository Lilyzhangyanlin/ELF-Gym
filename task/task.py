import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
import json
import yaml

from .task_meta import TaskMeta
from tools import yaml_utils

__all__ = ['Task']

class Task:

    def __init__(self, path: Path):
        self.path = path
        self._metadata = yaml_utils.load_pyd(TaskMeta, path)
        self._load_data()

    def _load_data(self):
        self._tables = {}
        self._task_sets = {}
        if self._metadata.table_path is None:
            return
        path = Path(self._metadata.table_path)
        for table_schema in self._metadata.table_schemas:
            table_path = path / f'{table_schema.name}.pqt'
            table = pd.read_parquet(table_path)
            column_names = [column_schema.name for column_schema in table_schema.columns]
            table = table[column_names]
            self._tables[table_schema.name] = table
        
        if self._metadata.task_split is None:
            return
        target_table = self._metadata.target_table
        table_data = self._tables[target_table]
        if isinstance(self._metadata.task_split, str):
            split_df = pd.read_parquet(self._metadata.task_split)
            self._task_sets['train_set'] = table_data.loc[split_df['split'] == 'train']
            self._task_sets['valid_set'] = table_data.loc[split_df['split'] == 'valid']
            self._task_sets['test_set'] = table_data.loc[split_df['split'] == 'test']
        elif isinstance(self._metadata.task_split, list):
            split = self._metadata.task_split
            train_num = int(len(self._tables[target_table]) * split[0])
            valid_num = int(len(self._tables[target_table]) * split[1])
            for table_schema in self._metadata.table_schemas:
                if table_schema.name == target_table:
                    target_table_schema = table_schema
                    break
            if target_table_schema.time_column is not None:
                table_data = self._tables[target_table].sort_values(target_table_schema.time_column)
            self._task_sets['train'] = table_data.iloc[:train_num]
            self._task_sets['valid'] = table_data.iloc[train_num:train_num+valid_num]
            self._task_sets['test'] = table_data.iloc[train_num+valid_num:]
            
    @property
    def metadata(self) -> TaskMeta:
        return self._metadata
    
    @property
    def name(self) -> str:
        return self._metadata.name

    @property
    def human_feature_desc(self) -> Dict[str, str]:
        return self._metadata.human_feature_desc
    
    @property
    def human_feature_impl(self) -> Dict[str, str]:
        return self._metadata.human_feature_impl

    @property
    def tables(self) -> Dict[str, pd.DataFrame]:
        return self._tables
    
    @property
    def task_sets(self) -> Dict[str, pd.DataFrame]:
        return self._task_sets

    def load_desc_and_code(self, path: Path):
        human_feature_desc = {}
        keys = []
        feature_description = pd.read_csv(path)['feature_description'].values
        for item in feature_description:
            items = item.split('-')
            key = items[0]
            value = '-'.join(items[1:])
            key = clean_string(key)
            value = clean_string(value)
            human_feature_desc[key] = value
            keys.append(key)
        self._metadata.human_feature_desc = human_feature_desc

        human_feature_impl = {}
        code = pd.read_csv(path)['code'].values
        for i, item in enumerate(code):
            key = keys[i]
            value = item
            human_feature_impl[key] = value
        self._metadata.human_feature_impl = human_feature_impl
    
    def load_description(self, path: Path):
        with open(path) as f:
            desc = json.load(f)
        for table_schema in self._metadata.table_schemas:
            table_name = table_schema.name
            table_desc = desc.get(table_name, {})
            for column_schema in table_schema.columns:
                column_name = column_schema.name
                column_schema.description = table_desc.get(column_name, "Unknown feature")
    
    def save(self, path: Path):
        yaml_utils.save_pyd(self._metadata, path)


import re
def clean_string(s):
    s = s.strip()
    s = re.sub(r'^`+|`+$', '', s)
    return s