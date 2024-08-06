from typing import Dict
import os
import pandas as pd
import json
from tqdm import tqdm 
import pydantic
from pathlib import Path

from task import Task, TaskMeta
from tools.llm_prompting import llm_compare_features
import tools.llm_completions
from .base import Evaluator, evaluator

from typing import Literal
from pydantic import BaseModel

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class SemanticEvaluatorConfig(BaseModel):
    llm: Literal['gpt', 'claude3', 'llama3', 'mixtral'] = 'gpt'

@evaluator
class SemanticEvaluator(Evaluator):
    config_class = SemanticEvaluatorConfig
    name = 'semantic'

    def evaluate(self, llm_outputs: Dict[str, Dict[str, str]], cache_dir: Path = None, use_cache: bool = False):
        table_desc = extract_table_desc(self._task.metadata)
        human_features = self._task.human_feature_desc
        func = getattr(tools.llm_completions, f'get_{self.config.llm}_completion')

        results = []
        for llm_name, llm_output in llm_outputs.items():
            logger.info(f"Evaluating LLM: {llm_name}...")
            count = 0
            common_features = []
            for human_feat_name, human_feat_desc in human_features.items():
                print()
                logger.info(f"Evaluating human feature: {human_feat_name}...")
                for llm_feat_name, llm_feat_desc in llm_output.items():
                    res = llm_compare_features(table_desc, human_feat_desc, llm_feat_desc, func)
                    if res == '1':
                        print(f"Description of human feature `{human_feat_name}` is equal Description of LLM's feature `{llm_feat_name}`.")
                        common_features.append((human_feat_name, llm_feat_name))
                        count += 1
                        break

            recall = count / len(human_features)
            logger.info(f'Recall of LLM: {recall}')
            logger.info(f'Common feature in human and LLM: {common_features}')
            
            result = {
                "llm_name": llm_name,
                "recall": recall,
                "common_features": common_features
            }
            results.append(result)

        return results

def extract_table_desc(metadata: TaskMeta):
    table_desc = {}
    for table_schema in metadata.table_schemas:
        table_name = table_schema.name
        column_desc = {}
        for column_schema in table_schema.columns:
            column_desc[column_schema.name] = column_schema.description
        table_desc[table_name] = column_desc
    return table_desc

