from typing import Dict
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm 
import pydantic
from pathlib import Path

from .functional_evaluator import generate_features
from .base import Evaluator, evaluator
from .ag import AGSolutionConfig, AGSolution

class DownstreamEvaluatorConfig(pydantic.BaseModel):
    time_budget: int = 600
    solution_config: AGSolutionConfig = None

@evaluator
class DownstreamEvaluator(Evaluator):
    config_class = DownstreamEvaluatorConfig
    name = 'downstream'

    def evaluate(self, llm_outputs: Dict[str, Dict[str, str]], cache_dir: Path = None, use_cache: bool = False):
        human_feature_impl = self._task.human_feature_impl
        target_table = self._task.metadata.target_table
        target_column = self._task.metadata.target_column
        original_task_df = self._task.task_sets
        dataframes = self._task._tables
        if 'metric' in self._task.metadata.dict():
            metric = self._task.metadata.metric
        else:
            metric = None

        traindf = original_task_df['train'].reset_index(drop=True)
        masked_valid = original_task_df['valid'].copy()
        masked_valid[target_column] = np.nan
        validdf = pd.concat([original_task_df['train'], masked_valid], axis=0).reset_index(drop=True)
        masked_test = original_task_df['test'].copy()
        masked_test[target_column] = np.nan
        testdf = pd.concat([original_task_df['train'], original_task_df['valid'], masked_test], axis=0).reset_index(drop=True)
        taskdf = {
            'train': traindf,
            'valid': validdf,
            'test': testdf
        }

        if self.config.solution_config is None:
            self.config.solution_config = AGSolutionConfig()
        solution = AGSolution(self.config.solution_config)

        human_cache_dir = cache_dir / 'human' if cache_dir else None
        new_human_task_df = generate_new_task_df(dataframes, original_task_df, taskdf, target_table, human_feature_impl, human_cache_dir, use_cache, self.config.time_budget)
        human_ckpt_dir = cache_dir / 'human_ckpt' if cache_dir else None
        if use_cache and human_ckpt_dir and os.path.exists(human_ckpt_dir):
            solution.load_from_checkpoint(human_ckpt_dir)
        else:
            solution.fit(new_human_task_df['train'], new_human_task_df['valid'], target_column, metric, human_ckpt_dir)
        human_train_metric = solution.evaluate(new_human_task_df['train'])
        human_valid_metric = solution.evaluate(new_human_task_df['valid'])
        human_test_metric = solution.evaluate(new_human_task_df['test'])

        results = []
        result = {
            "human_train_metric": human_train_metric,
            "human_valid_metric": human_valid_metric,
            "human_test_metric": human_test_metric,
        }
        results.append(result)

        for llm_name, llm_output in llm_outputs.items():
            llm_cache_dir = cache_dir / llm_name if cache_dir else None
            new_llm_task_df = generate_new_task_df(dataframes, original_task_df, taskdf, target_table, llm_output, llm_cache_dir, use_cache, self.config.time_budget)
            llm_ckpt_dir = cache_dir / f'{llm_name}_ckpt' if cache_dir else None
            if use_cache and llm_ckpt_dir and os.path.exists(llm_ckpt_dir):
                solution.load_from_checkpoint(llm_ckpt_dir)
            else:
                solution.fit(new_llm_task_df['train'], new_llm_task_df['valid'], target_column, metric, llm_ckpt_dir)
            llm_train_metric = solution.evaluate(new_llm_task_df['train'])
            llm_valid_metric = solution.evaluate(new_llm_task_df['valid'])
            llm_test_metric = solution.evaluate(new_llm_task_df['test'])

            result = {
                "llm_name": llm_name,
                "llm_train_metric": llm_train_metric,
                "llm_valid_metric": llm_valid_metric,
                "llm_test_metric": llm_test_metric
            }
            results.append(result)

        return results


def generate_new_task_df(dataframes, original_task_df, taskdf, target_table, desc_code, cache_dir, use_cache, time_budget):
    new_task_df = {}
    cut_pos = {
        'train': 0,
        'valid': len(original_task_df['train']),
        'test': len(original_task_df['train']) + len(original_task_df['valid'])
    }
    for mode in ['train', 'valid', 'test']:
        print(f'Mode: {mode}')
        dataframes[target_table] = taskdf[mode]

        cache_fname = cache_dir / f'{mode}_cache.pkl' if cache_dir else None
        new_features, _ = generate_features(dataframes, target_table, desc_code, cache_fname, use_cache, time_budget)
        
        new_features = pd.DataFrame(new_features)
        new_features = new_features.reset_index(drop=True)
        new_features = new_features.iloc[cut_pos[mode]:].reset_index(drop=True)
        table = pd.concat([original_task_df[mode].reset_index(drop=True), new_features], axis=1).reset_index(drop=True)
        print(f'new {mode} set shape:', table.shape)
        new_task_df[mode] = table

    # Rename duplicated columns
    for mode in ['train', 'valid', 'test']:
        cols=pd.Series(new_task_df[mode].columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        new_task_df[mode].columns=cols
        print(f'{mode} set columns ({len(new_task_df[mode].columns)}):', new_task_df[mode].columns)
    
    # Only save common columns
    common_columns = list(set(new_task_df['train'].columns) & set(new_task_df['valid'].columns) & set(new_task_df['test'].columns))
    for mode in ['train', 'valid', 'test']:
        new_task_df[mode] = new_task_df[mode][common_columns]
    new_common_columns = list(set(common_columns) - set(original_task_df['train'].columns))
    print(f'New common columns ({len(new_common_columns)}):', new_common_columns)

    return new_task_df


