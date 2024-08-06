from typing import Dict, Optional, Union, Literal
import os
import pandas as pd
import numpy as np
import traceback
import signal
import time
import pickle
import pydantic
from pathlib import Path

from .functional_evaluator import generate_features, evaluate_features
from .base import Evaluator, evaluator
from .utils import DOWNSAMPLE_FUNCTIONS, downsample

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class CorrelationEvaluatorConfig(pydantic.BaseModel):
    time_budget: int = 600
    threshold: float = 0.9
    downsample: Optional[Union[str, Literal['__specify__']]] = None

@evaluator
class CorrelationEvaluator(Evaluator):
    config_class = CorrelationEvaluatorConfig
    name = 'correlation'

    def evaluate(self, llm_outputs: Dict[str, Dict[str, str]], cache_dir: Path = None, use_cache: bool = False):
        human_feature_impl = self._task.human_feature_impl
        target_table = self._task.metadata.target_table
        dataframes = self._task._tables
        if self.config.downsample is not None:
            logger.info("Downsampling dataframes...")
            if self.config.downsample == '__specify__':
                if self._task.name in DOWNSAMPLE_FUNCTIONS:
                    dataframes = DOWNSAMPLE_FUNCTIONS[self._task.name](dataframes)
                else:
                    logger.info(f"WARNING: No downsample function specified for task {self._task.name}, skip it.")
            else:
                dataframes = downsample(dataframes, self.config.downsample, self._task.metadata.table_schemas)
        human_cache_fname = cache_dir / 'human' / 'cache.pkl' if cache_dir else None
        new_features_human, human_error = generate_features(dataframes, target_table, human_feature_impl, human_cache_fname, use_cache, self.config.time_budget)

        results = []
        for llm_name, llm_output in llm_outputs.items():
            llm_cache_fname = cache_dir / llm_name / 'cache.pkl' if cache_dir else None

            logger.info(f"Generating features for LLM: {llm_name}...")
            new_features_model, model_error = generate_features(dataframes, target_table, llm_output, llm_cache_fname, use_cache, self.config.time_budget)

            logger.info(f"Evaluating features for LLM: {llm_name}...")
            recall, overlap_model_name, overlap_human_name = evaluate_features(new_features_human, new_features_model, check_correlation_equality, threshold=self.config.threshold)

            result = {
                "llm_name": llm_name,
                "recall": recall,
                "overlap_model_name": overlap_model_name,
                "overlap_human_name": overlap_human_name,
                "num_ground_truth": len(new_features_human),
                "num_human_error": [len(e) for e in human_error],
                "human_error": human_error,
                "num_model_feat": len(new_features_model),
                "num_model_error": [len(e) for e in model_error],
                "model_error": model_error
            }
            results.append(result)

        return results

def check_correlation_equality(x, y, **kwargs):
    x = x.astype(float)
    y = y.astype(float)
    # replace nan values in x,y to its mean
    x[np.where(np.isnan(x))] = np.nanmean(x)
    y[np.where(np.isnan(y))] = np.nanmean(y)
    corr = np.corrcoef(x, y)[0,1]
    threshold = kwargs.get('threshold', 0.9)
    return corr >= threshold







