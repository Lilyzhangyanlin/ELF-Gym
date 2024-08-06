from typing import Dict, Optional, Union, Literal
import os
import pandas as pd
import numpy as np
import traceback
import signal
import time
# import pickle
import dill as pickle 
import pydantic
from pathlib import Path

from tools.llm_executor import get_code_output
from .base import Evaluator, evaluator
from .utils import DOWNSAMPLE_FUNCTIONS, downsample

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FunctionalEvaluatorConfig(pydantic.BaseModel):
    time_budget: int = 600
    downsample: Optional[Union[str, Literal['__specify__']]] = '__specify__'

@evaluator
class FunctionalEvaluator(Evaluator):
    config_class = FunctionalEvaluatorConfig
    name = 'functional'

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
        logger.info(f"Generating features for human features...")
        new_features_human, human_error = generate_features(dataframes, target_table, human_feature_impl, human_cache_fname, use_cache, self.config.time_budget)

        results = []
        for llm_name, llm_output in llm_outputs.items():
            llm_cache_fname = cache_dir / llm_name / 'cache.pkl' if cache_dir else None

            logger.info(f"Generating features for LLM: {llm_name}...")
            new_features_model, model_error = generate_features(dataframes, target_table, llm_output, llm_cache_fname, use_cache, self.config.time_budget)

            logger.info(f"Evaluating features for LLM: {llm_name}...")
            recall, overlap_model_name, overlap_human_name = evaluate_features(new_features_human, new_features_model, check_functional_equality)

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

def handler(signum, frame):
    raise TimeoutError("Timeout!")

def generate_features(dataframes, target_table, desc_code: Dict[str, str], cache_fname=None, use_cache=False, time_budget=600):
    if use_cache and cache_fname and os.path.exists(cache_fname):
        with open(cache_fname, 'rb') as f:
            new_features, error_list = pickle.load(f)
    else:
        new_features = {}
        time_error = []
        length_error = []
        code_error = []
        all_nan_error = []
        identical_error = []
        for index, (feature_name, code) in enumerate(desc_code.items()):
            print()
            try:
                # add time limit for each code block
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(time_budget)
                logger.info(f'Generating code for Feature #{index}: {feature_name}.')
                t0 = time.time()
                new_feature = get_code_output(dataframes, target_table, code)
                t1 = time.time()
                signal.alarm(0)
            except TimeoutError:
                print(f'TIMEOUT when executing generated code.')
                time_error.append(index)
                continue
            except:
                error = traceback.format_exc()
                print(f'Failed to execute generated code for line {index}')
                print(f'Error: {error}')
                code_error.append(index)
                continue
            if len(new_feature) != len(dataframes[target_table]):
                print(f'Length of new feature is not equal to length of target table, skip it.')
                length_error.append(index)
                continue
            logger.info(f"Finished execution in {t1 - t0} seconds.")
            name = new_feature.columns.tolist()
            if new_feature.shape[1] == 0:
                print(new_feature)
                print(f'ERROR: There is no new column generated, skip it.')
                code_error.append(index)
                continue
            elif new_feature.shape[1] > 1:
                print(f'WARNING: There are more than one new column generated, column names are {name}, only keep the last one {[name[-1]]}.')
            if not name:
                print(f'Empty new feature name.')
                key = index
                value = new_feature.values
            else:
                key = name[-1]
                value = new_feature[key].values
            
            # Change name if already exists
            if key in new_features:
                print(f'WARNING: Feature name {key} already exists, change it to {key}_{index}')
                key = f'{key}_{index}'

            if pd.isna(value).all():
                print(f'VALUEERROR: All values are nan, skip it')
                all_nan_error.append(index)
                continue
            elif (not pd.isna(value).any()) and len(pd.Series(value).value_counts()) == 1:
                print(f'The only value in the generated feature is {value[0]}.')
                print(f'VALUEERROR: Only one unique value in generated feature, skip it.')
                identical_error.append(index)
                continue
            else:
                new_features[key] = value
        error_list = [time_error, length_error, code_error, all_nan_error, identical_error]
        if cache_fname:
            cache_fname = Path(cache_fname)
            cache_fname.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_fname, 'wb') as f:
                pickle.dump([new_features, error_list], f)
    return new_features, error_list

def is_numeric_dtype(dtype):
    # Check if the dtype is a pandas extension type
    if pd.api.types.is_extension_array_dtype(dtype):
        # If it's a numeric extension type, treat it as numeric
        return pd.api.types.is_numeric_dtype(dtype)
    # Otherwise, use numpy's function to check for numeric dtype
    return np.issubdtype(dtype, np.number)

def check_functional_equality(v_human: np.ndarray, v_model: np.ndarray, **kwargs):
    return np.isclose(v_human.astype(float), v_model.astype(float), equal_nan=True).all()

def count_common_lists(human, model, check_func, **kwargs):
    count = 0
    common_keys_human = []
    common_keys_model = []
    for k_human, v_human in human.items():
        for k_model, v_model in model.items():
            if v_human.shape != v_model.shape:
                continue
            if is_numeric_dtype(v_human.dtype) and is_numeric_dtype(v_model.dtype):
                logger.info(f"Comparing numeric columns: {k_human}, {k_model}")
                v_human = v_human.astype(float)
                v_model = v_model.astype(float)
                v_model[np.where(np.isnan(v_human))] = v_human[np.where(np.isnan(v_human))]
                try:
                    if check_func(v_human, v_model, **kwargs):
                        count += 1
                        common_keys_human.append(k_human)
                        common_keys_model.append(k_model)
                        break
                except:
                    error = traceback.format_exc()
                    logger.info(f"Error when comparing {k_human} and {k_model}.")
                    logger.info(f"v_human: {v_human}")
                    logger.info(f"v_model: {v_model}")
                    logger.info(f"The error is: {error}\n")
            else:
                logger.info(f"Comparing non-numeric columns: {k_human}, {k_model}")
                try:
                    if np.array_equal(v_human, v_model):
                        count += 1
                        common_keys_human.append(k_human)
                        common_keys_model.append(k_model)
                        break
                except:
                    error = traceback.format_exc()
                    logger.info(f"Error when comparing {k_human} and {k_model}.")
                    logger.info(f"v_human: {v_human}")
                    logger.info(f"v_model: {v_model}")
                    logger.info(f"The error is: {error}\n")
    return count, common_keys_model, common_keys_human

def evaluate_features(new_features_human, new_features_model, check_func, **kwargs):
    num_overlap, overlap_model_name, overlap_human_name = count_common_lists(new_features_human, new_features_model, check_func, **kwargs)
    num_ground_truth = len(new_features_human)
    recall = num_overlap / num_ground_truth
    
    logger.info(f'Recall of LLM: {recall}')
    logger.info(f'Common Keys in human_gold_lists: {overlap_human_name}')
    return recall, overlap_model_name, overlap_human_name







