import logging
from pathlib import Path
from enum import Enum
import pydantic
import argparse
import os

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.features.feature_metadata import FeatureMetadata as AGFeatMeta

from .yaml_utils import save_pyd, load_pyd
from .logger_utils import enable_log

enable_log()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


HP_MODE_FULL = {
    "NN_TORCH": {},
    "GBM": [
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {},
        "GBMLarge",
    ],
    "CAT": {},
    "XGB": {},
    "FASTAI": {},
    "RF": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
    ],
    "XT": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
    ],
}

HP_FAST = {
    "NN_TORCH": {},
    "GBM": [
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {},
        "GBMLarge",
    ],
    "CAT": {},
    "XGB": {},
    "FASTAI": {},
}

HP_MODE_XGB = {
    "XGB": {},
}

HP_MODE_NN = {
    "NN_TORCH": {},
}

class AGMode(str, Enum):
    # Run AG's default setting with all available models.
    full = "full"
    # Only run xgboost.
    xgb = "xgb"
    # Only run neural network.
    nn = "nn"
    # Run AG's fast models.
    fast = "fast"

class AGSolutionConfig(pydantic.BaseModel):
    time_limit_sec : int = 3600
    mode : AGMode = "fast"
    # Whether to enable bagging and stacking
    use_ensembling : bool = True


class AGSolution:
    """AutoGluon solution class."""
    config_class = AGSolutionConfig
    name = "ag"

    def __init__(
        self,
        solution_config : AGSolutionConfig,
    ):
        self.solution_config = solution_config

    def fit(
        self,
        train_df,
        valid_df,
        label_name,
        metric,
        ckpt_path : Path,
    ):
        self.metric = metric
        self.label_name = label_name
        train_set = TabularDataset(train_df)
        if valid_df is not None:
            valid_set = TabularDataset(valid_df)
        else:
            valid_set = None

        self.predictor = TabularPredictor(
            label=label_name,
            eval_metric = metric,
            path=ckpt_path,
            verbosity=4
        )

        extra_kwargs = {}
        if not self.solution_config.use_ensembling:
            extra_kwargs['auto_stack'] = False
            extra_kwargs['num_bag_folds'] = 0
            extra_kwargs['num_bag_sets'] = 1
            extra_kwargs['num_stack_levels'] = 0
        else:
            extra_kwargs['auto_stack'] = True
            extra_kwargs['use_bag_holdout'] = True

        if self.solution_config.mode == AGMode.full:
            hparams = HP_MODE_FULL
        elif self.solution_config.mode == AGMode.xgb:
            hparams = HP_MODE_XGB
        elif self.solution_config.mode == AGMode.nn:
            hparams = HP_MODE_NN
        elif self.solution_config.mode == AGMode.fast:
            hparams = HP_FAST
        else:
            raise ValueError(f"Unknown AG mode: {self.solution_config.mode}")

        if torch.cuda.is_available():
            gpu_devices = [f"cuda:{devid}" for devid in range(torch.cuda.device_count())]
        else:
            gpu_devices = []
            
        self.predictor.fit(
            train_set,
            tuning_data=valid_set,
            num_gpus=len(gpu_devices),
            time_limit=self.solution_config.time_limit_sec,
            hyperparameters=hparams,
            **extra_kwargs
        )
        if ckpt_path is not None:
            self.checkpoint(ckpt_path)

        logger.info(f"Best model: {self.predictor.get_model_best()}")
        # logger.info("Evaluating train set:")
        # evaluate_train_result = self.predictor.evaluate(train_set)
        # logger.info(evaluate_train_result)
        # evaluate_valid_result = None
        # if valid_set is not None:
        #     logger.info("Evaluating valid set:")
        #     evaluate_valid_result = self.predictor.evaluate(valid_set)
        #     logger.info(evaluate_valid_result)

        # return evaluate_train_result, evaluate_valid_result

    def evaluate(
        self,
        test_df,
    ) -> float:

        test_set = TabularDataset(test_df)
        logger.info("Evaluating test set:")
        logger.info(self.predictor.leaderboard(test_set))
        evaluate_result = self.predictor.evaluate(test_set)
        logger.info(evaluate_result)

        return evaluate_result

    def checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        if self.predictor is not None:
            self.predictor.save()

    def load_from_checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        self.solution_config = load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.predictor = TabularPredictor.load(ckpt_path)

    def compute_feature_importance(
        self,
        train_df,
    ):
        train_set = TabularDataset(train_df)
        logger.info("Computing feature importance...")
        feature_importance = self.predictor.feature_importance(train_set)

        return feature_importance

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', type=str, help='Dataset path')
    parser.add_argument('-c', type=str, help='Config path', default = None)
    parser.add_argument('-p', type=str, help='Checkpoint path', default = None)
    parser.add_argument('--label', type=str, help='label column name')
    parser.add_argument('--metric', type=str, default=None)

    args = parser.parse_args()
    logger.info(args)

    if args.c is None:
        logger.info("Use default config.")
        config = AGSolutionConfig()
    else:
        config = load_pyd(AGSolutionConfig, args.c)
    logger.info(config)
    if args.p is not None:
        ckpt_path = Path(args.p)
        ckpt_path.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_path = None

    if os.path.exists(os.path.join(args.d, 'train.pqt')):
        train = pd.read_parquet(os.path.join(args.d, 'train.pqt'))
    else:
        train = pd.read_csv(os.path.join(args.d, 'train.csv'))
    if os.path.exists(os.path.join(args.d, 'test.pqt')):
        test = pd.read_parquet(os.path.join(args.d, 'test.pqt'))
    else:
        test = pd.read_csv(os.path.join(args.d, 'test.csv'))
    logger.info(f'train shape: {train.shape}')
    logger.info(f'test shape: {test.shape}')
    if os.path.exists(os.path.join(args.d, 'valid.pqt')):
        valid = pd.read_parquet(os.path.join(args.d, 'valid.pqt'))
        logger.info(f'valid shape: {valid.shape}')
    elif os.path.exists(os.path.join(args.d, 'valid.csv')):
        valid = pd.read_csv(os.path.join(args.d, 'valid.csv'))
        logger.info(f'valid shape: {valid.shape}')
    else:
        valid = None
        logger.info('No valid set provided.')

    solution = AGSolution(config)
    solution.fit(train, valid, args.label, args.metric, ckpt_path)
    solution.evaluate(test)


