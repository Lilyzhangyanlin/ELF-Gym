from pathlib import Path
import typer
import logging
import os
import numpy as np
import pandas as pd
import json

from evaluator import get_evaluator_choice, get_evaluator_class
from tools import yaml_utils
from task import Task

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

EvaluatorChoice = get_evaluator_choice()

def evaluate(
    task_path : str = typer.Argument(
        ...,
    ),
    llm_output_path : str = typer.Argument(
        ...,
    ),
    evaluator_name : EvaluatorChoice = typer.Argument(
        ...,
    ),
    output_path : str = typer.Argument(
        ...,
    ),
    config_path : Path = typer.Option(
        None,
        "--config_path", "-c",
    ),
    num : int  = typer.Option(
        None,
        "--num", "-n",
    ),
    cache_path : Path = typer.Option(
        None,
        "--cache",
    ),
    enable_cache : bool = typer.Option(
        False,
        "--enable-cache/--disable-cache",
    ),
):
    evaluator_class = get_evaluator_class(evaluator_name.value)
    if config_path is None:
        logger.info("No solution configuration file provided. Use default configuration.")
        config = evaluator_class.config_class()
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        config = yaml_utils.load_pyd(evaluator_class.config_class, config_path)

    logger.debug(f"Config:\n{config.json()}")

    logger.info("Loading task ...")
    task = Task(task_path)
    task_name = task.name
    logger.info(f"Task: {task_name} loaded.")

    logger.info("Creating evaluator ...")
    evaluator = evaluator_class(task, config)

    logger.info("Loading LLM output ...")
    llm_outputs = {}
    for file in os.listdir(llm_output_path):
        if file.endswith(".csv"):
            llm_name = file[:-4]
            logger.info(f"Loading LLM ({llm_name}) output ...")
            llm_output = pd.read_csv(os.path.join(llm_output_path, file))
            if num is not None:
                logger.info(f"Select top {num} features ...")
                llm_output = llm_output.head(num)
            names, descs, codes = [], [], []
            for i, row in llm_output.iterrows():
                fd = row['feature_description'].split('-')
                names.append(fd[0].strip())
                descs.append('-'.join(fd[1:]).strip())
                codes.append(row['code'])

            if evaluator.name == 'semantic':
                llm_outputs[llm_name] = dict(zip(names, descs))
            else:
                llm_outputs[llm_name] = dict(zip(names, codes))

    logger.info("Evaluating ...")
    cache_path = Path(cache_path) / task_name if cache_path else None
    results = evaluator.evaluate(
        llm_outputs,
        cache_path,
        enable_cache
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_fn = output_path / f"{task_name}_{evaluator_name.value}.json"
    for result in results:
        fp = open(output_fn, "a")
        json.dump(result, fp)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
        fp.close()
