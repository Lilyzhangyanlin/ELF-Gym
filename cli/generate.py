from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np
import pandas as pd
import json

from tools import yaml_utils
from task import Task
from tools.llm_completions import get_gpt_completion, get_claude3_completion, get_llama3_completion, get_mixtral_completion
from tools.llm_prompting import llm_propose_n_features, llm_write_code
from tools.evaluation import load_mock_data
from evaluator.semantic_evaluator import extract_table_desc
from evaluator.utils import DOWNSAMPLE_FUNCTIONS_FOR_GENERATE

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


LLMCOMPLETION = {
    'gpt4o': get_gpt_completion,
    'claude3sonnet': get_claude3_completion,
    'llama3': get_llama3_completion,
    'mixtral': get_mixtral_completion,
}

def generate(
    task_path : str = typer.Argument(
        ...,
    ),
    llm : str = typer.Argument(
        ...,
    ),
    output_path : str = typer.Argument(
        ...,
    ),
    num : int  = typer.Option(
        None,
        "--num", "-n",
    ),
    enable_real_data : bool = typer.Option(
        False,
        "--enable-real-data/--disable-real-data",
    ),
):

    if llm not in LLMCOMPLETION:
        raise ValueError(f"Unsupported LLM model: {llm}")

    logger.info("Loading task ...")
    task = Task(task_path)
    task_name = task.name
    logger.info(f"Task: {task_name} loaded.")

    table_desc = extract_table_desc(task.metadata)
    target_table = task.metadata.target_table
    target_column = task.metadata.target_column

    logger.info(f"Generating feature descriptions using {llm} ...")
    feature_descs = llm_propose_n_features(
        table_desc,
        target_table,
        target_column,
        LLMCOMPLETION[llm],
        num
    )

    dataframes = load_mock_data(task_name, {'tables': [schema.dict() for schema in task.metadata.table_schemas]})

    real_dataframes = None
    if enable_real_data:
        logger.info("Enable real data.")
        real_dataframes = task._tables
        if task_name in DOWNSAMPLE_FUNCTIONS_FOR_GENERATE:
            logger.info(f"Downsampling dataframes for task {task_name} ...")
            real_dataframes = DOWNSAMPLE_FUNCTIONS_FOR_GENERATE[task_name](real_dataframes)

    logger.info(f"Generating feature codes using {llm} ...")
    code_blocks = llm_write_code(
        dataframes,
        table_desc,
        target_table,
        target_column,
        feature_descs,
        LLMCOMPLETION[llm],
        real_dataframes = real_dataframes
    )
    df = pd.DataFrame({'feature_description': feature_descs, 'code': code_blocks})
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / f'{llm}.csv')

