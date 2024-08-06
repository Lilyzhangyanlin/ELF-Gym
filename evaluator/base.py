from enum import Enum
from typing import Tuple, Dict, Optional, List
import abc
from pathlib import Path
import pydantic

from task import Task

class Evaluator(object):
    config_class : pydantic.BaseModel = None
    name = 'base_evaluator'

    def __init__(self, task: Task, config):
        self._task = task
        self.config = config
        
    @abc.abstractmethod
    def evaluate(self, llm_outputs: Dict[str, Dict[str, str]], cache_dir: Path = None, use_cache: bool = False):
        pass

EVALUATOR_REGISTRY = {}

def evaluator(evaluator_class):
    global EVALUATOR_REGISTRY
    EVALUATOR_REGISTRY[evaluator_class.name] = evaluator_class
    return evaluator_class

def get_evaluator_class(name : str):
    global EVALUATOR_REGISTRY
    evaluator_class = EVALUATOR_REGISTRY.get(name, None)
    if evaluator_class is None:
        raise ValueError(f"Cannot find the evaluator of name {name}.")
    return evaluator_class

def get_evaluator_choice():
    names = EVALUATOR_REGISTRY.keys()
    return Enum("EvaluatorChoice", {name.upper() : name for name in names})
