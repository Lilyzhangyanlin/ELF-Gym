from pathlib import Path
import pydantic
import yaml
import json

def save_pyd(pyd_model : pydantic.BaseModel, path : Path):
    # NOTE(minjie: Read/write using JSON to convert enum into string.
    data_dict = json.loads(pyd_model.json())
    with open(path, "w") as f:
        f.write(yaml.dump(data_dict))

def load_pyd(pyd_model_class, path : Path) -> pydantic.BaseModel:
    with open(path, "r") as f:
        return pyd_model_class.parse_obj(yaml.safe_load(f))
