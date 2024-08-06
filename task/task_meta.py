from typing import Dict, Optional, List, Union
from enum import Enum
import pydantic


class ColumnDType(str, Enum):
    """Column data type model."""
    float_t = 'float'
    bool_t = 'bool'
    category_t = 'category'
    datetime_t = 'datetime'
    text_t = 'text'
    timestamp_t = 'timestamp'
    foreign_key = 'foreign_key'
    primary_key = 'primary_key'

DTYPE_EXTRA_FIELDS = {
    # link_to : A string in the format of <TABLE>.<COLUMN>
    ColumnDType.foreign_key : ["link_to"],
}

class ColumnSchema(pydantic.BaseModel):
    """Column schema model.

    Column schema allows extra fields other than the explicitly defined members.
    See `DTYPE_EXTRA_FIELDS` dictionary for more details.
    """
    class Config:
        extra = pydantic.Extra.allow
        use_enum_values = True

    # Column name.
    name : str
    # Column data type.
    dtype : ColumnDType
    # Column description.
    description : str = "Unknown feature"

class DataFormat(str, Enum):
    PARQUET = 'parquet'
    CSV = 'csv'

class TableSchema(pydantic.BaseModel):
    """Table schema model."""

    # Name of the table.
    name : str
    # Column schemas.
    columns: List[ColumnSchema]
    # Time column name.
    time_column: Optional[str]

class TaskMeta(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.allow
        use_enum_values = True
    
    name: str

    table_schemas: List[TableSchema]
    target_table: str
    target_column: str

    table_path: Optional[str] = None
    task_split: Optional[Union[List[float], str]] = None

    human_feature_desc: Optional[Dict[str, str]] = {}
    human_feature_impl: Optional[Dict[str, str]] = {}



