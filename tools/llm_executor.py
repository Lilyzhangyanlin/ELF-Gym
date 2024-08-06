from faker import Faker
from collections import namedtuple
import pandas as pd
import numpy as np
import time
import os
import traceback
from .jinja_utils import jinja_render

ColumnInfo = namedtuple('ColumnInfo', ['feature_desc', 'name'])

fake = Faker()

def generate_mock_data(table_schema, num_rows=100):
    mock_data = {column['name']: [] for column in table_schema['columns']}
    primary_keys = list(range(1, num_rows + 1))
    for _ in range(num_rows):
        for column in table_schema['columns']:
            if column['dtype'] == 'primary_key':
                primary_key_value = primary_keys.pop(0)
                mock_data[column['name']].append(primary_key_value)
            elif column['dtype'] == 'category':
                mock_data[column['name']].append(fake.random.choice(['A', 'B', 'C', 'D']))
            elif column['dtype'] == 'datetime':
                mock_data[column['name']].append(fake.date_time_between(start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 2)))
            elif column['dtype'] == 'foreign_key':
                mock_data[column['name']].append(fake.random_int(min=1, max=num_rows))
            elif column['dtype'] == 'text':
                mock_data[column['name']].append(fake.sentence(nb_words=5))
            elif column['dtype'] == 'float':
                mock_data[column['name']].append(fake.pyfloat(left_digits=3, right_digits=2, positive=True))
            elif column['dtype'] == 'bool':
                mock_data[column['name']].append(fake.random_int(min=0, max=1))
    return pd.DataFrame.from_dict(mock_data)

def load_mock_data(dataset_name, metadata):
    dataframes = {}
    for table in metadata['tables']:
        table_name = table['name']
        print(f"Generating mock data for table: {table_name}")
        df = generate_mock_data(table)
        dataframes[table_name] = df
    return dataframes

def run_code(dataframes, target_table_name, code_block):
    module_content = jinja_render(
        'module.jinja',
        table_names=dataframes.keys(),
        target_table_name=target_table_name,
        code_block=code_block,
    )
    unique_filename = f"_test_{np.random.randint(100000000)}.py"
    with open(unique_filename, 'w') as f:
        print(module_content, file=f)
    # import _test
    import importlib
    time.sleep(0.1)
    try:
        _test = importlib.import_module(unique_filename.rstrip(".py"))
        importlib.reload(_test)
        result = _test.func(dataframes)
    except:
        error = traceback.format_exc()
        os.remove(unique_filename)
        raise Exception(error)
    os.remove(unique_filename)
    return result

def get_code_output(dataframes, target_table_name, code):
    result_table = run_code(dataframes, target_table_name, code)
    diff_table = result_table.drop(columns=dataframes[target_table_name].columns, errors='ignore')
    return diff_table

def collect_new_features(dataframes, target_table_name, feature_descs, code_blocks):
    new_column_infos = []
    new_columns = []
    for feature_desc, code_block in zip(feature_descs, code_blocks):
        code_output = get_code_output(dataframes, target_table_name, code_block)
        for column in code_output.columns:
            new_column_infos.append(ColumnInfo(feature_desc, column))
            new_columns.append(code_output[column])
    return new_columns, new_column_infos
