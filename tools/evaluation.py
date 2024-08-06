import os
import yaml
import pandas as pd
from faker import Faker
from datetime import datetime
from collections import defaultdict
import traceback

fake = Faker()
current_dir = os.path.dirname(__file__)

def load_metadata(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

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

def generate_new_feature(feature, dataframes):
    new_features = defaultdict(list)  
    for index, row in feature.iterrows():
        try:
            locals().update(dataframes)
            code_string = row['pandas_formular']
            exec(code_string)
        except Exception as e:
            print(f"Error in row {index}: {e}")
            traceback.print_exc()
    return new_features

def count_common_lists(a, b):
    set_a = {tuple(lst) for lst in a.values()}
    set_b = {tuple(lst) for lst in b.values()}
    common_lists = set_a & set_b
    common_keys_a = {key for key, value in a.items() if tuple(value) in common_lists}
    common_keys_b = {key for key, value in b.items() if tuple(value) in common_lists}
    return len(common_lists), common_keys_a, common_keys_b

def evaluate_features(dataset_name, model_name):
    yaml_path = os.path.join(current_dir, f"../dataset/{dataset_name}/metadata.yaml")
    metadata = load_metadata(yaml_path)
    dataframes = load_mock_data(dataset_name, metadata)

    feature_model_path = os.path.join(current_dir, f"../test_feature/{dataset_name}/{model_name}.csv")
    feature_model = pd.read_csv(feature_model_path)
    new_features_model = generate_new_feature(feature_model, dataframes)

    feature_human_path = os.path.join(current_dir, f"../test_feature/{dataset_name}/human.csv")
    feature_human = pd.read_csv(feature_human_path)
    new_features_human = generate_new_feature(feature_human, dataframes)

    num_overlap, overlap_model_name, overlap_human_name = count_common_lists(new_features_model, new_features_human)
    num_ground_truth = len(new_features_human)
    recall = num_overlap / num_ground_truth
    
    print(f'Recall of {model_name} for {dataset_name}: {recall}')
    print(f'Common Keys in human_gold_lists: {overlap_human_name}')
    return recall, overlap_model_name, overlap_human_name

if __name__ == '__main__':
    # Example usage
    dataset_name = "avito"
    model_name = "gpt"
    evaluate_features(dataset_name, model_name)
