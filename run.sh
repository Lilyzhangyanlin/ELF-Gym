#!/bin/bash
dataname=airbnb
llm=gpt4o
evaluation_name=semantic
task_path=metadata/$dataname/$dataname.yaml
llm_output_path=./features_output/$dataname
output_path=./evaluation_result

# Generate features by LLM
python main.py generate $task_path $llm $llm_output_path

# Evaluate features
python main.py evaluate $task_path $llm_output_path $evaluation_name $output_path
