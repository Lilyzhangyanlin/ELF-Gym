import jinja2
import tqdm
import traceback
import signal

from .llm_executor import get_code_output
from .jinja_utils import jinja_render

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def handler(signum, frame):
    raise TimeoutError("Timeout!")

def llm_propose_features_get_prompt(table_desc, target_table, target_column):
    return jinja_render(
        'feature_proposal.jinja',
        table_desc=table_desc,
        target_table=target_table,
        target_column=target_column
    )


def llm_propose_features(table_desc, target_table, target_column, chat_completion_func):
    prompt = llm_propose_features_get_prompt(table_desc, target_table, target_column)
    message = chat_completion_func([prompt])
    print(message)
    lines = message.split('\n')
    feature_descs = [line[2:] for line in lines if line.startswith('* ')]
    return feature_descs

def llm_propose_n_features_get_prompt(table_desc, target_table, target_column, num_features=None):
    return jinja_render(
        'feature_proposal_num.jinja',
        table_desc=table_desc,
        target_table=target_table,
        target_column=target_column,
        num_features=num_features
    )

def llm_propose_n_features(table_desc, target_table, target_column, chat_completion_func, num_features=None, num_attempts=3):
    assert num_features is None or num_features > 0
    assert num_attempts > 0
    prompt = llm_propose_n_features_get_prompt(table_desc, target_table, target_column, num_features)
    prompts = [prompt]
    total_feature_descs = []
    feature_names = []
    for i in range(num_attempts):
        message = chat_completion_func([prompt])
        print(message)
        lines = message.split('\n')
        feature_descs = [line[2:] for line in lines if line.startswith('* ')]
        feature_descs = [fd for fd in feature_descs if fd.split('-')[0].strip() not in feature_names]
        total_feature_descs += feature_descs
        feature_names += [fd.split('-')[0].strip() for fd in feature_descs]
        
        prompts.append(message)
        if num_features is None:
            return total_feature_descs
        elif len(total_feature_descs) >= num_features:
            break
        logger.info(f"WARNING: Only generate {len(total_feature_descs)} features, continue to generate...")
        prompts.append(jinja_render('feature_proposal_continue.jinja'))

    print()
    logger.info("The full conversation is as follows:")
    for prompt in prompts:
        logger.info(prompt)

    logger.info(f"Finally generate {len(total_feature_descs)} features, keep {min(num_features, len(total_feature_descs))} features.")
    return total_feature_descs[:num_features]


def llm_write_code_get_prompt(table_desc, target_table, target_column, new_feature_desc):
    return jinja_render(
        'code_generation.jinja',
        table_desc=table_desc,
        target_table=target_table,
        target_column=target_column,
        new_feature_desc=new_feature_desc
    )


def llm_complain_prompt(error_message):
    return jinja_render(
        'complain.jinja',
        error=error_message
    )


def llm_write_code(dataframes, table_desc, target_table, target_column, feature_descs, chat_completion_func, num_attempts=3, real_dataframes = None):
    code_blocks = []
    for desc in tqdm.tqdm(feature_descs):
        prompt = llm_write_code_get_prompt(table_desc, target_table, target_column, desc)
        prompts = [prompt]
        for i in range(num_attempts):
            message = chat_completion_func(prompts)

            if message == 'None':
                code_blocks.append("")
                break
    
            # Claude 3, LLaMA and Mixtral ignores my instruction not to give any explanation.
            # Also, Claude 3 and Mixtral sometimes ignore my requirement to add a column with a specific name.
            # Sigh...
            try:
                code_block_start = message.index('```python') + 9
                code_block_end = message[code_block_start:].index('```') + code_block_start
                code_block = message[code_block_start:code_block_end]
            except ValueError:
                logger.info(f'Failed to generate feature for {desc} for attempt {i}.\n Original answer: {message}')
                continue

            # Sometimes LLMs generate unrunnable code, so we feed in the error message and let it try again...
            try:
                get_code_output(dataframes, target_table, code_block)
                if real_dataframes is not None:
                    # add time limit for each code block
                    signal.signal(signal.SIGALRM, handler)
                    signal.alarm(60)
                    get_code_output(real_dataframes, target_table, code_block)
                    signal.alarm(0)
            except TimeoutError:
                logger.info(f'TIMEOUT when executing generated code for {desc}.')
            except Exception as error:
                # error = traceback.format_exc()
                logger.info(f'Failed to execute generated code for {desc} for attempt {i}.\n Original answer: {message}\n Error: {error}\n')
                # Add the previous response and the error message
                prompts.append(message)
                prompts.append(llm_complain_prompt(error))
                continue

            code_blocks.append(code_block)
            break
        else:
            code_blocks.append("")
    return code_blocks

def llm_compare_features(table_desc,human_feature_des, model_feature_des, chat_completion_func):
    prompt = jinja_render(
        'feature_compare.jinja',
        table_desc=table_desc,
        human_feature_des=human_feature_des,
        model_feature_des=model_feature_des
    )
    message = chat_completion_func([prompt])
    return message