from openai import OpenAI
import boto3
import json
import io
import tiktoken
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_gpt_completion(prompts):
    encoding = tiktoken.encoding_for_model("gpt-4")

    messages = [
        {
            'role': 'user' if i % 2 == 0 else 'assistant',
            'content': prompt
        }
        for i, prompt in enumerate(prompts)
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        temperature=0.01,
    )
    message = chat_completion.choices[0].message.content
    
    prompts.append(message)
    total_tokens = 0
    for prompt in prompts:
        tokens = encoding.encode(prompt)
        total_tokens += len(tokens)
    logger.info(f"Total tokens in one GPT conversation: {total_tokens}")
    return message.strip()


def get_llama3_completion(prompts, temperature=0.02, max_new_tokens=10000):
    """
    `prompt`: The input to LLM
    """
    if not prompts or len(prompts) == 0:
        return None

    buf = io.StringIO()
    for i, prompt in enumerate(prompts):
        role = 'Human' if i % 2 == 0 else 'Assistant'
        buf.write(f'\n\n{role}: {prompt}')
    buf.write('\n\nAssistant: ')

    answer_buf = io.StringIO()

    for _ in range(5):
        body = json.dumps({
            "prompt": buf.getvalue(),
            "temperature": temperature,
            "max_gen_len": 2048,
        })
        modelId = 'meta.llama3-70b-instruct-v1:0'
        accept = 'application/json'
        contentType = 'application/json'
    
        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    
        response_body = json.loads(response.get('body').read())
        # text
        # print(response_body.get('completion'))
        answer = response_body['generation'].strip()
        answer_buf.write(answer)

        if response_body['stop_reason'] == 'length':
            # generation not complete - feed back the previous output and keep generating
            buf.write(answer)
        else:
            break
    return answer_buf.getvalue()

def get_claude3_completion(prompts, temperature=0.01, max_new_tokens=10000):
    """
    `prompt`: The input to LLM
    """
    if not prompts or len(prompts) == 0:
        return None
        
    messages = [
        {
            'role': 'user' if i % 2 == 0 else 'assistant',
            'content': [{"type": "text", "text": prompt}]
        }
        for i, prompt in enumerate(prompts)
    ]
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 100000,
        "temperature": temperature,
    })
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())
    # text
    # print(response_body.get('completion'))
    return response_body['content'][0]['text'].strip()

def get_mixtral_completion(prompts, temperature=0.01, max_new_tokens=10000):
    """
    `prompt`: The input to LLM
    """
    if not prompts or len(prompts) == 0:
        return None

    buf = io.StringIO()
    for i, prompt in enumerate(prompts):
        role = 'User' if i % 2 == 0 else 'Assistant'
        buf.write(f'\n\n{role}: {prompt}')
    buf.write('\n\nAssistant: ')

    answer_buf = io.StringIO()

    for _ in range(5):
        body = json.dumps({
            "prompt": buf.getvalue(),
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
        })
        modelId = 'mistral.mixtral-8x7b-instruct-v0:1'
        accept = 'application/json'
        contentType = 'application/json'
    
        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    
        response_body = json.loads(response.get('body').read())
        # text
        # print(response_body.get('completion'))
        answer = response_body['outputs'][0]['text'].strip()
        answer_buf.write(answer)

        if response_body['outputs'][0]['stop_reason'] == 'length':
            # generation not complete - feed back the previous output and keep generating
            buf.write(answer)
        else:
            break
    return answer_buf.getvalue()