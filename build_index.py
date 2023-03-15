import sys
sys.stdout.reconfigure(encoding='utf-8')

import openai
import numpy as np
import json
import textwrap


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')

def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


if __name__ == '__main__':
    alltext = open_file('input.txt')
    chunks = textwrap.wrap(alltext, 4000)
    result = list()
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='utf-8', errors='ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        print(info, '\n\n\n'.encode(encoding='utf-8', errors='ignore').decode())
        result.append(info)
    with open('index.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)