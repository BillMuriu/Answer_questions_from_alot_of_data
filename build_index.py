import openai
import json
import textwrap


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('api_key.txt')

def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


if __name__ == '__main__':
    alltext = open_file('input.txt')
    chunks = textwrap.wrap(alltext, 4000)
    result = list()
    count = 0
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        result.append({'content': chunk, 'vector': embedding})
    with open('index.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)