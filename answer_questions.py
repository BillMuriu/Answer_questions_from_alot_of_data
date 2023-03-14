import openai
import json
import numpy as np


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    

# openai.api_key = open_file('openaiapikey.txt')


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


if __name__ == '__main__':
    with open('index.json', 'r') as infile:
        data = json.load(infile)
    while True:
        query = input('Enter your question here:')
        #print(query)