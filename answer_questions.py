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


def search_index(text, nexusindex, count=5, olderthan=None):
    vector = gpt3_embedding(text)
    scores = list()
    for i in nexusindex:
        if i['vector'] == vector:  # this is identical, skip it
            continue
        if olderthan:
            timestamp = get_timestamp(i['filename'])
            if timestamp > olderthan:
                continue
        score = similarity(vector, i['vector'])
        scores.append({'filename': i['filename'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    results = list()
    for i in ordered:
        results.append({'filename': i['filename'], 'score': i['score'], 'content': read_file('nexus/'+i['filename'])})
    if len(results) > count:
        return results[0:count]
    else:
        return results



if __name__ == '__main__':
    with open('index.json', 'r') as infile:
        data = json.load(infile)
        #print(data)
    while True:
        query = input('Enter your question here:')
        #print(query)
        results = search_index(query)