import openai
import os
from time import time,sleep
import json
import textwrap
import re


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    save_gpt3_log(content, str(vector))
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
    

def build_index():
    files = os.listdir('nexus/')
    changes = False
    for file in files:
        if file_exists(file, nexusindex):
            continue
        changes = True
        vector = gpt3_embedding(read_file(memorydir+file))
        nexusindex.append({'filename':file, 'vector': vector})
    return nexusindex
    

if __name__ == '__main__':
    alltext = open_file('input.txt')
    chunks = textwrap.wrap(alltext, 2000)
    result = list()
    count = 0
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        result.append({'content': chunk, 'vector': embedding})
    with open('index.json', 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=2)