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


def search_index(text, data, count=10):
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    results = list()
    return results[0:count]


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


if __name__ == '__main__':
    with open('index.json', 'r') as infile:
        data = json.load(infile)
        #print(data)
    while True:
        query = input('Enter your question here:')
        #print(query)
        results = search_index(query, data)