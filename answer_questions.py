import openai
import json
import numpy as np


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()