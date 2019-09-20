import pandas as pd
import numpy as np
import spacy
from textacy.corpus import Corpus
import json


# General Utility Functions
def split_data(data, labels, split):
    '''
    Splits given `data`. Can be used for train/test and train/validation splits.
    '''

    split_samples = int(split * data.shape[0])
    X_main = data[:-split_samples]
    y_main = labels[:-split_samples]
    X_sub = data[-split_samples:]
    y_sub = labels[-split_samples:]

    return X_main, y_main, X_sub, y_sub

def shuffle_data(data, labels):
    ''' Randomly shuffles `data` and `labels` iterables '''
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]

def generate_onehot_label(label):
    if label == -1: return np.array([0, 1])
    elif label == 0: return np.array([0, 0])
    elif label == 1: return np.array([1, 0])

def calculate_vocab_size(texts):
    ''' returns vocabulary size and corpus object '''
    corpus = Corpus(spacy.load('en_core_web_lg'), texts=list(texts))
    return corpus.n_tokens

# Parsing JSON Results
def parse_for_trial_results(filename):
    f = open(filename)
    res = json.load(f)
    val_loss = [v[-1] for v in [r['val_loss'] for _, r in res.items()]]
    val_acc = [v[-1] for v in [r['val_acc'] for _, r in res.items()]]
    loss = [v[-1] for v in [r['loss'] for _, r in res.items()]]
    acc = [v[-1] for v in [r['acc'] for _, r in res.items()]]

    return val_loss, val_acc, loss, acc