import numpy as np
import spacy

def generate_pos_tags(texts):
    '''
    Returns part-of-speech tags for word in the center
    of each text in `texts`.

    `texts` must be an iterable of space-delimited
    strings

    For ABSA analysis, this function should be called
    on ngrams
    '''
    sp_docs = generate_docs(texts)
    word_pos_tags = _generate_pos_tags(sp_docs)
    return word_pos_tags

def _generate_pos_tags(docs):
    ''' Returns tag of middle word for each document '''
    tags = []
    for d in docs:
        doc_tag = d[int(len(d)/2)].tag
        tags.append(doc_tag)
    return np.stack(tags)

# SpaCy
def generate_docs(texts):
    ''' 
    `texts` must be an iterable of strings
    each string in `texts` is parsed into a spaCy `Doc` object
    '''
    nlp = spacy.load('en_core_web_lg')
    return nlp.pipe(texts, batch_size=10000, n_threads=8)

def get_doc_vectors(docs):
    ''' Returns word vectors for all texts in `docs` using spaCy '''
    return np.stack([d.vector for d in docs])

def get_doc_vector(doc):
    ''' spaCy `doc` returns mean of word vectors in `doc` '''
    return doc.vector

def get_doc_tokens(docs):
    tags = []
    for d in docs:
        doc_tags = [t.tag for t in d]
        tags.append(doc_tags)
    return np.stack(tags)
