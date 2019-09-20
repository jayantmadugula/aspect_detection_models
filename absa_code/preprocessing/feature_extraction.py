from textacy import preprocess, extract
from textacy.text_utils import detect_language

# Preprocessing
def preprocess_sentence(sent):
    # TODO check language?
    s = preprocess.normalize_whitespace(sent)
    return preprocess.preprocess_text(s, lowercase=True, transliterate=True, no_urls=True, no_phone_numbers=True, no_numbers=True, no_currency_symbols=True, no_contractions=True, no_accents=True)

# Basic Feature Extraction
def extract_ngrams(doc, n, min_freq=1):
    return extract.ngrams(doc, n, filter_stops=True, min_freq=min_freq)

def extract_svo(doc):
    return extract.subject_verb_object_triples(doc)

def extract_noun_chunks(doc, min_freq=1):
    return extract.noun_chunks(doc, drop_determiners=True, min_freq=min_freq)

def extract_named_entities(doc, min_freq=1):
    return extract.named_entities(doc, drop_determiners=True, min_freq=min_freq)
