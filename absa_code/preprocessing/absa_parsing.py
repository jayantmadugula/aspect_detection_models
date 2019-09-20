import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import feedparser as fp
from string import punctuation
from multiprocessing import Pool

NUM_PROCESSES = 8

# Convenience ABSA Function
def extract_data_ABSA16(cws=2, filepath='./data/ABSA16_Restaurants_Train_SB1_v2.xml', include_sentiment=False):
    '''
    Convenience Function that reads the ABSA16 XML file

    Final return value depends on `include_sentiment` flag.

    Returns three iterables:

    - `texts`
    - `polarities`
    - DataFrame containing either sentiment + target data or just target data
    '''
    raw_reviews = parse_xml_ABSA(filename=filepath)
    review_data = extract_all_data_ABSA(raw_reviews)
    targets = extract_target_info(review_data, context_window_size=cws)
    
    if include_sentiment:
        sentiment_df = pull_all_words_sentiment(review_data, cws)
        return review_data, targets, sentiment_df
    else:
        targets_df = pull_all_words_targets(review_data, cws)
        return review_data, targets, targets_df

# XML Data Handling
def parse_xml_ABSA(filename):
    '''
    Parses XML and returns children
    '''
    xml_tree = ET.parse(filename)
    root = xml_tree.getroot()
    children = root.getchildren()
    return children

def extract_all_data_ABSA(xml_reviews):
    ''' Convenience function to parse all ABSA reviews using `extract_data_ABSA()` '''
    with Pool(NUM_PROCESSES) as p:
        res = p.map(extract_data_ABSA, xml_reviews)
    return pd.concat(res, ignore_index=True)

def extract_data_ABSA(xml_review):
    '''
    Given review XML objects (see `parse_xml_ABSA()`)

    Returns pandas DataFrame with columns:
    * `review`: raw text of a review
    * `opinion_data`: specific information about the target
    '''
    sents = []
    opinion_attribs = []
    for r in xml_review[0]:
        if len(r) <= 1: continue

        opinion_df = pd.DataFrame([a.attrib for a in r[1]])
        opinion_df.loc[:, 'from'] = opinion_df['from'].astype(int)
        opinion_df.loc[:, 'to'] = opinion_df['to'].astype(int)
        
        sents.append(r[0].text)
        opinion_attribs.append(opinion_df)

    return pd.DataFrame({
        'review': sents, 
        'opinion_data': opinion_attribs
        })

# ABSA Target Handling
def extract_target_info(review_data, context_window_size, skip_null=True):
    ''' Get all targets with context windows for a given review '''
    targets = []
    for _, review in review_data.iterrows():
        for _, opinion in review['opinion_data'].iterrows():
            if skip_null and opinion['target'] == 'NULL': continue
            i = opinion['from']
            rev = review['review']
            target_window = pull_target(rev, i, n=context_window_size)
            polarity = _sentiment_to_int(opinion['polarity'])
            targets.append((target_window, polarity))

    return pd.DataFrame(targets, columns=('target', 'polarity'))

def pull_target(sentence, i, n=2):
    '''
    Returns the `i`-th word in `sentence` surrounded by `n` words on either side.

    Returned string has length `2n+1`.
    '''
    c_i = i
    count = 0
    sent_list =  sentence.split(' ')
    for w_i, word in enumerate(sent_list):
        if len(word) + count < c_i: count += len(word) + 1
        else: 
            # Padding, p_n prepends, a_n appends
            padding = 'inv'
            p_n = 0
            a_n = 0
            
            # Find window range [k:l]
            if w_i - n < 0: 
                p_n = 0 - (w_i - n) 
                k = 0
            else: k = w_i - n
            
            if w_i + n > len(sent_list) - 1: 
                a_n = (w_i + n) - len(sent_list) + 1
                l = len(sent_list)
            else: l = w_i + n + 1
                
            # Format and return target word with window
            target = sent_list[k:l]
            target = [padding]*p_n + target + [padding]*a_n
            return str(target).strip(punctuation) if len(target) == 1 else ' '.join([t.strip(punctuation) for t in target]).strip(punctuation)    

def _sentiment_to_int(polarity_str):
    '''
    Converts sentiment strings in ABSA dataset to numbers
    '''
    if polarity_str == 'negative': return -1
    elif polarity_str == 'positive': return 1
    else: return 0

def pull_all_words_sentiment(review_data, context_window_size):
    ''' 
    Target words at center of each phrase, labels are:
    * no sentiment (0)
    * neg (1)
    * neutral (2)
    * pos (3)

    Also calculates is (1)/is not (0) target labels.

    Phrases are under 'words' column
    Sentiment Labels are under 'polarity' column
    Is/Is Not Labels are under 'is_target' column

    Also includes `category` column, which indicates the target's aspect
     '''
    words = []
    for _, review in review_data.iterrows():
        rev = review['review']
        rev = ' '.join(rev.split())
        r = convert_word_char_indices(rev) # need starting index of each word
        word_windows = [pull_target(rev, i, n=context_window_size) for i in r]
        
        targets = [t for t in review['opinion_data']['target']]
        sentiments = [_sentiment_to_int(s) for s in review['opinion_data']['polarity']]
        char_starts = [c_i for c_i in review['opinion_data']['from']]
        categories = [c for c in review['opinion_data']['category']]
        
        # Get target word indices
        word_starts = []
        for t, s, c_i, c in zip(targets, sentiments, char_starts, categories):
            # Skip 'NULL' targets, since no word in sentence is a target
            if t == 'NULL': continue
            count = 0
            for w_i, word in enumerate(rev.split(' ')):
                if len(word) + count < c_i: count += len(word) + 1
                else: word_starts.append((w_i, s, c)); break

        # Assign labels
        labels = np.zeros(len(word_windows))
        categories = ['NULL'] * len(word_windows)
        # 0 = no sentiment, 1 = negative, 2 = neutral, 3 = positive
        for w_i, s, c in word_starts: 
            labels[w_i] = s + 2
            categories[w_i] = c

        df = pd.DataFrame(list(zip(word_windows, labels, categories)), columns=('words', 'polarity', 'category'))
        words.append(df)
        
    sentiment_data = pd.concat(words, ignore_index=True)
    sentiment_data['is_target'] = (sentiment_data['polarity'] > 0).astype(int)
    return sentiment_data

def pull_all_words_targets(review_data, context_window_size):
    ''' 
    Target words at center of each phrase, labels are is target (1) and is not target (0)

    Phrases are under `words` column \\
    Target labels are under `is_target` column \\
    Each target's aspects are under the `category` column    
    '''
    words = []
    for _, review in review_data.iterrows():
        rev = review['review']
        rev = ' '.join(rev.split())
        r = convert_word_char_indices(rev) # need starting index of each word
        word_windows = [pull_target(rev, i, n=context_window_size) for i in r]
        
        targets = [t for t in review['opinion_data']['target']]
        char_starts = [c_i for c_i in review['opinion_data']['from']]
        categories = [c for c in review['opinion_data']['category']]
        
        # Get target word indices
        word_starts = []
        for t, c_i, c in zip(targets, char_starts, categories):
            # Skip 'NULL' targets, since no word in sentence is a target
            if t == 'NULL': continue
            count = 0
            for w_i, word in enumerate(rev.split(' ')):
                if len(word) + count < c_i: count += len(word) + 1
                else: word_starts.append((w_i, c)); break

        # Assign labels
        labels = np.zeros(len(word_windows))
        categories = ['NULL'] * len(word_windows)
        for w_i, c in word_starts: 
            labels[w_i] = 1
            categories[w_i] = c
        df = pd.DataFrame(list(zip(word_windows, labels, categories)), columns=('words', 'is_target', 'category'))
        words.append(df)
        
    targets_df = pd.concat(words, ignore_index=True)
    return targets_df

def convert_word_char_indices(sentence):
    ''' 
    Takes a sentence (`str`) as input.

    Expects sentence with normalized spacing. 

    Returns a list of indices indicating the index of the 
    starting character of each word in the sentence, in order.
    '''
    char_counter = 1
    word_starts = [0]
    for i in range(1, len(sentence)):
        if sentence[i-1] == ' ': word_starts.append(char_counter)
        char_counter += 1
    return word_starts