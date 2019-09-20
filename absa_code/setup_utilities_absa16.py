import pandas as pd
import numpy as np

from .preprocessing import absa_parsing as ap
from .preprocessing import data_preprocessing as dp
from .preprocessing import embedding_generation as eg
from .preprocessing import text_processing as tp

'''
The code in this file is meant to simplify running experiments on the ABSA16 dataset.

As such, the functions defined in this file assume a specific structure, discussed in comments,
in the data, and may not work when not used in a particular manner.

Check the code in `if name==__main__` for an example of how to use these functions to setup
for training an ABSA model.
'''

def setup_absa16(cws, sentiment=False, sampling_type='default', rate=0.2):
    '''
    Returns parsed data from the ABSA16 dataset and corresponding onehot labels

    `type` must be:

    - default: returns the default targets data
    - upsample: returns an upsampled version of the targets data
    - downsample: returns a downsampled version of the targets data

    If either upsample or downsample are provided, `rate` defines the rate at 
    which the data will be upsampled or downsampled
    '''
    review_data, _, targets_df = ap.extract_data_ABSA16(cws=cws, include_sentiment=sentiment, filepath='./data/ABSA16_Restaurants_Train_SB1_v2.xml')
    
    # if sampling_type == 'default': data_df = targets_df
    # elif sampling_type == 'upsample': data_df = dp.upsample(targets_df, 'is_target', 1, upsample_rate=rate)
    # elif sampling_type == 'downsample': data_df = dp.downsample(targets_df, 'is_target', 0, downsample_rate=rate)
    # else: raise ValueError('The type parameter must be default, upsample, or downsample')
    data_df = targets_df

    target_onehot = pd.get_dummies(data_df['is_target']).values

    return review_data, data_df, target_onehot

def setup_embeddings(data_df, embedding_dim=100):
    '''
    Calculates embeddings for text in `data_df` and the part of speech tags of each potential target word

    `data_df` must have a feature with name `words`
    
    `embedding_dim` must be 50, 100, 200, or 300
    '''
    target_matrix = eg.generate_ngram_matrix(data_df['words'], emb_dim=embedding_dim, glove_path='./embedding_data/')
    target_vectors = eg.flatten_sentence_vectors(target_matrix)
    pos_tags = tp.generate_pos_tags(data_df['words'])

    return target_matrix, target_vectors, pos_tags

def handle_sampling(targets_df, sampling_type, rate=0.2):
    if sampling_type == 'default': data_df = targets_df
    elif sampling_type == 'upsample': data_df = dp.upsample(targets_df, 'is_target', 1, upsample_rate=rate)
    elif sampling_type == 'downsample': data_df = dp.downsample(targets_df, 'is_target', 0, downsample_rate=rate)
    else: raise ValueError('The type parameter must be default, upsample, or downsample')

    target_onehot = pd.get_dummies(data_df['is_target']).values

    return data_df, target_onehot


if __name__ == '__main__':
    # This code show how to use the functions in this file to 
    # setup the ABSA16 dataset for training a model.
    from .models import data_preparation as prep

    cws = 5
    train_test_split = 0.2
    sampling_type = 'upsample'
    sample_rate = 0.4

    # When running multiple tests, _only_ `prep.prepare_data()` needs to be used every run
    review_data, data_df, onehot_labels = setup_absa16(5, sampling_type=sampling_type, rate=sample_rate)
    # target_matrix, target_vectors, pos_tags = setup_embeddings(data_df, embedding_dim=100)
    train_test_data = prep.prepare_data(data_df, train_test_split)

    train_df = train_test_data[0]
    test_df = train_test_data[1]

    train_df, train_labels = handle_sampling(train_df, sampling_type, rate=sample_rate)
    test_labels = pd.get_dummies(data_df['is_target']).values

    train_matrix, train_vectors, train_pos_tags = setup_embeddings(train_df, embedding_dim=100)
    test_matrix, test_vectors, test_pos_tags = setup_embeddings(test_df, embedding_dim=100)

    # train_test_data = prep.prepare_data(data_df, train_test_split, target_matrix, target_vectors, onehot_labels, pos_tags)

    # Unpack split data
    train_df = train_test_data[0]
    test_df = train_test_data[1]

    # train_matrix = train_test_data[2]
    # test_matrix = train_test_data[3]

    # train_vectors = train_test_data[4]
    # test_vectors = train_test_data[5]

    # train_labels = train_test_data[6]
    # test_labels = train_test_data[7]

    # train_pos_tags = train_test_data[8]
    # test_pos_tags = train_test_data[9]

    print(train_df.head())
    # From here, we have training and validation/testing data available for an ABSA model
    # Look in absa_code.models for pre-built models and methods to construct custom models
