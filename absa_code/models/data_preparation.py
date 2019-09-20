import pandas as pd
import numpy as np
import math


def prepare_data(data_df, train_test_split, *args):
    '''
    Randomly splits the data according to the given
    `train_test_split` (0.2 means 80/20 train/test split).

    Only pandas DataFrames or numpy arrays may be provided
    in *args -- the same random split is applied to these
    as `data_df`
    '''
    num_samples = data_df.shape[0]
    train_size, test_size = calculate_train_test_split(
        num_samples, ratio=train_test_split)
    train_indices, test_indices = random_split_indices(
        num_samples, train_size)

    train_df = data_df.loc[train_indices].reset_index(drop=True)
    test_df = data_df.loc[test_indices].reset_index(drop=True)

    split_args = [train_df, test_df]
    for arr in args:
        split_args.append(arr[train_indices])
        split_args.append(arr[test_indices])

    return split_args


def calculate_train_test_split(num_samples, ratio=0.2):
    '''
    Given the number of samples and the train/test split, 
    this function will return the size of the training
    and testing datasets.
    '''
    train_size = math.ceil(num_samples * (1 - ratio))
    test_size = math.floor(num_samples * ratio)

    return train_size, test_size

def random_split_indices(num_samples, train_size):
    '''
    Given a data size and the size of the training
    data, this function will return
    two lists of indices, one for training data 
    and one for testing/validation data
    '''
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    return np.split(indices, [train_size])