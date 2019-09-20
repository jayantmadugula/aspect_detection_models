import numpy as np
import pandas as pd

def downsample(data, label, majority_label, downsample_rate=0.2):
    '''
    Constructs a downsampled version of the given iterable without regard
    for contents of `data`

    Rows with the majority label are randomly removed at the `downsample_rate`

    Input:

    * `data`: a multi-dimensional labelled data structure, such as a pandas DataFrame
    * `label`: a valid index in `data` containing information about each row's label
    * `majority_label`: rows with this label will be considered for downsampling
    * `downsample_rate`: percentage of data with the `majority_label` to be removed.

    '''
    majority_indices = data[data[label]==majority_label].index.values
    ds_size = int(majority_indices.shape[0] * downsample_rate)
    drop_inds = np.random.choice(majority_indices, size=ds_size, replace=False)
    return data.drop(index=drop_inds)

def upsample(data, label, minority_label, upsample_rate=0.2):
    '''
    Constructs an upsampled version of the given iterable without regard
    for contents of `data`

    Rows with the minority label are randomly duplicated at the `upsample_rate`
    but the outputted data structure is NOT shuffled

    Input:

    * `data`: a multi-dimensional labelled data structure, such as a pandas DataFrame
    * `label`: a valid index in `data` containing information about each row's label
    * `majority_label`: rows with this label will be considered for upsampling
    * `upsample_rate`: percentage of data with the `minority_label` to be duplicated

    '''
    minority_indices = data[data[label]==minority_label].index.values
    us_size = int(minority_indices.shape[0] * upsample_rate)
    add_inds = np.random.choice(minority_indices, size=us_size, replace=False)
    new_rows = data.iloc[add_inds]
    return pd.concat([data, new_rows], ignore_index=True)