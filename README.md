*Note: This README is still being added to and improved.*

# Introduction

This repository contains the code I wrote for my senior thesis as a Computer Science student at Columbia University. My thesis was focused on developing a semi-supervised model to find and label aspects in a corpus of text, part of the Aspect-Based Sentiment Analysis task. In my thesis, I broke aspect identification into two steps:

1. Aspect Detection
2. Aspect Identification

Aspect detection is a binary task, where a word is labeled 1, if it indicates an aspect, or 0, if it doesn't. Words the indicate an aspect are called "targets" from here on. Aspect identification is a multi-class problem, where each identified target is assigned to an aspect. The code in this `absa_code` provides a series of supervised models for aspect detection and a few basic clustering approaches for aspect identification.

`absa_code` also contains a series of functions meant to help with parsing the [SemEval 2016 ABSA dataset](http://alt.qcri.org/semeval2016/task5/), with analyzing and visualizing results from the various models, and with various preprocessing tasks. The supervised models all depend on pre-trained word embeddings. For my work, I used pretrained GloVe word embeddings, which can be found [here](https://nlp.stanford.edu/projects/glove/).

# Setup

## Requirements

The following packages are required to run the code in this repository:

* Python 3.6 or higher
* Pandas
* numPy
* scikit-learn
* feedparser
* spaCy (including the `web_core_lg` module)
* textacy
* Tensorflow
* Keras

The code in this repository also requires pre-trained word embeddings. By default, the code will look for a folder called `embedding_data` at the top level of this repository for pre-trained word embeddings. Of course, the code also requires a dataset to parse and train on. Here, the default location is a folder called `data` in the top level of this repository. Code is available in `absa_code/preprocessing/absa_parsing.py` that automatically parses the .xml data files from SemEval 2016. This code is called from `model_testing.py`, so you will need to replace those calls if another dataset is used.

## Running the Code

The code in `absa_code` is meant to be used as a module. The functions in each of the files contained in `absa_code` are mostly documented with their expected input, output, and purpose.

To make sure the code is working properly, running `model_testing.py` should work immediately. The code under `if name == '__main__'` can be used as an example of the functionality contained in `absa_code`. `model_testing.py` also contains a series of helper functions to streamline the process of preprocessing the data and training and testing models.