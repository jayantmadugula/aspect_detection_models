import json
'''
This file contains code to save, load, and read JSON files,
generally the output of training runs from Keras models
'''

def save_results(res, filename):
    f = open(filename, 'w+')
    json.dump(res, f)
    f.close()