import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import json

APRCH_PATH = './results/approach_test/'
CWT_PATH = './results/context_window_test/'

# Training Output Parsing
def parse_approach_test_results(filenames):
    '''
    Parses results from an architecture test given result filenames.


    Expects first level of results to contain values.
    '''
    results = {}
    for fn in filenames:
        f = open(APRCH_PATH + fn)
        fj = json.load(f)
        
        # Get all data
        vloss = fj['val_loss']
        vacc = fj['val_acc']
        loss = fj['loss']
        acc = fj['acc']
        vprec = np.stack(fj['val_precision'])
        vrec = np.stack(fj['val_recall'])
        
        results[fn] = (vloss, vacc, loss, acc, vprec, vrec)

        f.close()
    
    return results

def parse_param_test_results(filenames, bot, top, step=1):
    '''
    Parses `json` results where first level is param number and second level is values.
    '''
    results = {}
    for fn in filenames:
        f = open(CWT_PATH + fn)
        fj = json.load(f)
        for i in range(bot, top, step):
            # Get all data
            vloss = fj[str(i)]['val_loss']
            vacc = fj[str(i)]['val_acc']
            loss = fj[str(i)]['loss']
            acc = fj[str(i)]['acc']
            vprec = np.stack(fj[str(i)]['val_precision'])
            vrec = np.stack(fj[str(i)]['val_recall'])
            
            results[fn + str(i)] = (vloss, vacc, loss, acc, vprec, vrec)
        f.close()

    return results

def pull_approach_results(results, param_name, filenames):
    ''' 
    Returns list of values from architecture test results for given `param_name`.

    Expects `results` to be from `parse_approach_test_results`.
    '''
    i = ('vloss', 'vacc', 'loss', 'acc', 'vprec', 'vrec').index(param_name)

    res = []
    for fn in filenames:
        res.append(results[fn][i])

    return res

def pull_param_test_results(results, param_name, val_f):
    '''
    Returns list of values from parameter testing results for given `param_name`.

    Expects `results` to be from `parse_param_test_results`.
    '''
    i = ('vloss', 'vacc', 'loss', 'acc', 'vprec', 'vrec').index(param_name)

    res = []
    for _, v in results.items():
        res.append(val_f(v[i]))

    return res

# Graphing Functions
def graph_approach_results(results, labels, y_label, title, x_range=(0,500)):
    x = np.arange(x_range[0], x_range[1])
    j = x_range[1]

    for y, l in zip(results, labels):
        plt.plot(x, y[:j], label=l)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend(labels)
    plt.show()

def graph_approach_custom_results(results, labels, y_label, title, x_range=(0,500)):
    x = np.arange(x_range[0], x_range[1])
    j = x_range[1]

    legend_labels = []

    for y, l in zip(results, labels):
        y_arr = np.array(y[:j])
        for i in range(y_arr.shape[1]):
            plt.plot(x, y_arr[:,i], label=l + ' Class {}'.format(i))
            legend_labels.append(l + ' Class {}'.format(i))

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend(legend_labels)
    plt.show()

def graph_param_test_results(results, labels, y_label, title, param_name):
    h = int(len(results) / 2)
    x = np.arange(0, h)
    
    plt.plot(x, results[:h], label=labels[0])
    plt.plot(x, results[h:], label=labels[1])

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(y_label)
    plt.legend(labels)
    plt.show()

    # Graphing and Visualization
def plot_keras_history(history, filename):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filename)

def plot_keras_(history, filename, metric_desc, num_classes):
    pass 
    #TODO plot each label as a line on the plot (so num_classes == lines on plot)

    
# https://stackoverflow.com/questions/24120023/strange-error-with-matplotlib-axes-labels

aprch_filenames = [
    'binary-simple_lstm.json',
    'binary-three_layer_cnn.json',
    'combined-simple_lstm.json',
    'combined-three_layer_cnn.json',
    'sentiment-simple_lstm.json',
    'sentiment-three_layer_cnn.json'
]

aprch_labels = [
    'Binary Simple LSTM',
    'Binary Three Layer CNN',
    'Combined Simple LSTM',
    'Combined Three Layer CNN',
    'Sentiment Simple LSTM',
    'Sentiment Three Layer CNN'
]

cwt_filenames = [
    'binary-three_layer_cnn.json',
    'combined-three_layer_cnn.json'
]

cwt_labels = [
    'Binary Three Layer CNN',
    'Combined Three Layer CNN'
]


if __name__ == '__main__':
    # For Fall 2018 Tests
    a_res = parse_approach_test_results(aprch_filenames)
    vloss_a_res = pull_approach_results(a_res, 'vloss', aprch_filenames)
    graph_approach_results(vloss_a_res, aprch_labels, 'Loss', 'Validation Loss over Training Epochs')

    cwt_res = parse_param_test_results(cwt_filenames, 0, 3)
    vloss_cwt_res = pull_param_test_results(cwt_res, 'vloss', min)
    graph_param_test_results(vloss_cwt_res, cwt_labels, 'Loss', 'Min Loss over 500 Epochs for Context Window Size', 'Window Size')