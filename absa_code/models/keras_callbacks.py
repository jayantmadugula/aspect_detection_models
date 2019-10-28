import numpy as np
from tensorflow.keras.callbacks import Callback

from absa_code.analysis import model_performance_metrics as pm 

# Callbacks written for Keras models
# to calculate and record a model's 
# precision, recall, and f-score while training.

class BinaryClassCallback(Callback):
    '''
    This callback should be used for a model with
    a single input and a single `softmax` output that
    represents a binary decision.
    '''
    def on_epoch_end(self, epoch, logs={}):
        # use argmax to get numeric label
        preds = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        labels = np.argmax(self.validation_data[1], axis=1)

        # calculate metrics at current epoch
        prec = pm.calculate_class_precision(1, preds, labels)
        rec = pm.calculate_class_recall(1, preds, labels)
        fscore = pm.calculate_class_fscore(prec, rec)

        # log metrics
        logs['val_precision'] = prec
        logs['val_recall'] = rec
        logs['val_fscore'] = fscore

        # format and print during training
        print('\nPrecision: {}'.format(prec))
        print('\nRecall: {}'.format(rec))
        print('\nF-Score: {}'.format(fscore))
        print()

class BinaryClassMultiInputCallback(Callback):
    '''
    This callback should be used for a model with
    two inputs and a single `softmax` output that
    represents a binary decision.
    '''
    def on_epoch_end(self, epoch, logs={}):
        # use argmax to get numeric label
        preds = np.argmax(self.model.predict([self.validation_data[0], self.validation_data[1]]), axis=1)
        labels = np.argmax(self.validation_data[2], axis=1)

        # calculate metrics at current epoch
        prec = pm.calculate_class_precision(1, preds, labels)
        rec = pm.calculate_class_recall(1, preds, labels)
        fscore = pm.calculate_class_fscore(prec, rec)

        # log metrics
        logs['val_precision'] = prec
        logs['val_recall'] = rec
        logs['val_fscore'] = fscore

        # format and print during training
        print('\nPrecision: {}'.format(prec))
        print('\nRecall: {}'.format(rec))
        print('\nF-Score: {}'.format(fscore))
        print()

class BinaryClassMultiInputOutputCallback(Callback):
    '''
    This callback should be used for models with
    two inputs and two `softmax` outputs, each
    representing a binary decision.
    '''
    def on_epoch_end(self, epoch, logs={}):
        # use argmax to get numeric labels
        preds_all = self.model.predict([self.validation_data[0], self.validation_data[1]])
        preds_main = np.argmax(preds_all[0], axis=1) # output from model's main output
        preds_aux = np.argmax(preds_all[1], axis=1) # output from model's auxilliary output
        labels = np.argmax(self.validation_data[2], axis=1)

        # calculate metrics for both outputs at current epoch
        prec_main = pm.calculate_class_precision(1, preds_main, labels) # TODO FIX THIS
        rec_main = pm.calculate_class_recall(1, preds_main, labels)
        fscore_main = pm.calculate_class_fscore(prec_main, rec_main)

        prec_aux = pm.calculate_class_precision(1, preds_aux, labels)
        rec_aux = pm.calculate_class_recall(1, preds_aux, labels)
        fscore_aux = pm.calculate_class_fscore(prec_aux, rec_aux)

        # log metrics
        logs['val_precision_main'] = prec_main
        logs['val_recall_main'] = rec_main
        logs['val_fscore_main'] = fscore_main

        logs['val_precision_aux'] = prec_aux
        logs['val_recall_aux'] = rec_aux
        logs['val_fscore_aux'] = fscore_aux

        # format and print during training
        print('\nMain Precision: {}, Aux Precision: {}'.format(prec_main, prec_aux))
        print('\nMain Recall: {}, Aux Recall: {}'.format(rec_main, rec_aux))
        print('\nMain F-Score: {}, Aux F-Score: {}'.format(fscore_main, fscore_aux))
        print()