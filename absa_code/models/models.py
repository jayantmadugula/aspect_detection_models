import numpy as np
from sklearn.svm import SVC
from absa_code.analysis import model_performance_metrics as pm


class SKLearnSentimentPredictor():
    def __init__(self, C, kernel='rbf', d=3):
        self._model = SVC(C=C, kernel=kernel, degree=d)

    def train(self, train_data, train_labels):
        self._model.fit(train_data, train_labels)

    def test(self, test_data, test_labels):
        preds = self._model.predict(test_data)
        prec = pm.calculate_class_precision(1, preds, test_labels)
        rec = pm.calculate_class_recall(1, preds, test_labels)
        fscore = pm.calculate_class_fscore(prec, rec)
        return prec, rec, fscore

    def predict(self, data):
        return self._model.predict(data)

class DeepNeuralNetworkTargetPredictor():
    '''
    This is a base class that does not contain a specific model architecture

    For specific models, look at other files in the `models` directory \\
    Implementing a new Keras model as a subclass requires completing `_compile_model`
    '''
    def __init__(self, cws, emb_dim, label_len, optimizer='adam', loss='categorical_crossentropy', loss_weights=None):
        self._n = 2 * cws + 1
        self._emb_dim = emb_dim
        self._optimizer = optimizer
        self._model_loss = loss
        self._loss_weights = loss_weights
        self._output_size = label_len
        self._model = self._compile_model()

    def _compile_model(self):
        ''' Builds model based on parameters '''
        raise NotImplementedError()

    def train(self, train_data, train_labels, val_data, val_labels, e=5, callbacks=None):
        ''' 
        Trains compiled Keras model on given data and labels

        By default, will not perform a full training run, make sure to change `e` \\
        Returns the result from Keras' `fit` function
        ''' 
        return self._model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=e,
            callbacks=callbacks)

    def test(self, test_data, test_labels, callbacks=None):
        '''
        Runs trained Keras model on given test data and returns metrics
        '''
        return self._model.evaluate(test_data, test_labels, callbacks=callbacks)

    def predict(self, data):
        ''' Returns trained model's predictions on given data '''
        return self._model.predict(data)