from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate

from absa_code.models.models import DeepNeuralNetworkTargetPredictor

class SingleInputTargetLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(1028)(in_layer)
        d1 = Dense(512, activation='relu')(lstm_layer)
        d2 = Dense(128, activation='relu')(d1)
        out_layer = Dense(self._output_size, activation='softmax')(d2)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model

class DualInputOutputTargetLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(1028)(in_layer)
        d1 = Dense(512, activation='relu')(lstm_layer)
        d2 = Dense(128, activation='relu')(d1)
        aux_out = Dense(self._output_size, activation='softmax', name='aux_out')(d2)

        aux_in = Input(shape=(1,), name='pos_in')
        cat = Concatenate(axis=-1)([d2, aux_in])
        d3 = Dense(1024, activation='relu')(cat)
        main_out = Dense(self._output_size, activation='softmax', name='main_out')(d3)

        model = Model([in_layer, aux_in], [main_out, aux_out])
        model.compile(
            optimizer=self._optimizer,
            loss=self._model_loss,
            loss_weights=self._loss_weights
        )
        return model

class SimpleSingleInputTargetLSTM(DeepNeuralNetworkTargetPredictor):
    def _compile_model(self):
        in_layer = Input(shape=(self._n, self._emb_dim))
        lstm_layer = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(in_layer)
        out_layer = Dense(self._output_size, activation='softmax')(lstm_layer)

        model= Model(in_layer, out_layer)
        model.compile(
            optimizer=self._optimizer, 
            loss=self._model_loss,
            loss_weights=self._loss_weights)
        return model