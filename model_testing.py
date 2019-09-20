from absa_code.setup_utilities_absa16 import *
from absa_code.models import convolutional_models as cm
from absa_code.models import lstm_models as lm
from absa_code.models import deep_models as nn
from absa_code.models.models import SKLearnSentimentPredictor as SVM
from absa_code.models import keras_callbacks as callbacks
from absa_code.analysis import result_handling as rh

from multiprocessing import Pool
import json


def test_model_avg(data_df, target_data_tup, pos_tags, train_test_split, model_builder, model_tester, runs=2, epochs=5, num_p=4, filename=None):
    '''
    This function will automatically train a model type for a number of times (`runs`)

    `runs` defines the number of times a model architecture is trained \\
    `model_tester` should a tester function that works with `model_builder`
    '''
    target_matrix = target_data_tup[0]
    target_vectors = target_data_tup[1]

    res = []
    for _ in range(runs):
        res.append(model_tester(data_df, (target_matrix, target_vectors, onehot_labels), pos_tags, train_test_split, model_builder, epochs=5))

    if filename != None:
        f = open(filename, 'w+')
        json.dump([r.history for r in res], f)
        f.close()
        
    return res


def test_simple_model(train_test_df, data_tup, pos_tags, model_builder, epochs=5):
    '''
    This function offers a convenient way to test a keras model,
    passed in via `model_builder`, that meets the following requirements:

    * the model is a binary classifier
    * single input
    * single softmax output (shape = (?, 2))

    `data_df` must be a `pd.DataFrame` object that can be given to `prep.prepare_data()` \\ TODO update this
    `target_data_tup` should be a tuple containing `target_matrix`, `target_vectors`, and `target_onehot`
    Again, see `prep.prepare_data()` for details on these individual parameters \\
    `pos_tags` must be a 1D iterable containing part-of-speech information for each
    entry in `data_df` \\
    `model_builder` should be the initializer of a model that meets the above requirements
    '''
    from absa_code.models import data_preparation as prep
    
    train_df = train_test_df[0]
    test_df = train_test_df[1]

    train_data = data_tup[0]
    test_data = data_tup[1]
    train_labels = data_tup[2]
    test_labels = data_tup[3]

    # Calculate Parameters
    cws = int((train_data.shape[1] - 1) / 2)
    label_len = train_labels.shape[1]
    embedding_dim = train_data.shape[2]

    # Train Model
    model = model_builder(cws, embedding_dim, label_len)

    res = model.train(train_data, train_labels, test_data, test_labels, e=epochs, callbacks=[callbacks.BinaryClassCallback()])
    return res.history

def test_ml_model(C, data_df, target_data_tup, pos_tags, train_test_split, model_builder, epochs=5):
    '''
    This function offers a convenient way to test a keras model,
    passed in via `model_builder`, that meets the following requirements:

    * the model is a binary classifier
    * single input
    * single softmax output (shape = (?, 2))

    `data_df` must be a `pd.DataFrame` object that can be given to `prep.prepare_data()` \\
    `target_data_tup` should be a tuple containing `target_matrix`, `target_vectors`, and `target_onehot`
    Again, see `prep.prepare_data()` for details on these individual parameters \\
    `pos_tags` must be a 1D iterable containing part-of-speech information for each
    entry in `data_df` \\
    `model_builder` should be the initializer of a model that meets the above requirements

    '''
    from absa_code.models import data_preparation as prep
    
    # Unwrap `target_data_tup`
    target_matrix = target_data_tup[0]
    target_vectors = target_data_tup[1]
    target_onehot = target_data_tup[2]

    train_test_data = prep.prepare_data(data_df, train_test_split, target_matrix, target_vectors, target_onehot, pos_tags)

    # Calculate Parameters
    cws = int((target_matrix.shape[1] - 1) / 2)
    label_len = target_onehot.shape[1]
    embedding_dim = target_matrix.shape[2]

    # Unpack split data
    train_df = train_test_data[0]
    test_df = train_test_data[1]

    train_matrix = train_test_data[2]
    test_matrix = train_test_data[3]

    train_vectors = train_test_data[4]
    test_vectors = train_test_data[5]

    train_labels = train_test_data[6]
    test_labels = train_test_data[7]

    train_pos_tags = train_test_data[8]
    test_pos_tags = train_test_data[9]

    # Train Model
    model = model_builder(C)
    res = model.train(train_vectors, np.argmax(train_labels, axis=1))
    return dict(zip(('precision', 'recall', 'f_score'), model.test(test_vectors, np.argmax(test_labels, axis=1))))

def test_dual_input_model(data_df, target_data_tup, pos_tags, train_test_split, model_builder, epochs=5):
    from absa_code.models import data_preparation as prep

    # Unwrap `target_data_tup`
    target_matrix = target_data_tup[0]
    target_vectors = target_data_tup[1]
    target_onehot = target_data_tup[2]

    train_test_data = prep.prepare_data(data_df, train_test_split, target_matrix, target_vectors, target_onehot, pos_tags)

    # Calculate Parameters
    cws = int((target_matrix.shape[1] - 1) / 2)
    label_len = target_onehot.shape[1]
    embedding_dim = target_matrix.shape[2]

    # Unpack split data
    train_df = train_test_data[0]
    test_df = train_test_data[1]

    train_matrix = train_test_data[2]
    test_matrix = train_test_data[3]

    train_vectors = train_test_data[4]
    test_vectors = train_test_data[5]

    train_labels = train_test_data[6]
    test_labels = train_test_data[7]

    train_pos_tags = train_test_data[8]
    test_pos_tags = train_test_data[9]

    # Train Model
    model = model_builder(cws, embedding_dim, label_len)
    res = model.train([train_matrix, train_pos_tags], train_labels, [test_matrix, test_pos_tags], test_labels, e=epochs, callbacks=[callbacks.BinaryClassMultiInputCallback()])
    return res.history

def test_dual_input_output_model(data_df, target_data_tup, pos_tags, train_test_split, model_builder, epochs=5):
    from absa_code.models import data_preparation as prep

    # Unwrap `target_data_tup`
    target_matrix = target_data_tup[0]
    target_vectors = target_data_tup[1]
    target_onehot = target_data_tup[2]

    train_test_data = prep.prepare_data(data_df, train_test_split, target_matrix, target_vectors, target_onehot, pos_tags)

    # Calculate Parameters
    cws = int((target_matrix.shape[1] - 1) / 2)
    label_len = target_onehot.shape[1]
    embedding_dim = target_matrix.shape[2]

    # Unpack split data
    train_df = train_test_data[0]
    test_df = train_test_data[1]

    train_matrix = train_test_data[2]
    test_matrix = train_test_data[3]

    train_vectors = train_test_data[4]
    test_vectors = train_test_data[5]

    train_labels = train_test_data[6]
    test_labels = train_test_data[7]

    train_pos_tags = train_test_data[8]
    test_pos_tags = train_test_data[9]

    # Train Model
    model = model_builder(cws, embedding_dim, label_len)
    res = model.train([train_matrix, train_pos_tags], [train_labels, train_labels], [test_matrix, test_pos_tags], [test_labels, test_labels], e=epochs, callbacks=[callbacks.BinaryClassMultiInputOutputCallback()])
    return res.history

if __name__=='__main__':
    from absa_code.models import data_preparation as prep
    # Testing for Optimal Sampling value
    train_test_split = 0.2
    e = 200
    save_dir = './results/sampling_tests/'
    for sampling_type in ['upsample', 'downsample']:
        for i in range(2, 9, 2):
            sampling_rate = float(i) / 10
            ###############
            # Simple Models
            ###############
            
            # Simple LSTM
            cws = 9
            review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            train_test_data = prep.prepare_data(data_df, train_test_split, onehot_labels)

            train_df = train_test_data[0]
            test_df = train_test_data[1]
            test_labels = train_test_data[3]

            train_df, train_labels = handle_sampling(train_df, sampling_type, rate=sampling_rate)

            train_matrix, train_vectors, train_pos_tags = setup_embeddings(train_df, embedding_dim=100)
            test_matrix, test_vectors, test_pos_tags = setup_embeddings(test_df, embedding_dim=100)
            simple_lstm_res = test_simple_model((train_df, test_df), (train_matrix, test_matrix, train_labels, test_labels), train_test_split, lm.SimpleSingleInputTargetLSTM, epochs=e)
            # Deep Net
            cws = 7
            review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            train_test_data = prep.prepare_data(data_df, train_test_split, onehot_labels)

            train_df = train_test_data[0]
            test_df = train_test_data[1]
            test_labels = train_test_data[3]

            train_df, train_labels = handle_sampling(train_df, sampling_type, rate=sampling_rate)

            train_matrix, train_vectors, train_pos_tags = setup_embeddings(train_df, embedding_dim=100)
            test_matrix, test_vectors, test_pos_tags = setup_embeddings(test_df, embedding_dim=100)
            dn_res = test_simple_model((train_df, test_df), (train_matrix, test_matrix, train_labels, test_labels), train_test_split, nn.SingleInputTargetNetwork, epochs=e)

            # LSTM
            cws = 5
            review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            train_test_data = prep.prepare_data(data_df, train_test_split, onehot_labels)

            train_df = train_test_data[0]
            test_df = train_test_data[1]
            test_labels = train_test_data[3]

            train_df, train_labels = handle_sampling(train_df, sampling_type, rate=sampling_rate)

            train_matrix, train_vectors, train_pos_tags = setup_embeddings(train_df, embedding_dim=100)
            test_matrix, test_vectors, test_pos_tags = setup_embeddings(test_df, embedding_dim=100)
            lstm_res = test_simple_model((train_df, test_df), (train_matrix, test_matrix, train_labels, test_labels), train_test_split, lm.SingleInputTargetLSTM, epochs=e)

            # CNN
            cws = 3
            review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            train_test_data = prep.prepare_data(data_df, train_test_split, onehot_labels)

            train_df = train_test_data[0]
            test_df = train_test_data[1]
            test_labels = train_test_data[3]

            train_df, train_labels = handle_sampling(train_df, sampling_type, rate=sampling_rate)

            train_matrix, train_vectors, train_pos_tags = setup_embeddings(train_df, embedding_dim=100)
            test_matrix, test_vectors, test_pos_tags = setup_embeddings(test_df, embedding_dim=100)
            cnn_res = test_simple_model((train_df, test_df), (train_matrix, test_matrix, train_labels, test_labels), train_test_split, cm.SingleInputTargetPredictor, epochs=e)

            # #############
            # # Dual Models
            # #############
            
            # # LSTM
            # cws = 11
            # review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            # target_matrix, target_vectors, pos_tags = setup_embeddings(data_df, embedding_dim=100)
            # dual_lstm_res = test_dual_input_output_model(data_df, (target_matrix, target_vectors, onehot_labels), pos_tags, train_test_split, lm.DualInputOutputTargetLSTM, epochs=e)

            # # CNN
            # cws = 3
            # review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            # target_matrix, target_vectors, pos_tags = setup_embeddings(data_df, embedding_dim=100)
            # dual_cnn_res = test_dual_input_output_model(data_df, (target_matrix, target_vectors, onehot_labels), pos_tags, train_test_split, cm.DualInputLossTargetPredictor, epochs=e)

            # # Deep Net
            # cws = 3
            # review_data, data_df, onehot_labels = setup_absa16(cws, sampling_type=sampling_type, rate=sampling_rate)
            # target_matrix, target_vectors, pos_tags = setup_embeddings(data_df, embedding_dim=100)
            # dual_dn_res = test_dual_input_output_model(data_df, (target_matrix, target_vectors, onehot_labels), pos_tags, train_test_split, nn.DualInputOutputTargetNetwork, epochs=e)


            ##############
            # Save Results
            ##############

            # Simple Models

            f = open(save_dir + 'simple_lstm-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            json.dump(simple_lstm_res, f)
            f.close()

            f = open(save_dir + 'dn-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            json.dump(dn_res, f)
            f.close()

            f = open(save_dir + 'lstm-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            json.dump(lstm_res, f)
            f.close()

            f = open(save_dir + 'cnn-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            json.dump(cnn_res, f)
            f.close()

            # Dual Input Output Models

            # f = open(save_dir + '08_01-dual_lstm-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            # json.dump(dual_lstm_res, f)
            # f.close()

            # f = open(save_dir + '08_01-dual_cnn-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            # json.dump(dual_cnn_res, f)
            # f.close()

            # f = open(save_dir + '08_01-dual_dn-{}_{}.json'.format(sampling_type, sampling_rate), 'w+')
            # json.dump(dual_dn_res, f)
            # f.close()