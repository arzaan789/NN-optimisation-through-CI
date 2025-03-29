import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


def load_data_(file_specifier):
    if file_specifier not in ['train', 'validation', 'test']:
        raise Exception(
            "Cannot load file based on specifier '{}'. The only acceptable  file specifiers are 'train','validation','test'".format(
                file_specifier))
    data = pd.read_csv('./data/{}.csv'.format(file_specifier), header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

# Modified construct_regressor to handle dynamic architectures
def modified_construct_regressor(network, input_dim=21, output_dim=1):
    HIDDEN_LAYERS = network.layers[:-1]
    reg = MLPRegressor(hidden_layer_sizes=tuple(HIDDEN_LAYERS))
    reg.n_features_in_ = input_dim
    reg.n_layers_ = len(HIDDEN_LAYERS) + 2
    reg.n_outputs_ = output_dim
    reg.out_activation_ = 'identity'

    coefs = []
    intercepts = []

    for i in range(len(network.layers)):
        current_layer = network.layers[i]
        weight_matrix = np.array([neuron.weights for neuron in current_layer.neurons]).T
        coefs.append(weight_matrix)

        bias = np.array([neuron.bias for neuron in current_layer.neurons])
        intercepts.append(bias)

    reg.coefs_ = coefs
    reg.intercepts_ = intercepts

    return reg

def modified_evaluate_(X, y, params):
    reg = modified_construct_regressor(params)
    y_pred = reg.predict(X)
    return mean_squared_error(y, y_pred)

def modified_make_evaluator(file_specifier):
    X, y = load_data_(file_specifier)
    return lambda params: modified_evaluate_(X, y, params)
