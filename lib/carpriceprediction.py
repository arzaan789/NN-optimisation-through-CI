import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def load_data_(file_specifier):
    if file_specifier not in ['train','validation','test']:
        raise Exception("Cannot load file based on specifier '{}'. The only acceptable  file specifiers are 'train','validation','test'".format(file_specifier))
    data = pd.read_csv('./data/{}.csv'.format(file_specifier),header=None)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return X, y

def construct_regressor_(params):
    DATA_DIM = 21
    HIDDEN_LAYER_SIZE = 2
    N_OUTPUTS = 1

    reg = MLPRegressor(hidden_layer_sizes=(HIDDEN_LAYER_SIZE))
    reg.n_features_in_ = DATA_DIM
    reg.n_layers_ = 3
    reg.n_outputs_ = N_OUTPUTS
    reg.out_activation_ = 'identity'

    first_weights_size = DATA_DIM * HIDDEN_LAYER_SIZE
    second_weights_size = HIDDEN_LAYER_SIZE * N_OUTPUTS
    frm = 0; to = frm + first_weights_size; layer_1_weights = params[frm:to].reshape(-1,HIDDEN_LAYER_SIZE)
    frm = to; to = frm + second_weights_size; layer_2_weights = params[frm:to].reshape(HIDDEN_LAYER_SIZE,-1)
    coefs = [layer_1_weights,layer_2_weights]
    reg.coefs_ = coefs
    
    first_bias_size = HIDDEN_LAYER_SIZE
    second_bias_size = N_OUTPUTS
    frm = to; to = frm + first_bias_size; layer_1_bias = params[frm:to]
    frm = to; to = frm + second_bias_size; layer_2_bias = np.array(params[frm:to])
    intercepts = [layer_1_bias,layer_2_bias]
    reg.intercepts_ = intercepts
    
    return reg


def modified_construct_regressor(params):
    # Extract architecture parameters
    n_layers = int(params[0])
    hidden_units = [int(params[i]) for i in range(1, n_layers + 1)]
    params = params[n_layers + 1:]  # Remaining are weights and biases

    # Initialize MLP (dummy initialization)
    reg = MLPRegressor(hidden_layer_sizes=tuple(hidden_units))
    reg.n_features_in_ = 21  # Fixed input size
    reg.n_outputs_ = 1       # Single output
    reg.n_layers_ = n_layers + 2  # Input + Hidden + Output
    reg.out_activation_ = 'identity'

    # Split weights and biases
    coefs = []
    intercepts = []
    param_ptr = 0  # Current position in params array

    # Input → First Hidden Layer
    input_dim = 21
    hidden_dim = hidden_units[0]
    coefs.append(params[param_ptr:param_ptr + input_dim * hidden_dim].reshape(input_dim, hidden_dim))
    param_ptr += input_dim * hidden_dim

    # Hidden → Hidden Layers
    for i in range(n_layers - 1):
        input_dim = hidden_units[i]
        output_dim = hidden_units[i + 1]
        coefs.append(params[param_ptr:param_ptr + input_dim * output_dim].reshape(input_dim, output_dim))
        param_ptr += input_dim * output_dim

    # Last Hidden → Output
    input_dim = hidden_units[-1]
    coefs.append(params[param_ptr:param_ptr + input_dim * 1].reshape(input_dim, 1))
    param_ptr += input_dim * 1

    # Biases (hidden layers first, then output)
    for units in hidden_units:
        intercepts.append(params[param_ptr:param_ptr + units])
        param_ptr += units
    intercepts.append(params[param_ptr:param_ptr + 1])  # Output bias

    # Assign to regressor
    reg.coefs_ = coefs
    reg.intercepts_ = intercepts

    return reg

def evaluate_(X,y,params):
    reg = construct_regressor_(params)
    y_pred = reg.predict(X)
    return mean_squared_error(y,y_pred)

def modified_evaluate_(X, y, params):
    reg = modified_construct_regressor(params)
    y_pred = reg.predict(X)
    return mean_squared_error(y, y_pred)

def make_evaluator(file_specifier):
    X,y = load_data_(file_specifier)
    return lambda params : evaluate_(X,y,params)

def modified_make_evaluator(file_specifier):
    X, y = load_data_(file_specifier)
    return lambda params: modified_evaluate_(X, y, params)