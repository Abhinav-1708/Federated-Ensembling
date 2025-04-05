import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_neural_network_model(config):
    """
    Create a neural network model with the given configuration
    
    Args:
        config: Dictionary containing model hyperparameters
        
    Returns:
        Compiled Tensorflow/Keras model
    """
    # Set random seed for TensorFlow
    tf.random.set_seed(config['random_state'])
    
    # Extract hyperparameters
    hidden_layers = config['hidden_layers']
    activation = config['activation']
    
    # Create sequential model
    model = Sequential()
    
    # Add input layer (with placeholder for input shape that will be set during training)
    input_dim = config.get('input_dim', 14)  # fallback to 14 if not specified
    model.add(Dense(hidden_layers[0], activation=activation, input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    
    # Add hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(0.2))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer=config['optimizer'],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fit(model, X, y, **kwargs):
    """
    Custom fit method to handle input shape properly
    
    Args:
        model: Keras model
        X: Input features
        y: Target values
        **kwargs: Additional arguments for model.fit
        
    Returns:
        Training history
    """
    # Convert inputs to numpy arrays if they're not already
    X = np.array(X)
    y = np.array(y)
    
    # Update input shape if needed
    if model.input_shape[1] is None:
        input_shape = X.shape[1]
        
        # Recreate first layer with proper input shape
        config = model.layers[0].get_config()
        config['batch_input_shape'] = (None, input_shape)
        
        # Replace the layer
        hidden_layer = Dense.from_config(config)
        model.layers[0] = hidden_layer
    
    # Fit the model
    history = model.fit(X, y, **kwargs)
    
    return history