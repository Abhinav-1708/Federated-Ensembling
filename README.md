# Federated Ensemble Learning

This project implements a federated ensemble learning system that combines multiple machine learning models trained across decentralized clients. The system simulates a federated learning environment where models are trained locally on client devices and aggregated on a central server.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Components](#components)
   - [Configuration](#configuration)
   - [Server](#server)
   - [Client](#client)
   - [Models](#models)
   - [Data Preparation](#data-preparation)
   - [Ensemble Methods](#ensemble-methods)
4. [Model Updating Process](#model-updating-process)
5. [Ensemble Process](#ensemble-process)
6. [Web Dashboard](#web-dashboard)
7. [Example](#example)
8. [Usage](#usage)

## Overview

Federated learning allows training machine learning models across multiple devices or servers without exchanging the local data, addressing privacy and data locality issues. This implementation extends the federated learning paradigm by incorporating ensemble learning, which combines predictions from multiple models to improve overall performance and robustness.

The system uses three different models:
1. Random Forest
2. Gradient Boosting
3. Neural Network

Each client trains these models on their local data, and the server aggregates the updates to create a global ensemble.

## System Architecture

The system has a client-server architecture where:

- The **server** coordinates training, maintains global models, and performs ensembling
- **Clients** train models on local data
- Models are rotated among clients to ensure each model is trained on different data distributions

The overall workflow is:
1. Server initializes global models
2. For each round:
   - Server assigns models to clients
   - Clients train the models locally
   - Clients send updated models to the server
   - Server aggregates updates and updates global models
   - Server updates ensemble weights based on validation performance
3. Final evaluation is performed on a test set

## Components

### Configuration

The `Config` class (in `config.py`) centralizes all configuration parameters:

```python
config = Config(
    rounds=10,          # Number of federated rounds
    num_clients=3,      # Number of clients participating
    num_models=3,       # Number of models in the ensemble
    data_path='data/adult.csv'  # Path to the dataset
)
```

Key configuration components:
- General settings (rounds, clients, etc.)
- Data settings (test size, validation size)
- Model hyperparameters for each model type
- Client training settings (local epochs, batch size)
- Ensemble settings (method, weighting strategy)

### Server

The `Server` class (in `server.py`) is responsible for:

#### Key Functions:

- `__init__(config)`: Initializes the server with the given configuration, loads test data, and creates initial models
- `_initialize_models()`: Creates and initializes the three model types
- `_create_model_permutations()`: Creates a schedule for assigning models to clients in each round
- `get_model_index_for_client(client_id, round_num)`: Determines which model a client should train in a specific round
- `get_model(model_idx)`: Retrieves a model by its index
- `receive_model(client_id, model_idx, model)`: Receives a trained model from a client
- `update_global_models()`: Aggregates received models to update the global models
- `_update_ensemble_weights()`: Updates weights for ensemble based on model performance on the test set
- `ensemble_predict(X)`: Makes predictions using the weighted ensemble
- `evaluate()`: Evaluates model and ensemble performance
- `save_results()`: Saves results, models, and performance plots

### Client

The `Client` class (in `client.py`) represents a federated learning participant:

#### Key Functions:

- `__init__(client_id, config)`: Initializes the client with the given ID and configuration, loads and preprocesses local data
- `train(model, model_idx)`: Trains the given model on local data and returns the updated model
- `evaluate(models, weights)`: Evaluates models on client's validation data

### Models

Three model types are implemented in separate modules:

1. **Random Forest** (`models/random_forest.py`):
   - `create_random_forest_model(config)`: Creates a RandomForestClassifier with specified hyperparameters

2. **Gradient Boosting** (`models/gradient_boosting.py`):
   - `create_gradient_boosting_model(config)`: Creates a GradientBoostingClassifier with specified hyperparameters

3. **Neural Network** (`models/neural_network.py`):
   - `create_neural_network_model(config)`: Creates a TensorFlow/Keras sequential neural network

### Data Preparation

The `data/prepare_data.py` module handles data loading, preprocessing, and splitting:

#### Key Functions:

- `get_preprocessor()`: Retrieves or creates a global preprocessor for consistent feature transformation
- `create_and_fit_preprocessor(df)`: Creates and fits a preprocessor on the given dataframe
- `preprocess_data(df)`: Preprocesses data using the global preprocessor
- `download_data()`: Downloads the Adult Census Income dataset if not already available
- `split_data_by_age(df, num_clients)`: Splits data by age groups to simulate non-IID data distribution
- `prepare_client_data(num_clients)`: Prepares and splits data for clients and creates a test set
- `load_client_data(client_id)`: Loads data for a specific client
- `load_test_data()`: Loads the test dataset

### Ensemble Methods

The `ensemble.py` module implements different ensemble methods:

#### Key Functions:

- `majority_voting_ensemble(models, X)`: Simple majority voting ensemble
- `weighted_voting_ensemble(models, X, weights)`: Weighted voting ensemble based on model performance
- `stacking_ensemble(models, X_train, y_train, X)`: Stacking ensemble using logistic regression as meta-learner

## Model Updating Process

The model updating process works differently for different model types:

1. **For neural networks (model_idx == 2)**:
   - The server extracts weights from all received models
   - It computes the average weights across all clients who trained this model
   - The global neural network model is updated with these averaged weights
   
   ```python
   # Extract weights from all received models
   all_weights = [model.get_weights() for _, model in self.received_models[model_idx]]
   
   # Average the weights
   avg_weights = []
   for weights_list_tuple in zip(*all_weights):
       avg_weights.append(np.mean(weights_list_tuple, axis=0))
   
   # Update the global model weights
   self.global_models[model_idx].set_weights(avg_weights)
   ```

2. **For sklearn models (Random Forest and Gradient Boosting)**:
   - The server simply uses the most recently trained model
   - This is a simplification; in practice, parameter averaging could be implemented for these models as well
   
   ```python
   # For sklearn models, we just use the most recently trained model
   _, latest_model = self.received_models[model_idx][-1]
   self.global_models[model_idx] = latest_model
   ```

The server updates the ensemble weights after updating the global models based on individual model performance on the test set.

## Ensemble Process

The system uses a weighted voting ensemble by default. The process works as follows:

1. **Weight calculation**:
   - Each model's performance (accuracy) on the test set is measured
   - Weights are normalized so they sum to 1
   
   ```python
   # Calculate accuracy for each model on the test set
   accuracies = []
   for i, model in enumerate(self.global_models):
       y_pred = model.predict(self.X_test)
       
       # Convert neural network probabilities to binary predictions
       if i == 2:  # Neural Network
           y_pred = (y_pred > 0.5).astype(int)
       
       acc = accuracy_score(self.y_test, y_pred)
       accuracies.append(acc)
   
   # Normalize accuracies to get weights
   self.ensemble_weights = np.array(accuracies) / (sum(accuracies) + 1e-10)
   ```

2. **Weighted prediction**:
   - For each model in the ensemble:
     - Get predictions for the input data
     - Multiply the predictions by the model's weight
     - Add them to a weighted sum
   - Apply a threshold (0.5) to get the final binary predictions
   
   ```python
   # Get weighted predictions
   n_samples = X.shape[0]
   weighted_sum = np.zeros(n_samples)
   
   for i, model in enumerate(models):
       # Model-specific handling...
       preds = model.predict(X).astype(float)
       weighted_sum += weights[i] * preds
   
   # Convert to binary predictions
   ensemble_pred = (weighted_sum >= 0.5).astype(int)
   ```

The system handles different prediction formats:
- Neural networks output probabilities that need to be thresholded
- Models with `predict_proba` method use the probability of class 1
- Other models use standard predictions

## Web Dashboard

The project includes a web-based dashboard to visualize and explore the results of federated ensemble learning runs without having to navigate to the image files manually.

### Dashboard Features

- **Main Dashboard Page**:
  - Lists all available runs
  - Provides quick access to run results
  - Shows basic system information

- **Run Results Page**:
  - Displays performance plots over training rounds
  - Shows ensemble weights evolution
  - Presents final metrics for all models and the ensemble
  - Provides detailed information about the final model weights

### Running the Dashboard

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the federated learning simulation to generate results:
   ```bash
   python main.py --rounds 10 --clients 3 --models 3
   ```

3. Start the dashboard:
   ```bash
   python dashboard.py
   ```

4. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Example

Here's a simplified example of how the system works with 3 clients, 3 models, and 2 rounds:

### Round 1:
1. **Model Assignment**:
   - Client 0 is assigned Model 0 (Random Forest)
   - Client 1 is assigned Model 1 (Gradient Boosting)
   - Client 2 is assigned Model 0 (Random Forest) again

2. **Training**:
   - Client 0 trains Model 0 on its local data
   - Client 1 trains Model 1 on its local data
   - Client 2 trains Model 0 on its local data

3. **Model Update**:
   - Server receives Model 0 from Client 0 and Client 2
     - Uses the most recent version (from Client 2)
   - Server receives Model 1 from Client 1
     - Updates Model 1 accordingly
   - No updates for Model 2 (Neural Network)

4. **Weight Update**:
   - Server evaluates all models on the test set
   - Calculates accuracies: [0.85, 0.86, 0.41]
   - Updates weights: [0.40, 0.41, 0.19]

### Round 2:
1. **Model Assignment**:
   - Client 0 is assigned Model 1 (Gradient Boosting)
   - Client 1 is assigned Model 2 (Neural Network)
   - Client 2 is assigned Model 1 (Gradient Boosting) again

2. **Training and Updates**: (similar process)

3. **Final Ensemble**:
   - Each model makes predictions on the test set
   - Predictions are weighted by the final weights
   - Ensemble achieves 0.86 accuracy, better than any individual model

## Usage

Run the system with default parameters:
```bash
python main.py
```

Customize the run with command-line arguments:
```bash
python main.py --rounds 10 --clients 3 --models 3 --data_path 'data/adult.csv' --log_level 'INFO' --seed 42
```

The system will:
1. Download and prepare the Adult Census Income dataset if not already available
2. Split the data among clients based on age (non-IID distribution)
3. Run the specified number of federated rounds
4. Evaluate and save the final models and ensemble
5. Generate performance plots in the `results` directory

View the results in the web dashboard:
```bash
python dashboard.py
```
Then open http://localhost:5000 in your browser. 