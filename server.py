import logging
import numpy as np
import os
import json
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from models.random_forest import create_random_forest_model
from models.gradient_boosting import create_gradient_boosting_model
from models.neural_network import create_neural_network_model
from data.prepare_data import load_test_data, preprocess_data
from ensemble import weighted_voting_ensemble

class Server:
    """Central server in the federated ensemble learning system"""
    
    def __init__(self, config):
        """
        Initialize the server
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('server')
        
        # Initialize global models
        self.global_models = self._initialize_models()
        self.logger.info(f"Initialized {len(self.global_models)} global models")
        
        # Store received models from clients
        self.received_models = {i: [] for i in range(config.num_models)}
        
        # Track performance metrics
        self.performance_history = []
        
        # Create model permutations for each client
        self.model_permutations = self._create_model_permutations()
        
        # Create results directory if it doesn't exist
        if not os.path.exists(config.results_dir):
            os.makedirs(config.results_dir)
        
        # Load test data for evaluation
        X_test_raw, self.y_test = load_test_data()
        
        # Handle missing values and preprocess test data
        X_test_raw = X_test_raw.dropna()
        self.y_test = self.y_test.loc[X_test_raw.index]
        
        self.X_test, _ = preprocess_data(X_test_raw)
        
        # Ensemble weights (will be updated during training)
        self.ensemble_weights = np.ones(config.num_models) / config.num_models
    
    def _initialize_models(self):
        """Initialize the global models"""
        models = []
        
        # Model 0: Random Forest
        rf_model = create_random_forest_model(self.config.model_configs[0])
        models.append(rf_model)
        
        # Model 1: Gradient Boosting
        gb_model = create_gradient_boosting_model(self.config.model_configs[1])
        models.append(gb_model)
        
        # Model 2: Neural Network
        nn_model = create_neural_network_model(self.config.model_configs[2])
        models.append(nn_model)
        
        return models
    
    def _create_model_permutations(self):
        """Create model permutations for each client"""
        permutations = {}
        
        for t in range(self.config.rounds):
            for i in range(self.config.num_clients):
                if t % self.config.num_models == 0:
                    # Create a new shuffled permutation
                    perm = np.random.permutation(self.config.num_models)
                    permutations[(i, t)] = perm
                else:
                    # Use the same permutation as in the first round of this cycle
                    first_round_in_cycle = t - (t % self.config.num_models)
                    permutations[(i, t)] = permutations[(i, first_round_in_cycle)]
        
        return permutations
    
    def get_model_index_for_client(self, client_id, round_num):
        """
        Get the model index for a specific client in a specific round
        
        Args:
            client_id: ID of the client
            round_num: Current round number
            
        Returns:
            Model index assigned to this client
        """
        permutation = self.model_permutations[(client_id, round_num)]
        model_idx = permutation[round_num % self.config.num_models]
        
        # Add more detailed logging with model names for dashboard tracking
        model_names = ["Random Forest", "Gradient Boosting", "Neural Network"]
        self.logger.info(f"Assigning {model_names[model_idx]} (model_idx: {model_idx}) to Client {client_id} in round {round_num+1}")
        
        return model_idx
    
    def get_model(self, model_idx):
        """
        Get a model by index
        
        Args:
            model_idx: Index of the model
            
        Returns:
            The model
        """
        model_names = ["Random Forest", "Gradient Boosting", "Neural Network"]
        self.logger.info(f"Sending {model_names[model_idx]} to client for training")
        return self.global_models[model_idx]
    
    def receive_model(self, client_id, model_idx, model):
        """
        Receive a trained model from a client
        
        Args:
            client_id: ID of the client
            model_idx: Index of the model
            model: Trained model
        """
        model_names = ["Random Forest", "Gradient Boosting", "Neural Network"]
        self.logger.info(f"Received {model_names[model_idx]} from Client {client_id}")
        self.received_models[model_idx].append((client_id, model))
    
    def update_global_models(self):
        """Update the global models based on received models from clients"""
        self.logger.info("Server is updating global models with received updates")
        
        model_names = ["Random Forest", "Gradient Boosting", "Neural Network"]
        
        for model_idx in range(self.config.num_models):
            if not self.received_models[model_idx]:
                self.logger.warning(f"No updates received for {model_names[model_idx]} (model_idx: {model_idx})")
                continue
            
            clients_str = ', '.join([str(client_id) for client_id, _ in self.received_models[model_idx]])
            self.logger.info(f"Aggregating {model_names[model_idx]} updates from clients: {clients_str}")
            
            # For neural networks, we need to average weights
            if model_idx == 2:
                # Extract weights from all received models
                all_weights = [model.get_weights() for _, model in self.received_models[model_idx]]
                
                # Average the weights
                avg_weights = []
                for weights_list_tuple in zip(*all_weights):
                    avg_weights.append(np.mean(weights_list_tuple, axis=0))
                
                # Update the global model weights
                self.global_models[model_idx].set_weights(avg_weights)
                self.logger.info(f"Updated global {model_names[model_idx]} with averaged weights")
            else:
                # For sklearn models, we just use the most recently trained model
                # Note: This is a simplification, in a real system you might want to 
                # implement parameter averaging for these models as well
                client_id, latest_model = self.received_models[model_idx][-1]
                self.global_models[model_idx] = latest_model
                self.logger.info(f"Updated global {model_names[model_idx]} with model from Client {client_id}")
        
        # Clear received models for the next round
        self.received_models = {i: [] for i in range(self.config.num_models)}
        
        # Update ensemble weights based on test performance
        self._update_ensemble_weights()
    
    def _update_ensemble_weights(self):
        """Update weights for the ensemble based on model performance"""
        if self.X_test is None or self.y_test is None:
            self.logger.warning("Test data not available, using uniform weights for ensemble")
            return
        
        # Calculate accuracy for each model on the test set
        accuracies = []
        for i, model in enumerate(self.global_models):
            y_pred = model.predict(self.X_test)
            
            # Convert neural network probabilities to binary predictions
            if i == 2:  # Neural Network
                y_pred = (y_pred > 0.5).astype(int)
            
            acc = accuracy_score(self.y_test, y_pred)
            accuracies.append(acc)
        
        # Normalize accuracies to get weights (adding small epsilon to avoid division by zero)
        self.ensemble_weights = np.array(accuracies) / (sum(accuracies) + 1e-10)
        
        self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def ensemble_predict(self, X):
        """
        Make predictions using the ensemble of models
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        return weighted_voting_ensemble(self.global_models, X, self.ensemble_weights)
    
    def evaluate(self):
        """Evaluate current model performance"""
        if self.X_test is None or self.y_test is None:
            self.logger.warning("Test data not available, skipping evaluation")
            return
        
        results = {}
        
        # Evaluate individual models
        for i, model in enumerate(self.global_models):
            y_pred = model.predict(self.X_test)
            
            # Convert neural network probabilities to binary predictions
            if i == 2:  # Neural Network
                y_pred = (y_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            results[f'model_{i}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            self.logger.info(f"Model {i} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Evaluate ensemble
        ensemble_pred = self.ensemble_predict(self.X_test)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
        ensemble_precision = precision_score(self.y_test, ensemble_pred, zero_division=0)
        ensemble_recall = recall_score(self.y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(self.y_test, ensemble_pred, zero_division=0)
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'f1': ensemble_f1,
            'weights': self.ensemble_weights.tolist()
        }
        
        self.logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
        
        # Store results
        self.performance_history.append(results)
    
    def save_results(self):
        """Save results and models to disk"""
        timestamp = int(time.time())
        
        # Save performance history
        results_path = os.path.join(self.config.results_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
        
        # Create directory for models if it doesn't exist
        if not os.path.exists(self.config.models_dir):
            os.makedirs(self.config.models_dir)
        
        # Save models
        for i, model in enumerate(self.global_models):
            if i == 2:  # Neural Network
                model_path = os.path.join(self.config.models_dir, f'model_{i}_{timestamp}.h5')
                model.save(model_path)
            else:  # Sklearn models
                model_path = os.path.join(self.config.models_dir, f'model_{i}_{timestamp}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            self.logger.info(f"Model {i} saved to {model_path}")
        
        # Save ensemble weights
        weights_path = os.path.join(self.config.models_dir, f'ensemble_weights_{timestamp}.npy')
        np.save(weights_path, self.ensemble_weights)
        self.logger.info(f"Ensemble weights saved to {weights_path}")
        
        # Generate and save plots
        self._save_performance_plots(timestamp)
    
    def _save_performance_plots(self, timestamp):
        """Generate and save performance plots"""
        if not self.performance_history:
            return
        
        # Extract data for plotting
        rounds = list(range(1, len(self.performance_history) + 1))
        
        # Model accuracies over rounds
        plt.figure(figsize=(10, 6))
        for i in range(self.config.num_models):
            accuracies = [round_data[f'model_{i}']['accuracy'] for round_data in self.performance_history]
            plt.plot(rounds, accuracies, marker='o', label=f'Model {i}')
        
        # Ensemble accuracy
        ensemble_accuracies = [round_data['ensemble']['accuracy'] for round_data in self.performance_history]
        plt.plot(rounds, ensemble_accuracies, marker='s', linestyle='--', linewidth=2, color='black', label='Ensemble')
        
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Over Rounds')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(self.config.results_dir, f'accuracy_plot_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Performance plot saved to {plot_path}")
        
        # Plot ensemble weights over rounds
        plt.figure(figsize=(10, 6))
        for i in range(self.config.num_models):
            weights = [round_data['ensemble']['weights'][i] for round_data in self.performance_history]
            plt.plot(rounds, weights, marker='o', label=f'Model {i} Weight')
        
        plt.xlabel('Round')
        plt.ylabel('Weight')
        plt.title('Ensemble Weights Over Rounds')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        weights_plot_path = os.path.join(self.config.results_dir, f'weights_plot_{timestamp}.png')
        plt.savefig(weights_plot_path)
        plt.close()
        self.logger.info(f"Weights plot saved to {weights_plot_path}")