import logging
import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split

from data.prepare_data import load_client_data, preprocess_data

class Client:
    """Client in the federated ensemble learning system"""
    
    def __init__(self, client_id, config):
        """
        Initialize a client
        
        Args:
            client_id: ID of the client
            config: Configuration object
        """
        self.id = client_id
        self.config = config
        self.logger = logging.getLogger(f'client_{client_id}')
        
        # Load client data
        (X_train_raw, self.y_train), (X_val_raw, self.y_val) = load_client_data(client_id)
        
        # Handle missing values before preprocessing to avoid shape mismatch
        X_train_raw = X_train_raw.dropna()
        self.y_train = self.y_train.loc[X_train_raw.index]
        
        X_val_raw = X_val_raw.dropna()
        self.y_val = self.y_val.loc[X_val_raw.index]
        
        # Preprocess the data
        self.X_train, _ = preprocess_data(X_train_raw)
        self.X_val, _ = preprocess_data(X_val_raw)
        
        self.logger.info(f"Client {client_id} initialized with {self.X_train.shape[0]} training samples")
        
        # Track performance history
        self.training_history = []
        
    def train(self, model, model_idx):
        """
        Train the received model on local data
        
        Args:
            model: Model to train
            model_idx: Index of the model
            
        Returns:
            Trained model
        """
        model_names = ["Random Forest", "Gradient Boosting", "Neural Network"]
        model_name = model_names[model_idx]
        
        # Log the start of training with model details for dashboard tracking
        self.logger.info(f"Client {self.id} is now training {model_name} (model_idx: {model_idx})")
        
        # Make a deep copy of the model to train locally
        local_model = copy.deepcopy(model)
        
        # Start training timer
        start_time = time.time()
        
        # Train the model
        if model_idx == 2:  # Neural Network
            # For NN we use a different training approach
            history = local_model.fit(
                self.X_train, self.y_train,
                epochs=self.config.model_configs[model_idx]['epochs'],
                batch_size=self.config.model_configs[model_idx]['batch_size'],
                validation_data=(self.X_val, self.y_val),
                verbose=0
            )
            train_loss = history.history['loss'][-1]
            train_acc = history.history['accuracy'][-1]
            
            # Calculate validation metrics (specific for neural network)
            val_preds = local_model.predict(self.X_val)
            # Convert probabilities to binary predictions
            val_preds_binary = (val_preds > 0.5).astype(int).flatten()
            val_acc = np.mean(val_preds_binary == self.y_val.values)
        else:
            # For sklearn models
            local_model.fit(self.X_train, self.y_train)
            train_preds = local_model.predict(self.X_train)
            train_acc = np.mean(train_preds == self.y_train)
            train_loss = None  # sklearn models don't typically return loss values
            
            # Calculate validation metrics
            val_preds = local_model.predict(self.X_val)
            val_acc = np.mean(val_preds == self.y_val)
        
        # End training timer
        train_time = time.time() - start_time
        
        # Log results
        self.logger.info(f"Client {self.id} completed training {model_name} (model_idx: {model_idx}) - "
                        f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, "
                        f"Training time: {train_time:.2f}s")
        
        # Store training history
        history_entry = {
            'model_idx': model_idx,
            'model_name': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_time': train_time,
            'train_loss': train_loss
        }
        self.training_history.append(history_entry)
        
        # Sending model to server (log for UI tracking)
        self.logger.info(f"Client {self.id} sending {model_name} back to server")
        
        # Simulate communication delay
        time.sleep(self.config.communication_delay)
        
        return local_model
    
    def evaluate(self, models, weights=None):
        """
        Evaluate models on client's validation data
        
        Args:
            models: List of models to evaluate
            weights: Optional weights for ensemble
            
        Returns:
            Dict of evaluation metrics
        """
        results = {}
        
        # Evaluate individual models
        for i, model in enumerate(models):
            val_preds = model.predict(self.X_val)
            val_acc = np.mean(val_preds == self.y_val)
            results[f'model_{i}_acc'] = val_acc
        
        # Evaluate ensemble if weights are provided
        if weights is not None:
            ensemble_preds = np.zeros_like(self.y_val, dtype=float)
            
            for i, model in enumerate(models):
                model_preds = model.predict(self.X_val).astype(float)
                ensemble_preds += weights[i] * model_preds
            
            # Convert to binary predictions (threshold at 0.5)
            ensemble_preds = (ensemble_preds > 0.5).astype(int)
            ensemble_acc = np.mean(ensemble_preds == self.y_val)
            results['ensemble_acc'] = ensemble_acc
        
        return results