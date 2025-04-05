import argparse
import logging
import numpy as np
import os
import random
import time
from sklearn.metrics import accuracy_score, classification_report

from client import Client
from server import Server
from config import Config
from data.prepare_data import load_test_data

def setup_logging(log_level):
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Ensemble Learning')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds')
    parser.add_argument('--clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--models', type=int, default=3, help='Number of models')
    parser.add_argument('--data_path', type=str, default='data/adult.csv', help='Path to dataset')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def main():
    """Main function to run the federated ensemble learning simulation"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    logger = logging.getLogger('main')
    
    logger.info("Starting Federated Ensemble Learning simulation")
    logger.info(f"Configuration: {args}")
    
    # Create config
    config = Config(
        rounds=args.rounds,
        num_clients=args.clients,
        num_models=args.models,
        data_path=args.data_path
    )
    
    # Initialize server
    server = Server(config)
    
    # Initialize clients
    clients = []
    for i in range(config.num_clients):
        client = Client(client_id=i, config=config)
        clients.append(client)
        logger.info(f"Initialized Client {i} with {client.X_train.shape[0]} samples")
    
    # Federated training
    for t in range(config.rounds):
        logger.info(f"Round {t+1}/{config.rounds}")
        
        # For each client
        for i, client in enumerate(clients):
            # Get model from server based on permutation
            model_idx = server.get_model_index_for_client(i, t)
            model = server.get_model(model_idx)
            
            # Client trains the model locally
            trained_model = client.train(model, model_idx)
            
            # Client sends updated model to server
            server.receive_model(i, model_idx, trained_model)
        
        # Server updates global models
        server.update_global_models()
        
        # Evaluate current performance
        if (t+1) % config.eval_interval == 0:
            server.evaluate()
    
    # Final evaluation on test set
    logger.info("Training completed. Evaluating final models on test set...")
    X_test_raw, y_test = load_test_data()
    
    # Handle missing values before preprocessing
    X_test_raw = X_test_raw.dropna()
    y_test = y_test.loc[X_test_raw.index]
    
    # Preprocess the test data
    from data.prepare_data import preprocess_data
    X_test, _ = preprocess_data(X_test_raw)
    
    # Evaluate individual models
    for i, model in enumerate(server.global_models):
        y_pred = model.predict(X_test)
        
        # Convert neural network probabilities to binary predictions
        if i == 2:  # Neural Network
            y_pred = (y_pred > 0.5).astype(int).flatten()
            
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model {i} test accuracy: {accuracy:.4f}")
    
    # Evaluate ensemble
    ensemble_pred = server.ensemble_predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    logger.info(f"Ensemble test accuracy: {ensemble_accuracy:.4f}")
    logger.info("\nClassification Report (Ensemble):")
    logger.info(classification_report(y_test, ensemble_pred))
    
    # Save results
    server.save_results()
    
    logger.info("Federated Ensemble Learning simulation completed")

if __name__ == "__main__":
    main()