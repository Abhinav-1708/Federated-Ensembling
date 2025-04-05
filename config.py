class Config:
    """Configuration for the federated ensemble learning system"""
    
    def __init__(self, rounds=10, num_clients=3, num_models=3, data_path='data/adult.csv'):
        # General settings
        self.rounds = rounds
        self.num_clients = num_clients
        self.num_models = num_models
        self.data_path = data_path
        self.results_dir = 'results'
        self.models_dir = 'saved_models'
        self.eval_interval = 1  # Evaluate every N rounds
        self.seed = 42
        
        # Data settings
        self.test_size = 0.2
        self.valid_size = 0.1
        
        # Model hyperparameters
        self.model_configs = {
            # Random Forest
            0: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': self.seed
            },
            # Gradient Boosting
            1: {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': self.seed
            },
            # Neural Network
            2: {
                'hidden_layers': [64, 32, 16],
                'activation': 'relu',
                'optimizer': 'adam',
                'epochs': 5,
                'batch_size': 64,
                'validation_split': 0.1,
                'random_state': self.seed,
                'input_dim': 104  # Updated to match the preprocessed data dimensions
            }
        }
        
        # Client training settings
        self.local_epochs = 1
        self.batch_size = 32
        
        # Ensemble settings
        self.ensemble_method = 'weighted_voting'  # Options: 'majority_voting', 'weighted_voting', 'stacking'
        
        # Communication simulation
        self.communication_delay = 0.1  # Simulated delay in seconds