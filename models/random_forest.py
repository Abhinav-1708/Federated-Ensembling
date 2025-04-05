from sklearn.ensemble import RandomForestClassifier

def create_random_forest_model(config):
    """
    Create a random forest model with the given configuration
    
    Args:
        config: Dictionary containing model hyperparameters
        
    Returns:
        Initialized RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        random_state=config['random_state'],
        n_jobs=-1  # Use all available cores
    )
    
    return model