from sklearn.ensemble import GradientBoostingClassifier

def create_gradient_boosting_model(config):
    """
    Create a gradient boosting model with the given configuration
    
    Args:
        config: Dictionary containing model hyperparameters
        
    Returns:
        Initialized GradientBoostingClassifier
    """
    model = GradientBoostingClassifier(
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        random_state=config['random_state']
    )
    
    return model