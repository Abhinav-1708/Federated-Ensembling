import numpy as np
from sklearn.linear_model import LogisticRegression

def majority_voting_ensemble(models, X):
    """
    Simple majority voting ensemble
    
    Args:
        models: List of trained models
        X: Input features
        
    Returns:
        Ensemble predictions
    """
    # Get predictions from each model
    predictions = []
    for model in models:
        y_pred = model.predict(X)
        predictions.append(y_pred)
    
    # Stack predictions and compute majority vote (mode along axis 0)
    predictions = np.array(predictions)
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=predictions
    )
    
    return ensemble_pred

def weighted_voting_ensemble(models, X, weights=None):
    """
    Weighted voting ensemble
    
    Args:
        models: List of trained models
        X: Input features
        weights: List of weights for each model (default: equal weights)
        
    Returns:
        Ensemble predictions
    """
    if weights is None:
        # Equal weights
        weights = np.ones(len(models)) / len(models)
    
    # Get weighted predictions
    # Use shape[0] instead of len() to handle sparse matrices
    n_samples = X.shape[0]
    weighted_sum = np.zeros(n_samples)
    
    for i, model in enumerate(models):
        # Special handling for neural network (model index 2)
        if i == 2:  # Neural Network
            probs = model.predict(X)
            # Convert to 1D array if necessary and ensure binary values
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                weighted_sum += weights[i] * probs[:, 1]  # Use second column if multiple outputs
            else:
                weighted_sum += weights[i] * probs.flatten()  # Flatten if needed
        # For probability-based models (returns probability of class 1)
        elif hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X)
                if probs.shape[1] > 1:  # Binary classification
                    weighted_sum += weights[i] * probs[:, 1]  # Probability of class 1
                else:  # Single column output
                    weighted_sum += weights[i] * probs[:, 0]
            except:
                # Fallback to regular predictions
                preds = model.predict(X).astype(float)
                weighted_sum += weights[i] * preds
        else:
            # For models without probability output
            preds = model.predict(X).astype(float)
            weighted_sum += weights[i] * preds
    
    # Convert to binary predictions
    ensemble_pred = (weighted_sum >= 0.5).astype(int)
    
    return ensemble_pred

def stacking_ensemble(models, X_train, y_train, X):
    """
    Stacking ensemble using logistic regression as meta-learner
    
    Args:
        models: List of trained models
        X_train: Training features for meta-learner
        y_train: Training labels for meta-learner
        X: Input features for prediction
        
    Returns:
        Ensemble predictions
    """
    # Generate base predictions for training
    base_preds_train = np.column_stack([model.predict(X_train) for model in models])
    
    # Train meta-learner (logistic regression)
    meta_learner = LogisticRegression(random_state=42)
    meta_learner.fit(base_preds_train, y_train)
    
    # Generate base predictions for test data
    base_preds_test = np.column_stack([model.predict(X) for model in models])
    
    # Make final predictions
    ensemble_pred = meta_learner.predict(base_preds_test)
    
    return ensemble_pred