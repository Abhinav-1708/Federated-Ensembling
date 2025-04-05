import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_preparation')

# Global preprocessor
_PREPROCESSOR = None

def get_preprocessor():
    """Get the global preprocessor, creating it if it doesn't exist"""
    global _PREPROCESSOR
    
    if _PREPROCESSOR is None:
        # Check if preprocessor is saved to disk
        if os.path.exists('data/preprocessor.pkl'):
            with open('data/preprocessor.pkl', 'rb') as f:
                _PREPROCESSOR = pickle.load(f)
            logger.info("Loaded preprocessor from data/preprocessor.pkl")
        else:
            # Create and fit preprocessor on full dataset
            df = pd.read_csv('data/adult.csv')
            _PREPROCESSOR = create_and_fit_preprocessor(df)
            
            # Save preprocessor to disk
            with open('data/preprocessor.pkl', 'wb') as f:
                pickle.dump(_PREPROCESSOR, f)
            logger.info("Created new preprocessor and saved to data/preprocessor.pkl")
    
    return _PREPROCESSOR

def create_and_fit_preprocessor(df):
    """Create and fit a preprocessor on the given dataframe"""
    # Handle missing values
    df = df.dropna()
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # If 'income' is in numerical columns, remove it as it's the target
    if 'income' in numerical_cols:
        numerical_cols = numerical_cols.drop('income')
    
    # Create preprocessing pipelines for both numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Extract features
    X = df.drop('income', axis=1) if 'income' in df.columns else df
    
    # Fit the preprocessor
    preprocessor.fit(X)
    
    return preprocessor

def preprocess_data(df):
    """Preprocess the data using the global preprocessor"""
    # Ensure we have a preprocessor
    preprocessor = get_preprocessor()
    
    # Handle missing values
    df = df.dropna()
    
    # Extract features and target if 'income' is in df
    if 'income' in df.columns:
        X = df.drop('income', axis=1)
        y = df['income']
    else:
        X = df
        y = None
    
    # Transform the data
    X_processed = preprocessor.transform(X)
    
    # Convert sparse matrix to dense for TensorFlow compatibility
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()
    
    return X_processed, y

def download_data():
    """Download the Adult Census Income dataset if not already available"""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if not os.path.exists('data/adult.csv'):
        logger.info("Downloading Adult Census Income dataset...")
        # URL for the Adult dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        
        # Column names for the Adult dataset
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        
        # Download and save the dataset
        df = pd.read_csv(url, header=None, names=column_names, na_values='?', skipinitialspace=True)
        
        # Process the income column to binary (>50K = 1, <=50K = 0)
        df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
        
        # Save the processed dataset
        df.to_csv('data/adult.csv', index=False)
        logger.info("Dataset downloaded and saved to data/adult.csv")
    else:
        logger.info("Dataset already exists at data/adult.csv")

def split_data_by_age(df, num_clients=3):
    """Split data by age groups to simulate non-IID data distribution across clients"""
    # Sort data by age
    df_sorted = df.sort_values(by='age')
    
    # Calculate split points to divide into num_clients segments
    split_indices = [int(i * len(df_sorted) / num_clients) for i in range(1, num_clients)]
    
    # Split dataframe into num_clients parts
    client_dfs = np.split(df_sorted, split_indices)
    
    client_data = []
    for i, client_df in enumerate(client_dfs):
        logger.info(f"Client {i} data: {len(client_df)} samples, age range: {client_df['age'].min()}-{client_df['age'].max()}")
        client_data.append(client_df)
    
    return client_data

def prepare_client_data(num_clients=3):
    """Prepare and split data for clients and create test set"""
    # Download data if not available
    download_data()
    
    # Load the dataset
    df = pd.read_csv('data/adult.csv')
    logger.info(f"Loaded dataset with {len(df)} samples and {df.shape[1]} features")
    
    # Split into train and test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train set: {len(df_train)} samples, Test set: {len(df_test)} samples")
    
    # Split training data by age for clients (non-IID)
    client_dfs = split_data_by_age(df_train, num_clients)
    
    # Save client data
    if not os.path.exists('data/clients'):
        os.makedirs('data/clients')
    
    # Save each client's data
    for i, client_df in enumerate(client_dfs):
        client_df.to_csv(f'data/clients/client_{i}.csv', index=False)
    
    # Save test data
    df_test.to_csv('data/test.csv', index=False)
    
    logger.info("Client data preparation completed.")
    return client_dfs, df_test

def load_client_data(client_id):
    """Load data for a specific client"""
    client_path = f'data/clients/client_{client_id}.csv'
    
    if not os.path.exists(client_path):
        logger.error(f"Client data file {client_path} not found. Run prepare_client_data first.")
        return None, None
    
    df = pd.read_csv(client_path)
    
    # Split into features and target
    X = df.drop('income', axis=1)
    y = df['income']
    
    # Further split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    return (X_train, y_train), (X_val, y_val)

def load_test_data():
    """Load the test dataset"""
    test_path = 'data/test.csv'
    
    if not os.path.exists(test_path):
        logger.error(f"Test data file {test_path} not found. Run prepare_client_data first.")
        return None, None
    
    df_test = pd.read_csv(test_path)
    
    # Split into features and target
    X_test = df_test.drop('income', axis=1)
    y_test = df_test['income']
    
    return X_test, y_test

if __name__ == "__main__":
    # If run directly, prepare the data
    client_dfs, test_df = prepare_client_data(num_clients=3)
    logger.info("Data preparation complete")