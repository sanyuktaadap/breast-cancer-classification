import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def prepare_data(train_data_path, test_data_path):
    """Prepares data as follows
    1. Read data
    2. Remove columns - 'Unnamed: 32' and 'id'
    3. Separate labels and features
    4. Convert labels to 0 and 1 for B and M respectively
    5. Normalize train_data and test_data otb of train_data
    
    Args:
        train_data_path : Training dataset location
        test_data_path : Test dataset location

    Returns:
        Train and test features and labels.
    """
    
    # 1. Read data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)  
    
    # 2. Remove unwanted columns
    columns_to_remove = ["id", "Unnamed: 32"]
    train_data = train_data.drop(columns_to_remove, axis=1)
    test_data = test_data.drop(columns_to_remove, axis=1)
    
    # 3. Separate labels and features
    train_features = train_data.drop('diagnosis', axis=1)
    test_features = test_data.drop('diagnosis', axis=1)
    
    train_labels = train_data['diagnosis']
    test_labels = test_data['diagnosis']
    
    # 4. Label modification
    train_labels.replace(['B', 'M'], [0, 1], inplace=True)
    test_labels.replace(['B', 'M'], [0, 1], inplace=True)
    
    # 5. Data normalization
    scaler = preprocessing.MinMaxScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    
    return train_features, train_labels, test_features, test_labels

def prep_pca_data(train_features, test_features):
    """Perform PCA on train_features and generate new features
    For both training and testing
    """
    pca_30 = PCA(n_components=30)
    train_pca_features = pca_30.fit_transform(train_features)
    
    # pca_30 has been fit on train data already
    test_pca_features = pca_30.transform(test_features)
    
    return train_pca_features, test_pca_features