import pandas as pd
import numpy as np
import os

def load_data(dataset_path, selected_labels=[0,1,2,3,4,5]):
    train_df = pd.read_csv(os.path.join(dataset_path, 'sign_mnist_train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'sign_mnist_test.csv'))

    train_df = train_df[train_df['label'].isin(selected_labels)]
    test_df = test_df[test_df['label'].isin(selected_labels)]

    X_train = train_df.drop('label', axis=1).values / 255.0
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values / 255.0
    y_test = test_df['label'].values

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    return X_train, y_train, X_test, y_test
