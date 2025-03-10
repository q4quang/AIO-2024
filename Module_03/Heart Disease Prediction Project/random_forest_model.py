from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data
from utils import evaluate_model
import numpy as np

def train(X_train, y_train):
    rf = RandomForestClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        n_estimators=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    rf = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(rf, (X_train, y_train, X_test, y_test))
    print(f'Random Forest Training Accuracy: {train_acc:.2f}')
    print(f'Random Forest Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)