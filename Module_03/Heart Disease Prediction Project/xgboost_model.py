from xgboost import XGBClassifier
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    xgb = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    xgb = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(xgb, (X_train, y_train, X_test, y_test))
    print(f'XGBoost Training Accuracy: {train_acc:.2f}')
    print(f'XGBoost Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True) 


