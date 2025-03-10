from sklearn.ensemble import AdaBoostClassifier
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    ada = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ada.fit(X_train, y_train)
    return ada

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    ada = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(ada, (X_train, y_train, X_test, y_test))
    print(f'AdaBoost Training Accuracy: {train_acc:.2f}')
    print(f'AdaBoost Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)
    
