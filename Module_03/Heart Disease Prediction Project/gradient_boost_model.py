from sklearn.ensemble import GradientBoostingClassifier
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    gb = GradientBoostingClassifier(
        n_estimators=100,
        subsample=1.0,
        min_samples_split=2,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    return gb

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    gb = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(gb, (X_train, y_train, X_test, y_test))
    print(f'Gradient Boost Training Accuracy: {train_acc:.2f}')
    print(f'Gradient Boost Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc
    
if __name__ == '__main__':
    run(False)
    run(True)