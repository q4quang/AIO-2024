from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=2,
        random_state=42
    )
    dt.fit(X_train, y_train)
    return dt

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    dt = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(dt, (X_train, y_train, X_test, y_test))
    print(f'Decision Tree Training Accuracy: {train_acc:.2f}')
    print(f'Decision Tree Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)
