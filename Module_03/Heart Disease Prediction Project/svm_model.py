from sklearn.svm import SVC
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    return svm

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    svm = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(svm, (X_train, y_train, X_test, y_test))
    print(f'SVM Training Accuracy: {train_acc:.2f}')
    print(f'SVM Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)