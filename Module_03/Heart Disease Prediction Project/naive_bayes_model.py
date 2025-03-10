from sklearn.naive_bayes import GaussianNB
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    nb = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(nb, (X_train, y_train, X_test, y_test))
    print(f'Naive Bayes Training Accuracy: {train_acc:.2f}')
    print(f'Naive Bayes Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)