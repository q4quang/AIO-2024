from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import load_data
from utils import evaluate_model, store_model_performance

def train(X_train, y_train):
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',     
    )
    
    knn.fit(X_train, y_train)
    return knn

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)

    knn = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(knn, (X_train, y_train, X_test, y_test))
    print(f'KNN Training Accuracy: {train_acc:.2f}')
    print(f'KNN Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix:\n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)
    

