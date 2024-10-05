from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from data_preprocessing import load_data
from utils import evaluate_model

def train(X_train, y_train):
    dtc = DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    xgb = XGBClassifier()
    gc = GradientBoostingClassifier(random_state=42)
    svc = SVC(kernel='rbf', random_state=42)
    ad = AdaBoostClassifier(random_state=42)

    estimators = [('dtc', dtc), ('rfc', rfc), ('knn', knn), ('gc', gc), ('svc', svc), ('ad', ad)]
    stacking = StackingClassifier(estimators=estimators, final_estimator=xgb)
    stacking.fit(X_train, y_train)
    
    return stacking

def run(is_scaled=False):
    X_train, X_test, y_train, y_test = load_data(is_scaled=is_scaled)
    
    stacking = train(X_train, y_train)
    train_acc, test_acc, cm = evaluate_model(stacking, (X_train, y_train, X_test, y_test))
    print(f'Stacking Training Accuracy: {train_acc:.2f}')
    print(f'Stacking Test Accuracy: {test_acc:.2f}')
    print(f'Confusion Matrix: \n{cm}\n')
    
    return train_acc, test_acc

if __name__ == '__main__':
    run(False)
    run(True)
    