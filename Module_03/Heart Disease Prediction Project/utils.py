from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model, data):
    X_train, y_train, X_test, y_test = data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return train_acc, test_acc, cm

def store_model_performance(model, train_acc, test_acc, filepath='../assets/model_performance.csv'):
    try:
        with open(filepath, 'a') as f:
            f.write(f'{model},{round(train_acc, 2)},{round(test_acc, 2)}\n')
        print(f'Stored accuracy for {model}!!!')
    except Exception as e:
        print(f'Failed to store data for {model}: {e}!!!')

def plot_model_performance(filepath='../assets/model_performance.csv'):
    df = pd.read_csv(filepath, header=None, names=['Model', 'Train Accuracy', 'Test Accuracy'])

    plt.figure(figsize=(10, 6))
    bar_width = 0.35

    # Plot Train Accuracy
    plt.bar(range(len(df)), 
            df['Train Accuracy'],
            width=bar_width,
            label='Train Accuracy',
            color='b',
            alpha=0.7
    )
    # Plot Test Accuracy
    plt.bar([i + bar_width for i in range(len(df))],
            df['Test Accuracy'],
            width=bar_width,
            label='Test Accuracy',
            color='r',
            alpha=0.7
    )
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.xticks([i + bar_width/2 for i in range(len(df))], df['Model'])
    plt.legend()
    
    plt.tight_layout()
    plt.show()