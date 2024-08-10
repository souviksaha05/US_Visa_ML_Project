import os
import sys

import numpy as np 
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 

from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_clf(y_true, y_pred):
    """Evaluate classifier performance."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, f1, precision, recall, roc_auc

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    '''
    Evaluate models using RandomizedSearchCV for hyperparameter tuning.

    Args:
    - X_train: Training feature data.
    - y_train: Training target data.
    - X_test: Testing feature data.
    - y_test: Testing target data.
    - models: Dictionary of models to evaluate.
    - param_grids: Dictionary of parameter grids for each model.

    Returns:
    - A DataFrame containing the performance metrics of each model.
    '''
    try:
        models_list = []
        accuracy_list = []
        auc_list = []

        for model_name, model in models.items():
            print(f"Running RandomizedSearchCV for {model_name}...")
            
            # Use RandomizedSearchCV for hyperparameter tuning
            randomized_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[model_name],
                n_iter=100,
                cv=5,
                verbose=2,
                n_jobs=-1,
                scoring='accuracy'
            )
            randomized_search.fit(X_train, y_train)  # Fit randomized search to the training data
            
            best_model = randomized_search.best_estimator_  # Get the best model after hyperparameter tuning
            best_model.fit(X_train, y_train)  # Train the best model
            
            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Evaluate model performance
            model_train_accuracy, model_train_f1, model_train_precision, model_train_recall, model_train_rocauc_score = evaluate_clf(y_train, y_train_pred)
            model_test_accuracy, model_test_f1, model_test_precision, model_test_recall, model_test_rocauc_score = evaluate_clf(y_test, y_test_pred)
            
            models_list.append(model_name)
            accuracy_list.append(model_test_accuracy)
            auc_list.append(model_test_rocauc_score)
            
            print(f"Model performance for {model_name}")
            print('----------------------------------')
            print("Training set performance")
            print(f"- Accuracy: {model_train_accuracy:.4f}")
            print(f'- F1 score: {model_train_f1:.4f}')
            print(f'- Precision: {model_train_precision:.4f}')
            print(f'- Recall: {model_train_recall:.4f}')
            print(f'- ROC AUC Score: {model_train_rocauc_score:.4f}')
            
            print('----------------------------------')
            print("Test set performance")
            print(f'- Accuracy: {model_test_accuracy:.4f}')
            print(f'- F1 score: {model_test_f1:.4f}')
            print(f'- Precision: {model_test_precision:.4f}')
            print(f'- Recall: {model_test_recall:.4f}')
            print(f'- ROC AUC Score: {model_test_rocauc_score:.4f}')
            print('='*35)
            print('\n')
            
        report = pd.DataFrame(list(zip(models_list, accuracy_list, auc_list)), columns=['Model Name', 'Accuracy', 'ROC AUC']).sort_values(by=['Accuracy'], ascending=False)
            
        return report

    except Exception as e:
        raise CustomException(e, sys)


